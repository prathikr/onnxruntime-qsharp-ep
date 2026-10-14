#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plugin",
        default=repository_root / "build" / "libonnxruntime_qsharp_ep.dylib",
        type=Path,
    )
    parser.add_argument(
        "--model",
        default=repository_root / "testdata" / "logistic_regression.onnx",
        type=Path,
    )
    return parser.parse_args()


def run_cpu(model: Path, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        str(model),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    return session.run(None, {"X": features})


def run_quantum(
    plugin: Path,
    model: Path,
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    registration_name = "qsharp_ep_example"
    ort.register_execution_provider_library(registration_name, str(plugin))
    try:
        devices = [
            device
            for device in ort.get_ep_devices()
            if device.ep_name == "QSharpExecutionProvider"
        ]
        if len(devices) != 1:
            raise RuntimeError(f"Expected one QSharp EP device, found {len(devices)}")

        def execute() -> tuple[np.ndarray, np.ndarray]:
            options = ort.SessionOptions()
            options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            )
            options.add_provider_for_devices(devices, {})
            session = ort.InferenceSession(str(model), sess_options=options)
            return session.run(None, {"X": features})

        return execute()
    finally:
        ort.unregister_execution_provider_library(registration_name)


def main() -> None:
    args = parse_args()
    features = np.array(
        [
            [1.0, 0.5, -0.5, 0.25],
            [-1.0, 1.0, 0.5, -0.75],
            [0.25, -0.5, 1.0, 0.5],
        ],
        dtype=np.float32,
    )

    cpu_labels, cpu_probabilities = run_cpu(args.model, features)
    quantum_labels, quantum_probabilities = run_quantum(
        args.plugin,
        args.model,
        features,
    )

    print("row  cpu_probability  quantum_probability  cpu_label  quantum_label")
    for index in range(features.shape[0]):
        print(
            f"{index:>3}  {cpu_probabilities[index, 1]:>15.6f}  "
            f"{quantum_probabilities[index, 1]:>19.6f}  "
            f"{cpu_labels[index, 0]:>9}  {quantum_labels[index, 0]:>13}"
        )


if __name__ == "__main__":
    main()
