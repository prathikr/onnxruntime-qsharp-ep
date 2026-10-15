#!/usr/bin/env python3

"""Run a hybrid CPU/Q# cat classifier over the bundled cat and dog images."""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


IMAGE_SIZE = 224
RESIZE_SHORT_EDGE = 256
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Classify the bundled cat and dog with CPU ResNet features and a Q# head."
    )
    parser.add_argument(
        "--plugin",
        type=Path,
        default=repository_root / "build" / "libonnxruntime_qsharp_ep.dylib",
    )
    parser.add_argument(
        "--resnet",
        type=Path,
        default=repository_root / "testdata" / "cat_classifier" / "resnet18.onnx",
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=repository_root / "testdata" / "cat_classifier" / "images",
    )
    return parser.parse_args()


def preprocess_image(path: Path) -> np.ndarray:
    """Apply the ImageNet resize, center-crop, and normalization pipeline."""
    with Image.open(path) as source:
        image = source.convert("RGB")

    width, height = image.size
    scale = RESIZE_SHORT_EDGE / min(width, height)
    image = image.resize(
        (round(width * scale), round(height * scale)),
        Image.Resampling.BILINEAR,
    )

    left = (image.width - IMAGE_SIZE) // 2
    top = (image.height - IMAGE_SIZE) // 2
    image = image.crop((left, top, left + IMAGE_SIZE, top + IMAGE_SIZE))

    pixels = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return (pixels - IMAGENET_MEAN) / IMAGENET_STD


def run_cpu_model(model_path: Path, images: np.ndarray) -> np.ndarray:
    """Run the complete persisted classifier on CPU as the exact reference."""
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    return session.run(None, {"pixel_values": images})[0]


def profile_uses_qsharp(profile_path: Path) -> bool:
    """Confirm that ORT assigned the head's Conv node to the QSharp EP."""
    events = json.loads(profile_path.read_text(encoding="utf-8"))
    return any(
        event.get("args", {}).get("provider") == "QSharpExecutionProvider"
        for event in events
    )


def run_quantum_model(
    plugin_path: Path,
    model_path: Path,
    images: np.ndarray,
) -> np.ndarray:
    """Run the ResNet backbone on CPU and its final 1x1 Conv with Q#."""
    registration_name = "qsharp_ep_cat_classifier"
    ort.register_execution_provider_library(registration_name, str(plugin_path))
    try:
        devices = [
            device
            for device in ort.get_ep_devices()
            if device.ep_name == "QSharpExecutionProvider"
        ]
        if len(devices) != 1:
            raise RuntimeError(f"Expected one QSharp EP device, found {len(devices)}")

        # Keeping the session in this nested scope ensures it is destroyed
        # before the plugin library is unregistered in the finally block.
        def execute() -> np.ndarray:
            with tempfile.TemporaryDirectory(prefix="qsharp-cat-profile-") as temp_dir:
                options = ort.SessionOptions()
                # ORT may otherwise fold the following Add into Conv as its
                # optional bias, outside this intentionally narrow kernel.
                options.graph_optimization_level = (
                    ort.GraphOptimizationLevel.ORT_DISABLE_ALL
                )
                options.enable_profiling = True
                options.profile_file_prefix = str(Path(temp_dir) / "profile")
                options.add_provider_for_devices(devices, {})

                session = ort.InferenceSession(str(model_path), sess_options=options)
                probabilities = session.run(None, {"pixel_values": images})[0]
                profile_path = Path(session.end_profiling())
                if not profile_uses_qsharp(profile_path):
                    raise RuntimeError(
                        "The profile did not show Conv on QSharpExecutionProvider."
                    )
                return probabilities

        return execute()
    finally:
        ort.unregister_execution_provider_library(registration_name)


def main() -> None:
    args = parse_args()
    examples = [
        ("cat.jpg", "cat"),
        ("dog.jpg", "not-cat"),
    ]

    # The checked-in model already contains both phases of inference:
    #   1. ResNet-18 maps each image to a 512-value embedding on CPU.
    #   2. A 1x1 Conv + Flatten + Add + Sigmoid maps the pooled
    #      [N,512,1,1] tensor to a cat probability. Conv runs on QSharp.
    image_batch = np.stack(
        [preprocess_image(args.images / filename) for filename, _ in examples]
    ).astype(np.float32)

    # Run that identical model twice. CPU gives a stable reference, while the
    # Q# sparse-state simulator estimates Conv through finite shots and
    # therefore produces a nearby stochastic probability.
    cpu_probabilities = run_cpu_model(args.resnet, image_batch)
    quantum_probabilities = run_quantum_model(
        args.plugin,
        args.resnet,
        image_batch,
    )

    print("image     expected  cpu_probability  quantum_probability  quantum_label")
    for index, (filename, expected) in enumerate(examples):
        cpu_probability = float(cpu_probabilities[index, 0])
        quantum_probability = float(quantum_probabilities[index, 0])
        quantum_label = "cat" if quantum_probability >= 0.5 else "not-cat"
        print(
            f"{filename:<9} {expected:<9} {cpu_probability:>15.6f} "
            f"{quantum_probability:>20.6f}  {quantum_label}"
        )
        if quantum_label != expected:
            raise RuntimeError(
                f"{filename} was classified as {quantum_label}, expected {expected}."
            )


if __name__ == "__main__":
    main()
