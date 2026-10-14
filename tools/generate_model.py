#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def main() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    output_path = repository_root / "testdata" / "logistic_regression.onnx"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    weights = np.array([[0.7], [-0.4], [0.25], [0.5]], dtype=np.float32)
    bias = np.array([-0.1], dtype=np.float32)
    one = np.array([1.0], dtype=np.float32)
    zero = np.array([0.0], dtype=np.float32)

    nodes = [
        helper.make_node("MatMul", ["X", "weights"], ["linear"], name="quantum_matmul"),
        helper.make_node("Add", ["linear", "bias"], ["logit"], name="add_bias"),
        helper.make_node("Sigmoid", ["logit"], ["probability_one"], name="sigmoid"),
        helper.make_node("Sub", ["one", "probability_one"], ["probability_zero"], name="probability_zero"),
        helper.make_node(
            "Concat",
            ["probability_zero", "probability_one"],
            ["probabilities"],
            name="probabilities",
            axis=1,
        ),
        helper.make_node("Greater", ["logit", "zero"], ["positive"], name="threshold"),
        helper.make_node(
            "Cast",
            ["positive"],
            ["label"],
            name="label",
            to=TensorProto.INT64,
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "standard-logistic-regression",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, 4])],
        [
            helper.make_tensor_value_info("label", TensorProto.INT64, [None, 1]),
            helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, [None, 2]),
        ],
        [
            numpy_helper.from_array(weights, name="weights"),
            numpy_helper.from_array(bias, name="bias"),
            numpy_helper.from_array(one, name="one"),
            numpy_helper.from_array(zero, name="zero"),
        ],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 21)],
        ir_version=10,
    )
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(output_path)


if __name__ == "__main__":
    main()
