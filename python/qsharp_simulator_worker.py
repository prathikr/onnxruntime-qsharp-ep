#!/usr/bin/env python3

import argparse
from pathlib import Path

from qdk import Context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qsharp-source", required=True)
    parser.add_argument("--input", required=True)
    return parser.parse_args()


def read_input(path: Path) -> tuple[int, int, int, int, list[float], list[list[float]]]:
    tokens = path.read_text(encoding="utf-8").split()
    values = iter(tokens)
    batch_size = int(next(values))
    feature_count = int(next(values))
    shots = int(next(values))
    seed = int(next(values))
    weights = [float(next(values)) for _ in range(feature_count)]
    rows = [
        [float(next(values)) for _ in range(feature_count)]
        for _ in range(batch_size)
    ]
    return batch_size, feature_count, shots, seed, weights, rows


def main() -> None:
    args = parse_args()
    _, feature_count, shots, seed, weights, rows = read_input(Path(args.input))

    context = Context()
    context.eval(Path(args.qsharp_source).read_text(encoding="utf-8"))

    for row_index, row in enumerate(rows):
        terms = [row[index] * weights[index] for index in range(feature_count)]
        scale = max(1.0, *(abs(term) for term in terms))
        normalized_terms = [term / scale for term in terms]
        qsharp_terms = ", ".join(repr(float(term)) for term in normalized_terms)
        samples = context.run(
            f"SampleTerms([{qsharp_terms}])",
            shots=shots,
            seed=seed + row_index,
            type="sparse",
        )

        estimates = []
        for column in range(feature_count):
            ones = sum(str(sample[column]) == "One" for sample in samples)
            estimates.append(1.0 - 2.0 * ones / shots)

        print(scale * sum(estimates))


if __name__ == "__main__":
    main()
