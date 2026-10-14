# ONNX Runtime QSharp Execution Provider

Experimental out-of-tree ONNX Runtime plugin execution provider for Q# workloads.

The current prototype provides:

- `QSharpExecutionProvider` registration
- `QSharpSimulator` device discovery
- standard ONNX `MatMul` implemented with the local Q# sparse-state simulator
- CPU fallback for standard `Add`, `Sigmoid`, `Sub`, `Concat`, `Greater`, and `Cast`
- CPU differential correctness testing
- clean plugin unload

The same unmodified standard-operator model runs with CPU only or with Q# `MatMul` plus CPU fallback.

## Pinned ONNX Runtime

- Release: `v1.31.0`
- Minimum EP C API version: `31`

CMake accepts ONNX Runtime `1.31.0` or newer and requires EP C API version `31` or newer. The known-good build used commit `402979ac1bef89dcd55add41eaa7277c407599eb`, but that commit is recorded for reproducibility rather than enforced.

## Build and test on macOS

Install the QDK Python bridge and model-test dependencies into the workspace virtual environment:

```bash
../.venv/bin/pip install -r requirements.txt
```

The defaults expect:

- the `v1.31.0` source/header worktree at `../onnxruntime-1.31.0`
- the matching shared runtime at `../onnxruntime/build/ChunkwiseWebGPU/Release/libonnxruntime.dylib`

```bash
git -C ../onnxruntime worktree add --detach ../onnxruntime-1.31.0 v1.31.0
```

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Override `ONNXRUNTIME_ROOT` and `ONNXRUNTIME_LIBRARY` when using another location:

```bash
cmake -S . -B build \
  -DONNXRUNTIME_ROOT=/path/to/onnxruntime \
  -DONNXRUNTIME_LIBRARY=/path/to/libonnxruntime.dylib
```

The plugin artifact is `build/libonnxruntime_qsharp_ep.dylib`.

The saved fixture is `testdata/logistic_regression.onnx`. Tests load it directly rather than exporting a model on every run. Regenerate it only when the graph intentionally changes:

```bash
../.venv/bin/python tools/generate_model.py
```

The GoogleTest suite runs the identical model with CPU only and with the Quantum EP, checks profiling data to verify `MatMul` ran on `QSharpExecutionProvider`, and compares labels and probabilities. It contains five cases: small and medium inputs plus zero-feature, cancellation, and large-magnitude edge cases.

The test disables ONNX Runtime graph optimization so that `MatMul + Add` is not rewritten into `Gemm` before provider partitioning. Supporting optimized sessions requires a future standard `Gemm` implementation.

Run the manual Python example with:

```bash
../.venv/bin/python examples/logistic_regression.py
```

The current prototype starts a Python/Q# simulator worker for each inference call. A persistent worker or native QIR bridge is required before performance testing.

The worker uses the direct `qdk.Context` API and explicitly selects the QDK sparse-state simulator with `Context.run(..., type="sparse")`.
