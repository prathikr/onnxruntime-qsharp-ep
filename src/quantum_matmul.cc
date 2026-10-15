#include "quantum_ep/quantum_matmul.h"

#include <stdexcept>
#include <vector>

#include "quantum_ep/quantum_simulator.h"

QuantumMatMulKernel::QuantumMatMulKernel(const OrtApi& api, const OrtLogger& logger)
    : api_{api},
      logger_{logger} {
}

OrtStatus* QuantumMatMulKernel::Compute(OrtKernelContext* context) {
  try {
    Ort::KernelContext kernel_context{context};
    const Ort::ConstValue left = kernel_context.GetInput(0);
    const Ort::ConstValue right = kernel_context.GetInput(1);
    const auto left_shape = left.GetTensorTypeAndShapeInfo().GetShape();
    const auto right_shape = right.GetTensorTypeAndShapeInfo().GetShape();

    if (left_shape.size() != 2 || right_shape.size() != 2 ||
        left_shape[0] <= 0 || left_shape[1] <= 0 ||
        right_shape[0] != left_shape[1] || right_shape[1] != 1) {
      throw std::invalid_argument{
          "Quantum MatMul supports float tensors [M,K] x [K,1] with positive dimensions."};
    }

    const size_t row_count = static_cast<size_t>(left_shape[0]);
    const size_t shared_dimension = static_cast<size_t>(left_shape[1]);
    const std::vector<float> results = RunQuantumDotProducts(
        left.GetTensorData<float>(),
        row_count,
        shared_dimension,
        right.GetTensorData<float>());

    const std::vector<int64_t> output_shape{
        static_cast<int64_t>(row_count), 1};
    float* output =
        kernel_context.GetOutput(0, output_shape).GetTensorMutableData<float>();
    std::copy(results.begin(), results.end(), output);

    OrtStatus* log_status = api_.Logger_LogMessage(
        &logger_,
        ORT_LOGGING_LEVEL_INFO,
        "Executed standard ONNX MatMul with the local Q# simulator.",
        __FILE__,
        __LINE__,
        __func__);
    api_.ReleaseStatus(log_status);
    return nullptr;
  } catch (const Ort::Exception& exception) {
    return Ort::Status{exception}.release();
  } catch (const std::exception& exception) {
    return Ort::Status{exception.what(), ORT_EP_FAIL}.release();
  }
}
