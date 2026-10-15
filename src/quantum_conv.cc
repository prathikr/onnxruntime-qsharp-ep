#include "quantum_ep/quantum_conv.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "quantum_ep/quantum_simulator.h"

QuantumConvKernel::QuantumConvKernel(const OrtApi& api, const OrtLogger& logger)
    : api_{api},
      logger_{logger} {
}

OrtStatus* QuantumConvKernel::Compute(OrtKernelContext* context) {
  try {
    Ort::KernelContext kernel_context{context};
    const Ort::ConstValue input = kernel_context.GetInput(0);
    const Ort::ConstValue weights = kernel_context.GetInput(1);
    const auto input_shape = input.GetTensorTypeAndShapeInfo().GetShape();
    const auto weight_shape = weights.GetTensorTypeAndShapeInfo().GetShape();

    if (input_shape.size() != 4 || weight_shape.size() != 4 ||
        input_shape[0] <= 0 || input_shape[1] <= 0 ||
        input_shape[2] != 1 || input_shape[3] != 1 ||
        weight_shape[0] != 1 || weight_shape[1] != input_shape[1] ||
        weight_shape[2] != 1 || weight_shape[3] != 1) {
      throw std::invalid_argument{
          "Quantum Conv supports float NCHW [N,C,1,1] input and "
          "[1,C,1,1] weights."};
    }

    const size_t batch_size = static_cast<size_t>(input_shape[0]);
    const size_t channel_count = static_cast<size_t>(input_shape[1]);
    const std::vector<float> results = RunQuantumDotProducts(
        input.GetTensorData<float>(),
        batch_size,
        channel_count,
        weights.GetTensorData<float>());

    const std::vector<int64_t> output_shape{
        static_cast<int64_t>(batch_size), 1, 1, 1};
    float* output =
        kernel_context.GetOutput(0, output_shape).GetTensorMutableData<float>();
    std::copy(results.begin(), results.end(), output);

    OrtStatus* log_status = api_.Logger_LogMessage(
        &logger_,
        ORT_LOGGING_LEVEL_INFO,
        "Executed standard ONNX 1x1 Conv with the local Q# simulator.",
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
