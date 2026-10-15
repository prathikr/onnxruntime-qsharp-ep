#include "quantum_ep/quantum_ep.h"

#include <memory>
#include <vector>

#include "quantum_ep/quantum_conv.h"
#include "quantum_ep/quantum_ep_factory.h"
#include "quantum_ep/quantum_matmul.h"

namespace {

struct QuantumComputeInfo : OrtNodeComputeInfo {
  virtual ~QuantumComputeInfo() = default;
};

struct QuantumMatMulComputeInfo final : QuantumComputeInfo {
  QuantumMatMulComputeInfo(const OrtApi& api, const OrtLogger& logger)
      : kernel{api, logger} {
    ort_version_supported = ORT_API_VERSION;
    CreateState = CreateStateImpl;
    Compute = ComputeImpl;
    ReleaseState = ReleaseStateImpl;
  }

  static OrtStatus* ORT_API_CALL CreateStateImpl(
      OrtNodeComputeInfo* this_ptr,
      OrtNodeComputeContext* /*context*/,
      void** state) {
    *state = &static_cast<QuantumMatMulComputeInfo*>(this_ptr)->kernel;
    return nullptr;
  }

  static OrtStatus* ORT_API_CALL ComputeImpl(
      OrtNodeComputeInfo* /*this_ptr*/,
      void* state,
      OrtKernelContext* context) {
    return static_cast<QuantumMatMulKernel*>(state)->Compute(context);
  }

  static void ORT_API_CALL ReleaseStateImpl(
      OrtNodeComputeInfo* /*this_ptr*/,
      void* /*state*/) {
  }

  QuantumMatMulKernel kernel;
};

struct QuantumConvComputeInfo final : QuantumComputeInfo {
  QuantumConvComputeInfo(const OrtApi& api, const OrtLogger& logger)
      : kernel{api, logger} {
    ort_version_supported = ORT_API_VERSION;
    CreateState = CreateStateImpl;
    Compute = ComputeImpl;
    ReleaseState = ReleaseStateImpl;
  }

  static OrtStatus* ORT_API_CALL CreateStateImpl(
      OrtNodeComputeInfo* this_ptr,
      OrtNodeComputeContext* /*context*/,
      void** state) {
    *state = &static_cast<QuantumConvComputeInfo*>(this_ptr)->kernel;
    return nullptr;
  }

  static OrtStatus* ORT_API_CALL ComputeImpl(
      OrtNodeComputeInfo* /*this_ptr*/,
      void* state,
      OrtKernelContext* context) {
    return static_cast<QuantumConvKernel*>(state)->Compute(context);
  }

  static void ORT_API_CALL ReleaseStateImpl(
      OrtNodeComputeInfo* /*this_ptr*/,
      void* /*state*/) {
  }

  QuantumConvKernel kernel;
};

bool IsSupportedMatMul(const Ort::ConstNode& node) {
  const std::string domain = node.GetDomain();
  if ((!domain.empty() && domain != "ai.onnx") ||
      node.GetOperatorType() != "MatMul") {
    return false;
  }

  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.size() != 2 || outputs.size() != 1) {
    return false;
  }

  const auto left_info = inputs[0].TypeInfo();
  const auto right_info = inputs[1].TypeInfo();
  const auto output_info = outputs[0].TypeInfo();
  if (left_info.GetONNXType() != ONNX_TYPE_TENSOR ||
      right_info.GetONNXType() != ONNX_TYPE_TENSOR ||
      output_info.GetONNXType() != ONNX_TYPE_TENSOR) {
    return false;
  }

  const auto left = left_info.GetTensorTypeAndShapeInfo();
  const auto right = right_info.GetTensorTypeAndShapeInfo();
  const auto output = output_info.GetTensorTypeAndShapeInfo();
  if (left.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      right.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      output.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return false;
  }

  const auto left_shape = left.GetShape();
  const auto right_shape = right.GetShape();
  return left_shape.size() == 2 && right_shape.size() == 2 &&
         left_shape[1] > 0 && right_shape[0] == left_shape[1] &&
         right_shape[1] == 1;
}

bool HasIntAttribute(
    const Ort::ConstNode& node,
    const std::string& name,
    int64_t expected) {
  Ort::ConstOpAttr attribute{nullptr};
  if (!node.GetAttributeByName(name, attribute).IsOK()) {
    return false;
  }

  int64_t value{};
  return attribute.GetValue(value).IsOK() && value == expected;
}

bool HasIntsAttribute(
    const Ort::ConstNode& node,
    const std::string& name,
    const std::vector<int64_t>& expected) {
  Ort::ConstOpAttr attribute{nullptr};
  if (!node.GetAttributeByName(name, attribute).IsOK()) {
    return false;
  }

  std::vector<int64_t> values;
  return attribute.GetValueArray(values).IsOK() && values == expected;
}

bool IsSupportedConv(const Ort::ConstNode& node) {
  const std::string domain = node.GetDomain();
  if ((!domain.empty() && domain != "ai.onnx") ||
      node.GetOperatorType() != "Conv") {
    return false;
  }

  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.size() != 2 || outputs.size() != 1) {
    return false;
  }

  const auto input_info = inputs[0].TypeInfo();
  const auto weight_info = inputs[1].TypeInfo();
  const auto output_info = outputs[0].TypeInfo();
  if (input_info.GetONNXType() != ONNX_TYPE_TENSOR ||
      weight_info.GetONNXType() != ONNX_TYPE_TENSOR ||
      output_info.GetONNXType() != ONNX_TYPE_TENSOR) {
    return false;
  }

  const auto input = input_info.GetTensorTypeAndShapeInfo();
  const auto weights = weight_info.GetTensorTypeAndShapeInfo();
  const auto output = output_info.GetTensorTypeAndShapeInfo();
  if (input.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      weights.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      output.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return false;
  }

  const auto input_shape = input.GetShape();
  const auto weight_shape = weights.GetShape();
  const auto output_shape = output.GetShape();
  return HasIntsAttribute(node, "kernel_shape", {1, 1}) &&
         HasIntsAttribute(node, "strides", {1, 1}) &&
         HasIntsAttribute(node, "pads", {0, 0, 0, 0}) &&
         HasIntsAttribute(node, "dilations", {1, 1}) &&
         HasIntAttribute(node, "group", 1) &&
         input_shape.size() == 4 && weight_shape.size() == 4 &&
         output_shape.size() == 4 &&
         input_shape[1] > 0 && input_shape[2] == 1 && input_shape[3] == 1 &&
         weight_shape[0] == 1 && weight_shape[1] == input_shape[1] &&
         weight_shape[2] == 1 && weight_shape[3] == 1 &&
         output_shape[1] == 1 && output_shape[2] == 1 && output_shape[3] == 1;
}

}  // namespace

QuantumEp::QuantumEp(QuantumEpFactory& factory, const OrtLogger& logger)
    : OrtEp{},
      factory_{factory},
      api_{factory.Api()},
      name_{factory.Name()},
      logger_{logger} {
  ort_version_supported = ORT_API_VERSION;
  GetName = GetNameImpl;
  GetCapability = GetCapabilityImpl;
  Compile = CompileImpl;
  ReleaseNodeComputeInfos = ReleaseNodeComputeInfosImpl;

  OrtStatus* status = api_.Logger_LogMessage(
      &logger_,
      ORT_LOGGING_LEVEL_INFO,
      "Created QSharpExecutionProvider with standard MatMul and constrained "
      "1x1 Conv support.",
      __FILE__,
      __LINE__,
      __func__);
  api_.ReleaseStatus(status);
}

const char* ORT_API_CALL QuantumEp::GetNameImpl(const OrtEp* this_ptr) noexcept {
  return static_cast<const QuantumEp*>(this_ptr)->name_.c_str();
}

OrtStatus* ORT_API_CALL QuantumEp::GetCapabilityImpl(
    OrtEp* this_ptr,
    const OrtGraph* graph,
    OrtEpGraphSupportInfo* graph_support_info) noexcept {
  try {
    auto* ep = static_cast<QuantumEp*>(this_ptr);
    for (const auto& node : Ort::ConstGraph{graph}.GetNodes()) {
      if (IsSupportedMatMul(node) || IsSupportedConv(node)) {
        const OrtNode* nodes[]{node};
        OrtNodeFusionOptions options{};
        options.ort_version_supported = ORT_API_VERSION;
        options.drop_constant_initializers = false;
        QSHARP_EP_RETURN_IF_ERROR(ep->factory_.EpApi().EpGraphSupportInfo_AddNodesToFuse(
            graph_support_info, nodes, 1, &options));
      }
    }
    return nullptr;
  } catch (const Ort::Exception& exception) {
    return Ort::Status{exception}.release();
  } catch (const std::exception& exception) {
    return Ort::Status{exception.what(), ORT_EP_FAIL}.release();
  }
}

OrtStatus* ORT_API_CALL QuantumEp::CompileImpl(
    OrtEp* this_ptr,
    const OrtGraph** graphs,
    const OrtNode** fused_nodes,
    size_t count,
    OrtNodeComputeInfo** node_compute_infos,
    OrtNode** /*ep_context_nodes*/) noexcept {
  try {
    auto* ep = static_cast<QuantumEp*>(this_ptr);
    for (size_t index = 0; index < count; ++index) {
      const auto nodes = Ort::ConstGraph{graphs[index]}.GetNodes();
      if (nodes.size() != 1 ||
          (!IsSupportedMatMul(nodes[0]) && !IsSupportedConv(nodes[0]))) {
        return ep->api_.CreateStatus(
            ORT_EP_FAIL,
            "QSharpExecutionProvider expected one supported MatMul or Conv node.");
      }
      if (Ort::ConstNode{fused_nodes[index]}.GetEpName() != ep->name_) {
        return ep->api_.CreateStatus(
            ORT_EP_FAIL,
            "Compiled MatMul node was not assigned to QSharpExecutionProvider.");
      }

      if (IsSupportedMatMul(nodes[0])) {
        node_compute_infos[index] =
            new QuantumMatMulComputeInfo{ep->api_, ep->logger_};
      } else {
        node_compute_infos[index] =
            new QuantumConvComputeInfo{ep->api_, ep->logger_};
      }
    }
    return nullptr;
  } catch (const Ort::Exception& exception) {
    return Ort::Status{exception}.release();
  } catch (const std::exception& exception) {
    return Ort::Status{exception.what(), ORT_EP_FAIL}.release();
  }
}

void ORT_API_CALL QuantumEp::ReleaseNodeComputeInfosImpl(
    OrtEp* /*this_ptr*/,
    OrtNodeComputeInfo** node_compute_infos,
    size_t count) noexcept {
  for (size_t index = 0; index < count; ++index) {
    delete static_cast<QuantumComputeInfo*>(node_compute_infos[index]);
  }
}
