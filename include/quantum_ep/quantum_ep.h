#pragma once

#include <memory>
#include <string>

#include "quantum_ep/api_helpers.h"

class QuantumEpFactory;

class QuantumEp final : public OrtEp {
 public:
  QuantumEp(QuantumEpFactory& factory, const OrtLogger& logger);

 private:
  static const char* ORT_API_CALL GetNameImpl(const OrtEp* this_ptr) noexcept;
  static OrtStatus* ORT_API_CALL GetCapabilityImpl(
      OrtEp* this_ptr,
      const OrtGraph* graph,
      OrtEpGraphSupportInfo* graph_support_info) noexcept;
  static OrtStatus* ORT_API_CALL CompileImpl(
      OrtEp* this_ptr,
      const OrtGraph** graphs,
      const OrtNode** fused_nodes,
      size_t count,
      OrtNodeComputeInfo** node_compute_infos,
      OrtNode** ep_context_nodes) noexcept;
  static void ORT_API_CALL ReleaseNodeComputeInfosImpl(
      OrtEp* this_ptr,
      OrtNodeComputeInfo** node_compute_infos,
      size_t count) noexcept;

  QuantumEpFactory& factory_;
  const OrtApi& api_;
  std::string name_;
  const OrtLogger& logger_;
};
