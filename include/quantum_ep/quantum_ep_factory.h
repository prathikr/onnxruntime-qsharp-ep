#pragma once

#include <string>

#include "quantum_ep/api_helpers.h"

class QuantumEpFactory final : public OrtEpFactory {
 public:
  QuantumEpFactory(const OrtApi& api, const OrtEpApi& ep_api, const OrtLogger& default_logger);

  const OrtApi& Api() const noexcept {
    return api_;
  }

  const std::string& Name() const noexcept {
    return name_;
  }

  const OrtEpApi& EpApi() const noexcept {
    return ep_api_;
  }

 private:
  static const char* ORT_API_CALL GetNameImpl(const OrtEpFactory* this_ptr) noexcept;
  static const char* ORT_API_CALL GetVendorImpl(const OrtEpFactory* this_ptr) noexcept;
  static uint32_t ORT_API_CALL GetVendorIdImpl(const OrtEpFactory* this_ptr) noexcept;
  static const char* ORT_API_CALL GetVersionImpl(const OrtEpFactory* this_ptr) noexcept;
  static OrtStatus* ORT_API_CALL GetSupportedDevicesImpl(
      OrtEpFactory* this_ptr,
      const OrtHardwareDevice* const* devices,
      size_t num_devices,
      OrtEpDevice** ep_devices,
      size_t max_ep_devices,
      size_t* num_ep_devices) noexcept;
  static OrtStatus* ORT_API_CALL CreateEpImpl(
      OrtEpFactory* this_ptr,
      const OrtHardwareDevice* const* devices,
      const OrtKeyValuePairs* const* ep_metadata,
      size_t num_devices,
      const OrtSessionOptions* session_options,
      const OrtLogger* logger,
      OrtEp** ep) noexcept;
  static void ORT_API_CALL ReleaseEpImpl(OrtEpFactory* this_ptr, OrtEp* ep) noexcept;
  static OrtStatus* ORT_API_CALL CreateDataTransferImpl(
      OrtEpFactory* this_ptr,
      OrtDataTransferImpl** data_transfer) noexcept;
  const OrtApi& api_;
  const OrtEpApi& ep_api_;
  const std::string name_{"QSharpExecutionProvider"};
  const std::string vendor_{"QSharp"};
  const std::string version_{"0.1.0"};
};
