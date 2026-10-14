#include "quantum_ep/quantum_ep_factory.h"

#include <memory>

#include "quantum_ep/quantum_ep.h"

QuantumEpFactory::QuantumEpFactory(
    const OrtApi& api,
    const OrtEpApi& ep_api,
    const OrtLogger& /*default_logger*/)
    : OrtEpFactory{},
      api_{api},
      ep_api_{ep_api} {
  ort_version_supported = ORT_API_VERSION;
  GetName = GetNameImpl;
  GetVendor = GetVendorImpl;
  GetVendorId = GetVendorIdImpl;
  GetVersion = GetVersionImpl;
  GetSupportedDevices = GetSupportedDevicesImpl;
  CreateEp = CreateEpImpl;
  ReleaseEp = ReleaseEpImpl;
  CreateDataTransfer = CreateDataTransferImpl;
}

const char* ORT_API_CALL QuantumEpFactory::GetNameImpl(const OrtEpFactory* this_ptr) noexcept {
  return static_cast<const QuantumEpFactory*>(this_ptr)->name_.c_str();
}

const char* ORT_API_CALL QuantumEpFactory::GetVendorImpl(const OrtEpFactory* this_ptr) noexcept {
  return static_cast<const QuantumEpFactory*>(this_ptr)->vendor_.c_str();
}

uint32_t ORT_API_CALL QuantumEpFactory::GetVendorIdImpl(const OrtEpFactory* /*this_ptr*/) noexcept {
  return 0;
}

const char* ORT_API_CALL QuantumEpFactory::GetVersionImpl(const OrtEpFactory* this_ptr) noexcept {
  return static_cast<const QuantumEpFactory*>(this_ptr)->version_.c_str();
}

OrtStatus* ORT_API_CALL QuantumEpFactory::GetSupportedDevicesImpl(
    OrtEpFactory* this_ptr,
    const OrtHardwareDevice* const* devices,
    size_t num_devices,
    OrtEpDevice** ep_devices,
    size_t max_ep_devices,
    size_t* num_ep_devices) noexcept {
  auto* factory = static_cast<QuantumEpFactory*>(this_ptr);
  *num_ep_devices = 0;

  for (size_t i = 0; i < num_devices && *num_ep_devices < max_ep_devices; ++i) {
    if (factory->api_.HardwareDevice_Type(devices[i]) != OrtHardwareDeviceType_CPU) {
      continue;
    }

    OrtKeyValuePairs* metadata_value = nullptr;
    factory->api_.CreateKeyValuePairs(&metadata_value);
    UniqueKeyValuePairs metadata{metadata_value, OrtKeyValuePairsDeleter{&factory->api_}};

    OrtKeyValuePairs* options_value = nullptr;
    factory->api_.CreateKeyValuePairs(&options_value);
    UniqueKeyValuePairs options{options_value, OrtKeyValuePairsDeleter{&factory->api_}};

    factory->api_.AddKeyValuePair(metadata.get(), "backend", "QSharpSimulator");
    factory->api_.AddKeyValuePair(metadata.get(), "execution_mode", "local");
    factory->api_.AddKeyValuePair(options.get(), "backend", "qdk-simulator");

    OrtEpDevice* ep_device = nullptr;
    QSHARP_EP_RETURN_IF_ERROR(factory->ep_api_.CreateEpDevice(
        factory, devices[i], metadata.get(), options.get(), &ep_device));
    ep_devices[(*num_ep_devices)++] = ep_device;
  }

  return nullptr;
}

OrtStatus* ORT_API_CALL QuantumEpFactory::CreateEpImpl(
    OrtEpFactory* this_ptr,
    const OrtHardwareDevice* const* /*devices*/,
    const OrtKeyValuePairs* const* /*ep_metadata*/,
    size_t num_devices,
    const OrtSessionOptions* /*session_options*/,
    const OrtLogger* logger,
    OrtEp** ep) noexcept {
  auto* factory = static_cast<QuantumEpFactory*>(this_ptr);
  *ep = nullptr;

  if (num_devices != 1) {
    return factory->api_.CreateStatus(
        ORT_INVALID_ARGUMENT,
        "QSharpExecutionProvider requires exactly one simulator device.");
  }

  try {
    auto quantum_ep = std::make_unique<QuantumEp>(*factory, *logger);
    *ep = quantum_ep.release();
    return nullptr;
  } catch (const std::exception& exception) {
    return factory->api_.CreateStatus(ORT_EP_FAIL, exception.what());
  }
}

void ORT_API_CALL QuantumEpFactory::ReleaseEpImpl(
    OrtEpFactory* /*this_ptr*/,
    OrtEp* ep) noexcept {
  delete static_cast<QuantumEp*>(ep);
}

OrtStatus* ORT_API_CALL QuantumEpFactory::CreateDataTransferImpl(
    OrtEpFactory* /*this_ptr*/,
    OrtDataTransferImpl** data_transfer) noexcept {
  *data_transfer = nullptr;
  return nullptr;
}
