#include <memory>

#include "quantum_ep/quantum_ep_factory.h"

#if defined(__APPLE__)
#define QSHARP_EP_EXPORT __attribute__((visibility("default")))
#elif defined(_WIN32)
#define QSHARP_EP_EXPORT __declspec(dllexport)
#else
#define QSHARP_EP_EXPORT __attribute__((visibility("default")))
#endif

extern "C" {

QSHARP_EP_EXPORT OrtStatus* CreateEpFactories(
    const char* /*registration_name*/,
    const OrtApiBase* api_base,
    const OrtLogger* default_logger,
    OrtEpFactory** factories,
    size_t max_factories,
    size_t* num_factories) {
  if (api_base == nullptr || default_logger == nullptr || factories == nullptr || num_factories == nullptr) {
    return nullptr;
  }

  const OrtApi* api = api_base->GetApi(ORT_API_VERSION);
  if (api == nullptr) {
    return nullptr;
  }

  Ort::InitApi(api);
  *num_factories = 0;

  if (max_factories < 1) {
    return api->CreateStatus(ORT_INVALID_ARGUMENT, "QSharp EP requires space for one factory.");
  }

  try {
    auto factory = std::make_unique<QuantumEpFactory>(*api, *api->GetEpApi(), *default_logger);
    factories[0] = factory.release();
    *num_factories = 1;
    return nullptr;
  } catch (const std::exception& exception) {
    return api->CreateStatus(ORT_EP_FAIL, exception.what());
  }
}

QSHARP_EP_EXPORT OrtStatus* ReleaseEpFactory(OrtEpFactory* factory) {
  delete static_cast<QuantumEpFactory*>(factory);
  return nullptr;
}

}  // extern "C"
