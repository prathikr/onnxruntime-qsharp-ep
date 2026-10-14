#pragma once

#include <memory>

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#define QSHARP_EP_RETURN_IF_ERROR(expression) \
  do {                                        \
    OrtStatus* status = (expression);         \
    if (status != nullptr) {                  \
      return status;                          \
    }                                         \
  } while (false)

struct OrtKeyValuePairsDeleter {
  const OrtApi* api{};

  void operator()(OrtKeyValuePairs* value) const noexcept {
    if (value != nullptr) {
      api->ReleaseKeyValuePairs(value);
    }
  }
};

using UniqueKeyValuePairs = std::unique_ptr<OrtKeyValuePairs, OrtKeyValuePairsDeleter>;
