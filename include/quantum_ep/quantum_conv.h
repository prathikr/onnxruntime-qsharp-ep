#pragma once

#include "quantum_ep/api_helpers.h"

class QuantumConvKernel {
 public:
  QuantumConvKernel(const OrtApi& api, const OrtLogger& logger);

  OrtStatus* Compute(OrtKernelContext* context);

 private:
  const OrtApi& api_;
  const OrtLogger& logger_;
};
