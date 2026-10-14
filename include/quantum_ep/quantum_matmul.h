#pragma once

#include <cstdint>

#include "quantum_ep/api_helpers.h"

class QuantumMatMulKernel {
 public:
  QuantumMatMulKernel(const OrtApi& api, const OrtLogger& logger);

  OrtStatus* Compute(OrtKernelContext* context);

 private:
  const OrtApi& api_;
  const OrtLogger& logger_;
};
