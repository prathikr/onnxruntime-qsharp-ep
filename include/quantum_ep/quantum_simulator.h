#pragma once

#include <cstddef>
#include <vector>

std::vector<float> RunQuantumDotProducts(
    const float* rows,
    size_t row_count,
    size_t feature_count,
    const float* weights);
