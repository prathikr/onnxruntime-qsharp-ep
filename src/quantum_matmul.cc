#include "quantum_ep/quantum_matmul.h"

#include <array>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

namespace {

std::string ShellQuote(const std::string& value) {
  std::string quoted{"'"};
  for (const char character : value) {
    if (character == '\'') {
      quoted += "'\\''";
    } else {
      quoted += character;
    }
  }
  quoted += '\'';
  return quoted;
}

std::vector<float> RunSimulator(
    const float* left,
    size_t row_count,
    size_t shared_dimension,
    const float* right) {
  std::array<char, 64> input_template{};
  std::strncpy(input_template.data(), "/tmp/qsharp-ep-XXXXXX", input_template.size() - 1);
  const int file_descriptor = mkstemp(input_template.data());
  if (file_descriptor == -1) {
    throw std::runtime_error{"Unable to create Q# simulator input: " + std::string{std::strerror(errno)}};
  }
  close(file_descriptor);

  const std::string input_path{input_template.data()};
  try {
    std::ofstream input{input_path};
    if (!input) {
      throw std::runtime_error{"Unable to open Q# simulator input file."};
    }

    input << row_count << ' ' << shared_dimension << " 20000 42\n";
    for (size_t index = 0; index < shared_dimension; ++index) {
      input << right[index] << (index + 1 == shared_dimension ? '\n' : ' ');
    }
    for (size_t row = 0; row < row_count; ++row) {
      for (size_t column = 0; column < shared_dimension; ++column) {
        input << left[row * shared_dimension + column]
              << (column + 1 == shared_dimension ? '\n' : ' ');
      }
    }
    input.close();

    const std::string command =
        ShellQuote(QSHARP_EP_PYTHON_EXECUTABLE) + " " +
        ShellQuote(QSHARP_EP_WORKER_SCRIPT) + " --qsharp-source " +
        ShellQuote(QSHARP_EP_QSHARP_SOURCE) + " --input " +
        ShellQuote(input_path);

    std::unique_ptr<FILE, decltype(&pclose)> process{popen(command.c_str(), "r"), pclose};
    if (!process) {
      throw std::runtime_error{"Unable to start the Q# simulator worker."};
    }

    std::string output;
    std::array<char, 512> buffer{};
    while (fgets(buffer.data(), static_cast<int>(buffer.size()), process.get()) != nullptr) {
      output += buffer.data();
    }

    FILE* process_handle = process.release();
    const int process_status = pclose(process_handle);
    if (process_status == -1 || !WIFEXITED(process_status) || WEXITSTATUS(process_status) != 0) {
      throw std::runtime_error{"Q# simulator worker failed: " + output};
    }

    std::istringstream results_stream{output};
    std::vector<float> results(row_count);
    for (float& result : results) {
      if (!(results_stream >> result)) {
        throw std::runtime_error{"Q# simulator returned malformed output: " + output};
      }
    }

    std::remove(input_path.c_str());
    return results;
  } catch (...) {
    std::remove(input_path.c_str());
    throw;
  }
}

}  // namespace

QuantumMatMulKernel::QuantumMatMulKernel(const OrtApi& api, const OrtLogger& logger)
    : api_{api},
      logger_{logger} {
}

OrtStatus* QuantumMatMulKernel::Compute(OrtKernelContext* context) {
  try {
    Ort::KernelContext kernel_context{context};
    const Ort::ConstValue left = kernel_context.GetInput(0);
    const Ort::ConstValue right = kernel_context.GetInput(1);
    const auto left_shape = left.GetTensorTypeAndShapeInfo().GetShape();
    const auto right_shape = right.GetTensorTypeAndShapeInfo().GetShape();

    if (left_shape.size() != 2 || right_shape.size() != 2 ||
        left_shape[0] <= 0 || left_shape[1] <= 0 ||
        right_shape[0] != left_shape[1] || right_shape[1] != 1) {
      throw std::invalid_argument{
          "Quantum MatMul supports float tensors [M,K] x [K,1] with positive dimensions."};
    }

    const size_t row_count = static_cast<size_t>(left_shape[0]);
    const size_t shared_dimension = static_cast<size_t>(left_shape[1]);
    const std::vector<float> results = RunSimulator(
        left.GetTensorData<float>(),
        row_count,
        shared_dimension,
        right.GetTensorData<float>());

    const std::vector<int64_t> output_shape{
        static_cast<int64_t>(row_count), 1};
    float* output =
        kernel_context.GetOutput(0, output_shape).GetTensorMutableData<float>();
    std::copy(results.begin(), results.end(), output);

    OrtStatus* log_status = api_.Logger_LogMessage(
        &logger_,
        ORT_LOGGING_LEVEL_INFO,
        "Executed standard ONNX MatMul with the local Q# simulator.",
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
