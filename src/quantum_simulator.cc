#include "quantum_ep/quantum_simulator.h"

#include <array>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

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

}  // namespace

std::vector<float> RunQuantumDotProducts(
    const float* rows,
    size_t row_count,
    size_t feature_count,
    const float* weights) {
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

    input << row_count << ' ' << feature_count << " 20000 42\n";
    for (size_t index = 0; index < feature_count; ++index) {
      input << weights[index] << (index + 1 == feature_count ? '\n' : ' ');
    }
    for (size_t row = 0; row < row_count; ++row) {
      for (size_t column = 0; column < feature_count; ++column) {
        input << rows[row * feature_count + column]
              << (column + 1 == feature_count ? '\n' : ' ');
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
