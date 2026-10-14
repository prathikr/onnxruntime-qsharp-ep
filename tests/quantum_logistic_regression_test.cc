#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"

namespace {

constexpr const char* kRegistrationName = "qsharp_ep_tests";
constexpr const char* kProviderName = "QSharpExecutionProvider";
constexpr float kProbabilityTolerance = 0.02f;

struct InferenceResult {
  std::vector<int64_t> labels;
  std::vector<float> probabilities;
};

class QuantumLogisticRegressionTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    env_ = std::make_unique<Ort::Env>(
        ORT_LOGGING_LEVEL_WARNING,
        "qsharp-ep-cpp-tests");
    env_->RegisterExecutionProviderLibrary(
        kRegistrationName,
        ORT_TSTR(QSHARP_EP_PLUGIN_PATH));

    for (const auto& device : env_->GetEpDevices()) {
      if (std::string{device.EpName()} == kProviderName) {
        quantum_device_ = device;
        break;
      }
    }

    ASSERT_TRUE(quantum_device_);
    const auto metadata = quantum_device_.EpMetadata().GetKeyValuePairs();
    const auto backend = metadata.find("backend");
    ASSERT_NE(backend, metadata.end());
    EXPECT_EQ(backend->second, "QSharpSimulator");
  }

  static void TearDownTestSuite() {
    quantum_device_ = Ort::ConstEpDevice{nullptr};
    env_->UnregisterExecutionProviderLibrary(kRegistrationName);
    env_.reset();
  }

  static InferenceResult RunCpu(const std::vector<float>& features) {
    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    Ort::Session session{
        *env_,
        ORT_TSTR(QSHARP_EP_MODEL_PATH),
        options};
    return Run(session, features);
  }

  static InferenceResult RunQuantum(
      const std::vector<float>& features,
      const std::string& profile_prefix) {
    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    options.EnableProfiling(profile_prefix.c_str());
    options.AppendExecutionProvider_V2(
        *env_,
        {quantum_device_},
        Ort::KeyValuePairs{});

    Ort::Session session{
        *env_,
        ORT_TSTR(QSHARP_EP_MODEL_PATH),
        options};
    InferenceResult result = Run(session, features);

    auto allocator = Ort::AllocatorWithDefaultOptions{};
    const auto profile_path = session.EndProfilingAllocated(allocator);
    std::ifstream profile{profile_path.get()};
    if (!profile) {
      ADD_FAILURE() << "Unable to open ONNX Runtime profile " << profile_path.get();
      return result;
    }
    const std::string profile_contents{
        std::istreambuf_iterator<char>{profile},
        std::istreambuf_iterator<char>{}};
    EXPECT_NE(
        profile_contents.find("\"provider\" : \"QSharpExecutionProvider\""),
        std::string::npos);
    profile.close();
    EXPECT_EQ(std::remove(profile_path.get()), 0);

    return result;
  }

  static void ExpectMatchesCpu(
      const std::vector<float>& features,
      const std::string& profile_prefix,
      bool compare_labels = true) {
    ASSERT_EQ(features.size() % 4, 0U);
    const InferenceResult cpu = RunCpu(features);
    const InferenceResult quantum = RunQuantum(features, profile_prefix);

    ASSERT_EQ(quantum.labels.size(), cpu.labels.size());
    ASSERT_EQ(quantum.probabilities.size(), cpu.probabilities.size());

    if (compare_labels) {
      EXPECT_EQ(quantum.labels, cpu.labels);
    }

    for (size_t index = 0; index < cpu.probabilities.size(); ++index) {
      EXPECT_NEAR(
          quantum.probabilities[index],
          cpu.probabilities[index],
          kProbabilityTolerance)
          << "probability index " << index;
    }
  }

 private:
  static InferenceResult Run(
      Ort::Session& session,
      const std::vector<float>& features) {
    const int64_t row_count = static_cast<int64_t>(features.size() / 4);
    const std::array<int64_t, 2> shape{row_count, 4};
    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input = Ort::Value::CreateTensor<float>(
        memory_info,
        const_cast<float*>(features.data()),
        features.size(),
        shape.data(),
        shape.size());

    const char* input_names[]{"X"};
    const char* output_names[]{"label", "probabilities"};
    auto outputs = session.Run(
        Ort::RunOptions{nullptr},
        input_names,
        &input,
        1,
        output_names,
        2);

    const size_t label_count =
        outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
    const size_t probability_count =
        outputs[1].GetTensorTypeAndShapeInfo().GetElementCount();
    const int64_t* labels = outputs[0].GetTensorData<int64_t>();
    const float* probabilities = outputs[1].GetTensorData<float>();

    return {
        std::vector<int64_t>{labels, labels + label_count},
        std::vector<float>{
            probabilities,
            probabilities + probability_count}};
  }

  static std::unique_ptr<Ort::Env> env_;
  static Ort::ConstEpDevice quantum_device_;
};

std::unique_ptr<Ort::Env> QuantumLogisticRegressionTest::env_;
Ort::ConstEpDevice QuantumLogisticRegressionTest::quantum_device_{nullptr};

TEST_F(QuantumLogisticRegressionTest, SmallSingleRowInput) {
  ExpectMatchesCpu(
      {1.0f, 0.5f, -0.5f, 0.25f},
      "qsharp_small_input");
}

TEST_F(QuantumLogisticRegressionTest, MediumEightRowBatch) {
  ExpectMatchesCpu(
      {
          1.0f, 0.5f, -0.5f, 0.25f,
          -1.0f, 1.0f, 0.5f, -0.75f,
          0.25f, -0.5f, 1.0f, 0.5f,
          2.0f, -1.0f, 0.0f, 0.5f,
          -0.5f, -0.25f, -1.0f, 0.0f,
          0.75f, 0.25f, 0.5f, -0.5f,
          -1.5f, -0.5f, 0.25f, 0.75f,
          0.5f, -1.0f, -0.25f, 1.0f,
      },
      "qsharp_medium_input");
}

TEST_F(QuantumLogisticRegressionTest, EdgeAllZeroFeatures) {
  ExpectMatchesCpu(
      {0.0f, 0.0f, 0.0f, 0.0f},
      "qsharp_zero_features");
}

TEST_F(QuantumLogisticRegressionTest, EdgeCancellingWeightedTerms) {
  ExpectMatchesCpu(
      {1.0f, 1.0f, -1.2f, 0.0f},
      "qsharp_cancelling_terms");
}

TEST_F(QuantumLogisticRegressionTest, EdgeLargeMagnitudeSaturation) {
  ExpectMatchesCpu(
      {100.0f, -100.0f, 100.0f, 100.0f},
      "qsharp_large_magnitude");
}

}  // namespace

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
