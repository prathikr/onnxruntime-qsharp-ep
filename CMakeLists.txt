cmake_minimum_required(VERSION 3.24)

project(onnxruntime_qsharp_ep VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CTest)
include(cmake/OnnxRuntimePin.cmake)

set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../onnxruntime-1.31.0" CACHE PATH
    "Path to the pinned ONNX Runtime source tree")
set(ONNXRUNTIME_LIBRARY
    "${CMAKE_CURRENT_SOURCE_DIR}/../onnxruntime/build/ChunkwiseWebGPU/Release/libonnxruntime.dylib"
    CACHE FILEPATH "Path to the ONNX Runtime shared library")
get_filename_component(ONNXRUNTIME_LIBRARY_DIR "${ONNXRUNTIME_LIBRARY}" DIRECTORY)

set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include/onnxruntime/core/session")
set(ONNXRUNTIME_C_API_HEADER "${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_c_api.h")

if(NOT EXISTS "${ONNXRUNTIME_C_API_HEADER}")
  message(FATAL_ERROR "ONNX Runtime headers were not found under ${ONNXRUNTIME_INCLUDE_DIR}")
endif()

if(NOT EXISTS "${ONNXRUNTIME_LIBRARY}")
  message(FATAL_ERROR "ONNX Runtime shared library was not found at ${ONNXRUNTIME_LIBRARY}")
endif()

file(STRINGS "${ONNXRUNTIME_C_API_HEADER}" ORT_API_VERSION_LINE
     REGEX "^#define ORT_API_VERSION [0-9]+$")
if(NOT ORT_API_VERSION_LINE MATCHES "^#define ORT_API_VERSION ([0-9]+)$")
  message(FATAL_ERROR
          "Unable to determine the ONNX Runtime API version from "
          "${ONNXRUNTIME_C_API_HEADER}")
endif()
set(ONNXRUNTIME_API_VERSION "${CMAKE_MATCH_1}")
if(ONNXRUNTIME_API_VERSION LESS QSHARP_EP_ORT_MIN_API_VERSION)
  message(FATAL_ERROR
          "Unsupported ONNX Runtime API version ${ONNXRUNTIME_API_VERSION}. "
          "Expected >= ${QSHARP_EP_ORT_MIN_API_VERSION}.")
endif()

set(ONNXRUNTIME_VERSION "" CACHE STRING
    "ONNX Runtime version when VERSION_NUMBER is unavailable")
if(ONNXRUNTIME_VERSION STREQUAL "" AND EXISTS "${ONNXRUNTIME_ROOT}/VERSION_NUMBER")
  file(STRINGS "${ONNXRUNTIME_ROOT}/VERSION_NUMBER" ONNXRUNTIME_VERSION LIMIT_COUNT 1)
endif()
if(ONNXRUNTIME_VERSION STREQUAL "")
  message(FATAL_ERROR
          "Unable to determine the ONNX Runtime version. "
          "Set ONNXRUNTIME_VERSION to a compatible 1.31.x release.")
endif()
if(ONNXRUNTIME_VERSION VERSION_LESS QSHARP_EP_ORT_MIN_VERSION)
  message(FATAL_ERROR
          "Unsupported ONNX Runtime version ${ONNXRUNTIME_VERSION}. "
          "Expected >= ${QSHARP_EP_ORT_MIN_VERSION}.")
endif()

message(STATUS
        "Using ONNX Runtime ${ONNXRUNTIME_VERSION} with EP API "
        "${ONNXRUNTIME_API_VERSION}")

add_library(onnxruntime_qsharp_ep SHARED
  src/ep_entry.cc
  src/quantum_matmul.cc
  src/quantum_ep.cc
  src/quantum_ep_factory.cc)

target_include_directories(onnxruntime_qsharp_ep
  PRIVATE
    "${ONNXRUNTIME_INCLUDE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/include")

target_compile_definitions(onnxruntime_qsharp_ep PRIVATE
  QSHARP_EP_PYTHON_EXECUTABLE="${CMAKE_CURRENT_SOURCE_DIR}/../.venv/bin/python"
  QSHARP_EP_WORKER_SCRIPT="${CMAKE_CURRENT_SOURCE_DIR}/python/qsharp_simulator_worker.py"
  QSHARP_EP_QSHARP_SOURCE="${CMAKE_CURRENT_SOURCE_DIR}/qsharp/QuantumMatMul.qs")

target_compile_options(onnxruntime_qsharp_ep PRIVATE
  -Wall
  -Wextra
  -Wpedantic
  -Werror)

set_target_properties(onnxruntime_qsharp_ep PROPERTIES
  OUTPUT_NAME "onnxruntime_qsharp_ep"
  BUILD_RPATH "${ONNXRUNTIME_LIBRARY_DIR}")

if(BUILD_TESTING)
  set(GTEST_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../onnxruntime/build/ChunkwiseWebGPU/Release/_deps/googletest-src/googletest")
  if(NOT EXISTS "${GTEST_ROOT}/include/gtest/gtest.h" OR NOT EXISTS "${GTEST_ROOT}/src/gtest-all.cc")
    message(FATAL_ERROR "GoogleTest artifacts from the local ONNX Runtime build were not found")
  endif()

  add_executable(qsharp_ep_tests
    tests/quantum_logistic_regression_test.cc
    "${GTEST_ROOT}/src/gtest-all.cc")
  target_include_directories(qsharp_ep_tests PRIVATE
    "${ONNXRUNTIME_INCLUDE_DIR}"
    "${GTEST_ROOT}/include"
    "${GTEST_ROOT}")
  target_link_libraries(qsharp_ep_tests PRIVATE
    "${ONNXRUNTIME_LIBRARY}")
  target_compile_definitions(qsharp_ep_tests PRIVATE
    QSHARP_EP_PLUGIN_PATH="$<TARGET_FILE:onnxruntime_qsharp_ep>"
    QSHARP_EP_MODEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/testdata/logistic_regression.onnx")
  target_compile_options(qsharp_ep_tests PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Werror)
  set_target_properties(qsharp_ep_tests PROPERTIES
    BUILD_RPATH "${ONNXRUNTIME_LIBRARY_DIR}")

  add_test(NAME qsharp_ep_tests COMMAND qsharp_ep_tests)
endif()
