cmake_minimum_required(VERSION 3.18)

project(
    tensorrt_resnet
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

find_path(
    TENSORRT_INCLUDE_DIR
    NvInfer.h
    HINTS
        /usr/include/aarch64-linux-gnu
        /usr/include
)

find_library(
    TENSORRT_LIBRARY
    nvinfer
    HINTS
        /usr/lib/aarch64-linux-gnu
)

find_library(
    TENSORRT_ONNX_PARSER_LIBRARY
    nvonnxparser
    HINTS
        /usr/lib/aarch64-linux-gnu
)

if(NOT TENSORRT_ONNX_PARSER_LIBRARY)
    message(FATAL_ERROR "TensorRT ONNX parser not found")
endif()

if(NOT TENSORRT_INCLUDE_DIR)
    message(FATAL_ERROR "TensorRT headers not found")
endif()

if(NOT TENSORRT_LIBRARY)
    message(FATAL_ERROR "TensorRT library not found")
endif()

add_executable(
    trt_infer
    apps/infer.cpp
)

target_include_directories(
    trt_infer
    PRIVATE
        ${TENSORRT_INCLUDE_DIR}
)

target_link_libraries(
    trt_infer
    PRIVATE
        ${TENSORRT_LIBRARY}
        CUDA::cudart
)

add_executable(
    trt_build
    apps/build_engine.cpp
)

target_include_directories(
    trt_build
    PRIVATE
        ${TENSORRT_INCLUDE_DIR}
)

target_link_libraries(
    trt_build
    PRIVATE
        ${TENSORRT_LIBRARY}
        ${TENSORRT_ONNX_PARSER_LIBRARY}
        CUDA::cudart
)

add_executable(
    trt_infer_dynamic
    apps/infer_dynamic.cpp
)

target_include_directories(
    trt_infer_dynamic
    PRIVATE
        ${TENSORRT_INCLUDE_DIR}
)

target_link_libraries(
    trt_infer_dynamic
    PRIVATE
        ${TENSORRT_LIBRARY}
        CUDA::cudart
)

add_executable(
    trt_benchmark
    apps/benchmark_dynamic.cpp
)

target_include_directories(
    trt_benchmark
    PRIVATE
        ${TENSORRT_INCLUDE_DIR}
)

target_link_libraries(
    trt_benchmark
    PRIVATE
        ${TENSORRT_LIBRARY}
        CUDA::cudart
)