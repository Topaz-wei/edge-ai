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
    yolo_build
    apps/build_engine.cpp
)

target_include_directories(
    yolo_build
    PRIVATE
        ${TENSORRT_INCLUDE_DIR}
)

target_link_libraries(
    yolo_build
    PRIVATE
        ${TENSORRT_LIBRARY}
        ${TENSORRT_ONNX_PARSER_LIBRARY}
        CUDA::cudart
)

add_executable(
    yolo_infer_raw
    apps/infer_raw.cpp
)

target_include_directories(
    yolo_infer_raw
    PRIVATE
        ${TENSORRT_INCLUDE_DIR}
)

target_link_libraries(
    yolo_infer_raw
    PRIVATE
        ${TENSORRT_LIBRARY}
        ${TENSORRT_ONNX_PARSER_LIBRARY}
        CUDA::cudart
)

add_executable(
    yolo_postprocess_raw
    apps/postprocess_raw.cpp
)

target_include_directories(
    yolo_postprocess_raw
    PRIVATE
        ${PROJECT_INCLUDE_DIR}
)
