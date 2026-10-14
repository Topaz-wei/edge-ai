cmake_minimum_required(VERSION 3.18)

project(
    tensorrt_resnet
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)

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

add_library(trt_runner STATIC src/trt_runner.cpp)
target_include_directories(trt_runner PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${TENSORRT_INCLUDE_DIR})
target_link_libraries(trt_runner PUBLIC ${TENSORRT_LIBRARY} CUDA::cudart)

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
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${TENSORRT_INCLUDE_DIR}
)

target_link_libraries(
    yolo_infer_raw
    PRIVATE
        trt_runner
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
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Compile the existing implementations without their standalone entry points.
add_executable(
    yolo_infer
    apps/yolo_infer.cpp
    apps/yolo_preprocess.cpp
    apps/postprocess_raw.cpp
)

# 仅对 yolo_infer 目标定义宏，禁用其共用源文件中的独立 main
target_compile_definitions(yolo_infer PRIVATE YOLO_NO_MAIN)

target_include_directories(
    yolo_infer
    PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${TENSORRT_INCLUDE_DIR}
)

target_link_libraries(
    yolo_infer
    PRIVATE
        trt_runner
        ${OpenCV_LIBS}
        ${TENSORRT_LIBRARY}
        CUDA::cudart
)

add_executable(yolo_preprocess apps/yolo_preprocess.cpp)
target_include_directories(yolo_preprocess PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(yolo_preprocess PRIVATE ${OpenCV_LIBS})

include(CTest)
if(BUILD_TESTING)
    add_executable(test_preprocess_cpp tests/test_preprocess_cpp.cpp apps/yolo_preprocess.cpp)
    target_compile_definitions(test_preprocess_cpp PRIVATE YOLO_NO_MAIN)
    target_include_directories(test_preprocess_cpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(test_preprocess_cpp PRIVATE ${OpenCV_LIBS})
    add_test(NAME preprocess_cpp COMMAND test_preprocess_cpp ${CMAKE_CURRENT_SOURCE_DIR}/data/images/test.jpg)
endif()

if(BUILD_TESTING)
    # Run explicitly with a locally captured pre-refactor raw output as baseline.
    add_executable(test_trt_runner tests/test_trt_runner.cpp apps/yolo_preprocess.cpp apps/postprocess_raw.cpp)
    target_compile_definitions(test_trt_runner PRIVATE YOLO_NO_MAIN)
    target_include_directories(test_trt_runner PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(test_trt_runner PRIVATE trt_runner ${OpenCV_LIBS})
endif()

add_executable(benchmark_yolo
    apps/benchmark_yolo.cpp
    apps/yolo_preprocess.cpp
    apps/postprocess_raw.cpp
)
target_compile_definitions(benchmark_yolo PRIVATE YOLO_NO_MAIN)
target_include_directories(benchmark_yolo PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(benchmark_yolo PRIVATE trt_runner ${OpenCV_LIBS})

# CUDA v3 is optional so CPU/reference builds remain available without nvcc.
option(YOLO_BUILD_CUDA_PIPELINE "Build CUDA preprocessing/postprocessing and Graph pipeline" ON)
if(YOLO_BUILD_CUDA_PIPELINE)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 87)
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    # NVTX3 is header-only. Its Linux injection loader uses dlopen/dlsym.
    find_path(
        NVTX3_INCLUDE_DIR
        NAMES
            nvtx3/nvToolsExt.h
        HINTS
            ${CUDAToolkit_INCLUDE_DIRS}
    )

    if(NOT NVTX3_INCLUDE_DIR)
        message(FATAL_ERROR "NVTX3 headers not found; install NVTX3 or set NVTX3_INCLUDE_DIR")
    endif()

    add_library(
        cuda_yolo
        STATIC
        src/cuda_yolo.cu
        src/cuda_pipeline.cpp
    )
    target_include_directories(
        cuda_yolo
        PUBLIC
        ${OpenCV_INCLUDE_DIRS}
        PRIVATE
        ${NVTX3_INCLUDE_DIR}
    )
    target_link_libraries(
        cuda_yolo
        PUBLIC
        trt_runner
        ${OpenCV_LIBS}
        PRIVATE
        ${CMAKE_DL_LIBS}
    )
    target_compile_options(
        cuda_yolo
        PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>
    )
endif()
if(YOLO_BUILD_CUDA_PIPELINE AND BUILD_TESTING)
    add_executable(
        test_cuda_yolo
        tests/test_cuda_yolo.cpp
        apps/yolo_preprocess.cpp
        apps/postprocess_raw.cpp
    )
    target_compile_definitions(
        test_cuda_yolo
        PRIVATE
        YOLO_NO_MAIN
    )
    target_link_libraries(
        test_cuda_yolo
        PRIVATE
        cuda_yolo
    )
endif()
if(YOLO_BUILD_CUDA_PIPELINE)
    add_executable(
        yolo_infer_cuda
        apps/yolo_infer_cuda.cpp
        apps/postprocess_raw.cpp
    )
    target_compile_definitions(
        yolo_infer_cuda
        PRIVATE
        YOLO_NO_MAIN
    )
    target_link_libraries(
        yolo_infer_cuda
        PRIVATE
        cuda_yolo
    )
    add_executable(
        benchmark_yolo_cuda
        apps/benchmark_yolo_cuda.cpp
        apps/yolo_preprocess.cpp
        apps/postprocess_raw.cpp
    )
    target_compile_definitions(
        benchmark_yolo_cuda
        PRIVATE
        YOLO_NO_MAIN
    )
    target_link_libraries(
        benchmark_yolo_cuda
        PRIVATE
        cuda_yolo
    )
endif()
