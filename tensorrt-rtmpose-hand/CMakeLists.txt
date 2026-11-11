# RTMPose 专用几何和离线验证入口，复用公共 runtime。
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
add_library(rtmpose_reference STATIC src/reference.cpp)
target_compile_features(rtmpose_reference PUBLIC cxx_std_17)
target_include_directories(rtmpose_reference PUBLIC include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(rtmpose_reference PUBLIC ${OpenCV_LIBS})
add_executable(rtmpose_reference_test tests/test_reference.cpp)
target_link_libraries(rtmpose_reference_test PRIVATE rtmpose_reference)
add_test(NAME rtmpose_reference COMMAND rtmpose_reference_test)
find_package(OpenSSL REQUIRED COMPONENTS Crypto)
add_executable(rtmpose_fp32 apps/verify_fp32.cpp)
target_link_libraries(rtmpose_fp32 PRIVATE rtmpose_reference edge_trt_runtime OpenSSL::Crypto)

# 模型构建入口复用公共 builder；OpenCV 仅用于读取本模块 JSON。
if(NOT TARGET edge_trt_builder)
    message(FATAL_ERROR "RTMPose build tool requires EDGE_TRT_BUILD_BUILDER=ON")
endif()
add_executable(rtmpose_build apps/build_engine.cpp)
target_compile_features(rtmpose_build PRIVATE cxx_std_17)
target_include_directories(rtmpose_build PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(rtmpose_build PRIVATE edge_trt_builder ${OpenCV_LIBS} OpenSSL::Crypto)

# 性能入口独立于冻结的数值验证程序，不改变历史可执行文件。
add_executable(rtmpose_benchmark apps/benchmark.cpp)
target_link_libraries(rtmpose_benchmark PRIVATE rtmpose_reference edge_trt_runtime)

# INT8 数据策略与校准器保留在模型模块，不改变既有精度的入口。
add_library(rtmpose_calibrator STATIC src/calibrator.cpp)
target_link_libraries(rtmpose_calibrator PUBLIC rtmpose_reference edge_trt_builder OpenSSL::Crypto)
add_executable(rtmpose_int8 apps/build_int8_engine.cpp)
target_link_libraries(rtmpose_int8 PRIVATE rtmpose_calibrator)
add_executable(test_hand_calibrator tests/test_hand_calibrator.cpp)
target_link_libraries(test_hand_calibrator PRIVATE rtmpose_calibrator)
add_test(NAME hand_calibrator COMMAND test_hand_calibrator)

# 中间激活探针与正式模型入口隔离，仍复用公共构建与执行实现。
add_executable(rtmpose_probe apps/activation_probe.cpp)
target_link_libraries(rtmpose_probe PRIVATE rtmpose_reference edge_trt_builder edge_trt_runtime)
