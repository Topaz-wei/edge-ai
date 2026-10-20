find_path(TensorRT_INCLUDE_DIR NvInfer.h HINTS /usr/include/aarch64-linux-gnu /usr/include)
find_library(TensorRT_LIBRARY nvinfer HINTS /usr/lib/aarch64-linux-gnu)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TensorRT REQUIRED_VARS TensorRT_INCLUDE_DIR TensorRT_LIBRARY)
if(TensorRT_FOUND AND NOT TARGET TensorRT::Runtime)
    add_library(TensorRT::Runtime UNKNOWN IMPORTED)
    set_target_properties(TensorRT::Runtime PROPERTIES
        IMPORTED_LOCATION "${TensorRT_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}")
endif()
