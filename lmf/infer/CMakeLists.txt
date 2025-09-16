cmake_minimum_required(VERSION 3.10)

project(TensorRTInfer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(tensorrt_inference_test_v10
    test/tensorrt_inference_test.cpp
    cpp/Tensorrt_inference_v10.cpp
    cpp/ImagePreprocessor.cpp
    cpp/TensorRTEngine.cpp
    cpp/DetectionPostprocessor.cpp
    cpp/LoggerV10.cpp
)

target_include_directories(tensorrt_inference_test_v10 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/hpp
)

target_link_libraries(tensorrt_inference_test_v10
    ${OpenCV_LIBS}
    nvinfer
    nvonnxparser
    nvinfer_plugin
    cuda
    cudart
)

set_target_properties(tensorrt_inference_test_v10 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)
