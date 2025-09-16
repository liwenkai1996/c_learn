cmake_minimum_required(VERSION 3.10)

project(PaddleOCR_Cpp LANGUAGES CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置编译选项
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -Wno-deprecated-declarations -Wno-deprecated")
    message(STATUS "编译模式: Debug (包含调试信息)")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Wno-deprecated-declarations -Wno-deprecated")
    message(STATUS "编译模式: Release (优化模式)")
endif()

# 设置路径 (与项目根目录保持一致)
set(opencv_home /home/opt/local/opencv-4.8.0/release)
set(cuda_home /usr/local/cuda)
set(tensorrt_home /home/opt/local/tensorrt)

# 检查TensorRT头文件
if(NOT EXISTS "${tensorrt_home}/include/NvInfer.h")
    message(FATAL_ERROR "TensorRT headers not found at ${tensorrt_home}/include")
endif()

# 检查TensorRT库文件
if(NOT EXISTS "${tensorrt_home}/lib/libnvinfer.so")
    message(FATAL_ERROR "TensorRT libraries not found at ${tensorrt_home}/lib")
endif()

# 包含目录
include_directories(
    ${opencv_home}/include
    ${opencv_home}/include/opencv4
    ${cuda_home}/include
    ${tensorrt_home}/include
    ${CMAKE_SOURCE_DIR}/third_part/clipper
    hpp
    .
)

# 链接目录
link_directories(
    ${opencv_home}/lib
    ${cuda_home}/lib64
    ${tensorrt_home}/lib
)

# 设置OpenCV库 (与项目根目录保持一致)
set(OpenCV_LIBS 
    opencv_core 
    opencv_imgproc 
    opencv_highgui 
    opencv_imgcodecs
    opencv_videoio
    opencv_features2d
    opencv_calib3d
    opencv_flann
    opencv_ml
    opencv_objdetect
    opencv_photo
    opencv_stitching
    opencv_video
    opencv_dnn
    opencv_gapi
)

# 源文件
set(SOURCES
    cpp/engine.cpp
    cpp/utils.cpp
    cpp/ocr_detector.cpp
    cpp/ocr_orientation.cpp
    cpp/ocr_recognizer.cpp
    cpp/ocr_pipeline.cpp
    cpp/main.cpp
    ${CMAKE_SOURCE_DIR}/third_part/clipper/clipper.cpp
)

# 创建可执行文件
add_executable(ocr_demo ${SOURCES})

# 链接库 (适配TensorRT 10+)
target_link_libraries(ocr_demo
    ${OpenCV_LIBS}
    nvinfer
    nvonnxparser
    nvinfer_plugin
    cuda
    cudart
)

# 设置输出目录
set_target_properties(ocr_demo PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

# 安装目标
install(TARGETS ocr_demo DESTINATION bin)

# 打印配置信息
message(STATUS "OCR项目配置完成:")
message(STATUS "  OpenCV路径: ${opencv_home}")
message(STATUS "  CUDA路径: ${cuda_home}")
message(STATUS "  TensorRT路径: ${tensorrt_home}")
message(STATUS "  编译模式: ${CMAKE_BUILD_TYPE}")
message(STATUS "")
message(STATUS "可用的目标:")
message(STATUS "  ocr_demo - PaddleOCR推理程序")
message(STATUS "")
message(STATUS "使用方法:")
message(STATUS "  ./ocr_demo <det_model> <rec_model> <dict> [image]")
message(STATUS "  示例: ./ocr_demo PP-OCRv5_server_det.trt PP-OCRv5_server_rec.trt dict.txt test.jpg")
