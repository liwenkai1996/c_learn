cmake_minimum_required(VERSION 3.10)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# 设置项目名称
project(CppLearning)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置编译选项
# 默认使用Debug模式，可以通过 -DCMAKE_BUILD_TYPE=Release 切换到Release模式
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

# 设置路径
set(opencv_home /home/opt/local/opencv-4.8.0/release)
set(cuda_home /usr/local/cuda)
set(tensorrt_home /home/opt/local/tensorrt)

# 添加包含目录
INCLUDE_DIRECTORIES(
    ${opencv_home}/include
    ${opencv_home}/include/opencv4
    ${cuda_home}/include
    ${tensorrt_home}/include
)

# 添加库目录
LINK_DIRECTORIES(
    ${opencv_home}/lib
    ${cuda_home}/lib64
    ${tensorrt_home}/lib
)

# 设置OpenCV库
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

set(OpenCV_FOUND TRUE)
message(STATUS "OpenCV路径设置完成: ${opencv_home}")

# 检查OpenCV是否找到
if(OpenCV_FOUND)
    message(STATUS "OpenCV库配置完成:")
    message(STATUS "  路径: ${opencv_home}")
    message(STATUS "  包含目录: ${opencv_home}/include")
    message(STATUS "  库目录: ${opencv_home}/lib")
    
    # 子目录：lmf/infer（内部包含 tensorrt_inference_test_v10）
    add_subdirectory(lmf/infer)

else()
    message(WARNING "OpenCV库未找到，将跳过opencv_image_reader的编译")
endif()

# 追加：包含推理服务器子项目
add_subdirectory(lmf/sever)
message(STATUS "已包含子目录: lmf/sever (生成 inference_server 与 test_client 目标)")

# 添加OCR子项目
add_subdirectory(ocr/infer)
message(STATUS "已包含子目录: ocr/infer (生成 ocr_demo 目标)")

# 添加清理目标
add_custom_target(clean-all
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}
    COMMENT "Cleaning build directory"
)

# 打印帮助信息
message(STATUS "项目配置完成！")
message(STATUS "可用的目标:")
message(STATUS "  tensorrt_inference_test_v10 - TensorRT推理测试程序")
message(STATUS "  inference_server - 推理HTTP服务 (在 lmf/sever)")
message(STATUS "  test_client - 推理服务测试客户端 (在 lmf/sever)")
message(STATUS "  ocr_demo - PaddleOCR推理程序 (在 ocr/infer)")
message(STATUS "")
message(STATUS "使用方法:")
message(STATUS "  mkdir build && cd build")
message(STATUS "  cmake ..")
message(STATUS "  make -j")
message(STATUS "")
message(STATUS "运行程序:")
message(STATUS "  ./tensorrt_inference_test_v10")
message(STATUS "  ./inference_server  (或使用 lmf/sever/deploy.sh)")
message(STATUS "  ./ocr_demo  (或使用 ocr/infer/test_ocr.sh)")
