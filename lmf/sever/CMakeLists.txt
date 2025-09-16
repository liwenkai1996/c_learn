cmake_minimum_required(VERSION 3.10)
project(InferenceServer)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找依赖包（OpenCV 由顶层提供，不在此处 find_package）
find_package(CUDA REQUIRED)
find_package(CURL REQUIRED)
find_package(PkgConfig REQUIRED)

# 查找 JSON 库
pkg_check_modules(JSONCPP jsoncpp)

# 设置包含目录（继承顶层的 OpenCV/TensorRT/CUDA 头路径）
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../infer/hpp
    ${CUDA_INCLUDE_DIRS}
    ${CURL_INCLUDE_DIRS}
    ${JSONCPP_INCLUDE_DIRS}
    /usr/local/include/drogon
    /usr/include/x86_64-linux-gnu/curl
    ${CMAKE_SOURCE_DIR}/third_part/nlohmann
)

# 设置库目录
link_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../infer/cpp
    ${JSONCPP_LIBRARY_DIRS}
)

# ==================== 编译推理引擎库 ====================

# 推理引擎源文件（包含 infer 模块实现，供服务器复用）
set(ENGINE_SOURCES
    cpp/InferenceEngine.cpp
    ${CMAKE_SOURCE_DIR}/lmf/infer/cpp/Tensorrt_inference_v10.cpp
    ${CMAKE_SOURCE_DIR}/lmf/infer/cpp/ImagePreprocessor.cpp
    ${CMAKE_SOURCE_DIR}/lmf/infer/cpp/TensorRTEngine.cpp
    ${CMAKE_SOURCE_DIR}/lmf/infer/cpp/DetectionPostprocessor.cpp
    ${CMAKE_SOURCE_DIR}/lmf/infer/cpp/LoggerV10.cpp
)

# 创建推理引擎静态库（不暴露在默认构建中，仅供 inference_server 依赖）
add_library(inference_engine STATIC ${ENGINE_SOURCES})
set_target_properties(inference_engine PROPERTIES EXCLUDE_FROM_ALL TRUE)
set_target_properties(inference_engine PROPERTIES EXCLUDE_FROM_DEFAULT_BUILD TRUE)

# 链接依赖库（OpenCV_LIBS 由顶层定义）
target_link_libraries(inference_engine
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    nvinfer
    nvonnxparser
    nvinfer_plugin
    cuda
    cudart
)

# 将第三方头标记为 SYSTEM，屏蔽其产生的告警
target_include_directories(inference_engine SYSTEM PRIVATE
    /home/opt/local/tensorrt/include
    /usr/local/cuda/include
    /usr/local/include/drogon
    /usr/include/x86_64-linux-gnu/curl
    ${CMAKE_SOURCE_DIR}/third_part/nlohmann
    /home/opt/local/opencv-4.8.0/release/include/opencv4
)

# ==================== 编译推理服务器 ====================

# 服务器源文件
set(SERVER_SOURCES
    cpp/InferenceServer.cpp
    cpp/main.cpp
)

# 创建推理服务器可执行文件
add_executable(inference_server ${SERVER_SOURCES})

# 查找 Drogon 库
find_package(Drogon REQUIRED)

# 链接依赖库
target_link_libraries(inference_server
    inference_engine
    Drogon::Drogon
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    ${CURL_LIBRARIES}
    ${JSONCPP_LIBRARIES}
)

# 同样对服务可执行目标应用 SYSTEM includes
target_include_directories(inference_server SYSTEM PRIVATE
    /home/opt/local/tensorrt/include
    /usr/local/cuda/include
    /usr/local/include/drogon
    /usr/include/x86_64-linux-gnu/curl
    ${CMAKE_SOURCE_DIR}/third_part/nlohmann
    /home/opt/local/opencv-4.8.0/release/include/opencv4
)

# 将项目根路径以编译期宏传入程序
target_compile_definitions(inference_server PRIVATE PROJECT_ROOT_PATH="${CMAKE_SOURCE_DIR}")

# 目标级头文件路径，确保能找到 nlohmann/json.hpp
target_include_directories(inference_server PRIVATE
    ${CMAKE_SOURCE_DIR}/third_part
)

# ==================== 编译测试客户端 ====================

# 客户端源文件
set(CLIENT_SOURCES
    cpp/test_client.cpp
)

# 创建测试客户端可执行文件
add_executable(test_client ${CLIENT_SOURCES})

# 链接依赖库
target_link_libraries(test_client
    ${OpenCV_LIBS}
    ${CURL_LIBRARIES}
    ${JSONCPP_LIBRARIES}
)

# 目标级头文件路径，确保能找到 nlohmann/json.hpp
target_include_directories(test_client PRIVATE
    ${CMAKE_SOURCE_DIR}/third_part/nlohmann
)

# ==================== 安装规则 ====================

# 安装推理服务器
install(TARGETS inference_server
    RUNTIME DESTINATION bin
)

# 安装测试客户端
install(TARGETS test_client
    RUNTIME DESTINATION bin
)

# 安装推理引擎库
install(TARGETS inference_engine
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

# 安装头文件
install(FILES 
    hpp/InferenceEngine.hpp
    DESTINATION include
)

# ==================== 编译选项 ====================

# 设置编译选项
target_compile_options(inference_engine PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -O3 -fPIC>
    $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra -O3 -fPIC>
    $<$<CXX_COMPILER_ID:MSVC>:/W3 /O2>
)

target_compile_options(inference_server PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -O3>
    $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra -O3>
    $<$<CXX_COMPILER_ID:MSVC>:/W3 /O2>
)

target_compile_options(test_client PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -O3>
    $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra -O3>
    $<$<CXX_COMPILER_ID:MSVC>:/W3 /O2>
)

# ==================== 输出信息 ====================

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Building inference engine library (reuse OpenCV from top-level)")
message(STATUS "Building inference server (Drogon + curl)")
message(STATUS "Building test client (curl)")
message(STATUS "Using Drogon HTTP server library (headers at /usr/local/include/drogon)")
message(STATUS "Using libcurl (headers at /usr/include/x86_64-linux-gnu/curl)")
message(STATUS "Using nlohmann/json headers at ${CMAKE_SOURCE_DIR}/third_part/nlohmann")

# ==================== 注意事项 ====================

message(WARNING "Note: Make sure Drogon library is properly installed and findable by CMake")
message(WARNING "Note: You may need to set DROGON_ROOT or Drogon_DIR to help CMake find Drogon")
