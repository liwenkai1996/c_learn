#include "../hpp/InferenceServer.hpp"
#include <iostream>
#include <signal.h>
#include <unistd.h>
// 不再使用命令行解析，改为函数传参/环境变量方式
#include <cstdlib>
#include <fstream>
#include "../../third_part/nlohmann/json.hpp"

// 全局服务器实例
std::unique_ptr<InferenceServer> g_server = nullptr;

// 信号处理函数
void signal_handler(int signal) {
    std::cout << "\nReceived signal " << signal << ", shutting down..." << std::endl;
    
    if (g_server) {
        g_server->stop();
    }
    
    exit(0);
}

// 通过函数参数启动服务器
int run_inference_server(const std::string& model_path, int port, int device_id, int num_threads);

// 简单配置结构
struct ServerConfigJson {
    std::string model_path;
    int port{8080};
    int device_id{0};
    int threads{4};
};

static bool load_config_from_file(const std::string& path, ServerConfigJson& out) {
    try {
        std::ifstream ifs(path);
        if (!ifs.is_open()) {
            std::cerr << "Error: cannot open config file: " << path << std::endl;
            return false;
        }
        nlohmann::json j;
        ifs >> j;
        if (!j.contains("model_path")) {
            std::cerr << "Error: config missing required field: model_path" << std::endl;
            return false;
        }
        out.model_path = j.at("model_path").get<std::string>();
        if (j.contains("port")) out.port = j.at("port").get<int>();
        if (j.contains("device_id")) out.device_id = j.at("device_id").get<int>();
        if (j.contains("threads")) out.threads = j.at("threads").get<int>();
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error parsing config file: " << e.what() << std::endl;
        return false;
    }
}

int main(int, char**) {
    // 设置信号处理
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    // 使用编译期注入的项目根路径拼接配置文件路径
    std::string config_path = std::string(PROJECT_ROOT_PATH) + "/config/server_config.json";

    ServerConfigJson cfg;
    if (!load_config_from_file(config_path, cfg)) {
        return 1;
    }

    return run_inference_server(cfg.model_path, cfg.port, cfg.device_id, cfg.threads);
}

int run_inference_server(const std::string& model_path, int port, int device_id, int num_threads) {
    // 参数校验
    if (port <= 0 || port > 65535) {
        std::cerr << "Error: Invalid port number: " << port << std::endl;
        return 1;
    }
    if (device_id < 0) {
        std::cerr << "Error: Invalid device ID: " << device_id << std::endl;
        return 1;
    }
    if (num_threads <= 0 || num_threads > 32) {
        std::cerr << "Error: Invalid number of threads: " << num_threads << std::endl;
        return 1;
    }

    // 打印配置信息
    std::cout << "=== Inference Server Configuration ===" << std::endl;
    std::cout << "Model path: " << model_path << std::endl;
    std::cout << "Server port: " << port << std::endl;
    std::cout << "GPU device: " << device_id << std::endl;
    std::cout << "Worker threads: " << num_threads << std::endl;
    std::cout << "=====================================" << std::endl;

    try {
        // 创建推理服务器
        g_server = std::make_unique<InferenceServer>();

        // 设置回调函数
        g_server->set_request_callback([](const std::string& request) {
            std::cout << "Received request: " << request.substr(0, 100) << "..." << std::endl;
        });

        g_server->set_error_callback([](const std::string& error) {
            std::cerr << "Error: " << error << std::endl;
        });

        // 初始化服务器
        std::cout << "Initializing server..." << std::endl;
        if (!g_server->initialize(model_path, device_id, port, num_threads)) {
            std::cerr << "Failed to initialize server" << std::endl;
            return 1;
        }

        // 启动服务器
        std::cout << "Starting server..." << std::endl;
        if (!g_server->start()) {
            std::cerr << "Failed to start server" << std::endl;
            return 1;
        }

        std::cout << "Server started successfully!" << std::endl;
        std::cout << "Server is running on port " << port << std::endl;
        std::cout << "Press Ctrl+C to stop the server" << std::endl;

        // 主循环
        while (g_server->is_running()) {
            sleep(1);

            // 定期打印状态信息
            static int counter = 0;
            if (++counter % 60 == 0) {  // 每分钟打印一次
                std::cout << "Server status: " << g_server->get_status() << std::endl;
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "Server stopped" << std::endl;
    return 0;
}
