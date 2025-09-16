#include <iostream>
#include <opencv2/opencv.hpp>
#include <curl/curl.h>
#include <json/json.h>
#include <fstream>
#include <filesystem> // Added for std::filesystem::path

// 回调函数：接收 HTTP 响应数据
size_t write_callback(void* contents, size_t size, size_t nmemb, std::string* userp) {
    userp->append((char*)contents, size * nmemb);
    return size * nmemb;
}

// 发送推理请求
bool send_inference_request(const std::string& server_url, 
                          const std::string& image_path,
                          float confidence_threshold = 0.5f) {
    CURL* curl = curl_easy_init();
    if (!curl) {
        std::cerr << "Failed to initialize CURL" << std::endl;
        return false;
    }
    
    // 读取图像文件
    std::ifstream image_file(image_path, std::ios::binary);
    if (!image_file.is_open()) {
        std::cerr << "Failed to open image file: " << image_path << std::endl;
        curl_easy_cleanup(curl);
        return false;
    }
    
    // 获取文件大小
    image_file.seekg(0, std::ios::end);
    size_t file_size = image_file.tellg();
    image_file.seekg(0, std::ios::beg);
    
    // 读取图像数据
    std::vector<char> image_data(file_size);
    image_file.read(image_data.data(), file_size);
    image_file.close();
    
    // 构建请求数据
    std::string boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW";
    std::string request_data;
    
    // 添加图像数据
    request_data += "--" + boundary + "\r\n";
    request_data += "Content-Disposition: form-data; name=\"image\"; filename=\"" + 
                   std::filesystem::path(image_path).filename().string() + "\"\r\n";
    request_data += "Content-Type: image/jpeg\r\n\r\n";
    request_data.append(image_data.begin(), image_data.end());
    request_data += "\r\n";
    
    // 添加置信度阈值
    request_data += "--" + boundary + "\r\n";
    request_data += "Content-Disposition: form-data; name=\"confidence_threshold\"\r\n\r\n";
    request_data += std::to_string(confidence_threshold) + "\r\n";
    
    request_data += "--" + boundary + "--\r\n";
    
    // 设置 CURL 选项
    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, ("Content-Type: multipart/form-data; boundary=" + boundary).c_str());
    
    curl_easy_setopt(curl, CURLOPT_URL, server_url.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, request_data.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, request_data.length());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    
    std::string response;
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
    
    // 执行请求
    CURLcode res = curl_easy_perform(curl);
    
    // 清理
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    
    if (res != CURLE_OK) {
        std::cerr << "CURL request failed: " << curl_easy_strerror(res) << std::endl;
        return false;
    }
    
    // 解析响应
    std::cout << "Response received: " << response.length() << " bytes" << std::endl;
    
    try {
        Json::Value root;
        Json::Reader reader;
        
        if (reader.parse(response, root)) {
            if (root.isMember("status_code") && root["status_code"].asInt() == 0) {
                std::cout << "Inference successful!" << std::endl;
                std::cout << "Inference time: " << root["inference_time_ms"].asDouble() << " ms" << std::endl;
                
                if (root.isMember("detections")) {
                    const Json::Value& detections = root["detections"];
                    std::cout << "Detections found: " << detections.size() << std::endl;
                    
                    for (const auto& det : detections) {
                        std::cout << "  - Class: " << det["class_id"].asInt()
                                  << ", Confidence: " << det["confidence"].asFloat()
                                  << ", Box: (" << det["x"].asInt() << "," << det["y"].asInt()
                                  << "," << det["width"].asInt() << "," << det["height"].asInt() << ")"
                                  << std::endl;
                    }
                }
            } else {
                std::cerr << "Inference failed: " << root["status_message"].asString() << std::endl;
                return false;
            }
        } else {
            std::cerr << "Failed to parse JSON response" << std::endl;
            return false;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error parsing response: " << e.what() << std::endl;
        return false;
    }
    
    return true;
}

// 测试健康检查
bool test_health_check(const std::string& server_url) {
    CURL* curl = curl_easy_init();
    if (!curl) {
        return false;
    }
    
    std::string response;
    curl_easy_setopt(curl, CURLOPT_URL, (server_url + "/health").c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
    
    CURLcode res = curl_easy_perform(curl);
    curl_easy_cleanup(curl);
    
    if (res != CURLE_OK) {
        return false;
    }
    
    std::cout << "Health check response: " << response << std::endl;
    return true;
}

// 测试状态查询
bool test_status_query(const std::string& server_url) {
    CURL* curl = curl_easy_init();
    if (!curl) {
        return false;
    }
    
    std::string response;
    curl_easy_setopt(curl, CURLOPT_URL, (server_url + "/status").c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
    
    CURLcode res = curl_easy_perform(curl);
    curl_easy_cleanup(curl);
    
    if (res != CURLE_OK) {
        return false;
    }
    
    std::cout << "Status response: " << response << std::endl;
    return true;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <server_url> <image_path> [confidence_threshold]" << std::endl;
        std::cout << "Example: " << argv[0] << " http://localhost:8080 test.jpg 0.5" << std::endl;
        return 1;
    }
    
    std::string server_url = argv[1];
    std::string image_path = argv[2];
    float confidence_threshold = (argc > 3) ? std::stof(argv[3]) : 0.5f;
    
    std::cout << "=== Inference Client Test ===" << std::endl;
    std::cout << "Server URL: " << server_url << std::endl;
    std::cout << "Image path: " << image_path << std::endl;
    std::cout << "Confidence threshold: " << confidence_threshold << std::endl;
    std::cout << "=============================" << std::endl;
    
    // 初始化 CURL
    curl_global_init(CURL_GLOBAL_ALL);
    
    try {
        // 测试健康检查
        std::cout << "\n1. Testing health check..." << std::endl;
        if (!test_health_check(server_url)) {
            std::cout << "Health check failed - server may not be running" << std::endl;
        }
        
        // 测试状态查询
        std::cout << "\n2. Testing status query..." << std::endl;
        if (!test_status_query(server_url)) {
            std::cout << "Status query failed" << std::endl;
        }
        
        // 测试推理请求
        std::cout << "\n3. Testing inference request..." << std::endl;
        if (send_inference_request(server_url, image_path, confidence_threshold)) {
            std::cout << "Inference test completed successfully!" << std::endl;
        } else {
            std::cout << "Inference test failed!" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
    }
    
    // 清理 CURL
    curl_global_cleanup();
    
    std::cout << "\nTest completed!" << std::endl;
    return 0;
}
