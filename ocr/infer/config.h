#ifndef OCR_CONFIG_H
#define OCR_CONFIG_H

#include <string>

namespace OCRConfig {
    // 模型文件路径配置
    const std::string DET_MODEL_PATH = "/root/.paddlex/official_models/PP-OCRv5_server_det/PP-OCRv5_server_det.trt";
    const std::string REC_MODEL_PATH = "/root/.paddlex/official_models/PP-OCRv5_server_rec/PP-OCRv5_server_rec.trt";
    const std::string ORI_MODEL_PATH = "/root/.paddlex/official_models/PP-LCNet_x1_0_textline_ori/PP-OCRv5_server_ori.trt";
    const std::string DICT_PATH = "/home/work/liwenkai/c_learn/ocr/infer/final_dict.txt";
    
    // 图像路径配置
    const std::string IMAGE_PATH = "/home/work/liwenkai/c_learn/图片-SM- (13).jpg";
    
    // 输出配置
    const std::string OUTPUT_PATH = "/home/work/liwenkai/c_learn/ocr_result.jpg";
    
    // 推理参数配置
    const float DET_THRESHOLD = 0.3f;
    const float DET_BOX_THRESHOLD = 0.6f;
    const float UNCLIP_RATIO = 1.5f; 
    const float REC_THRESHOLD = 0.5f;
    
    // 保存配置
    const bool SAVE_RESULT = true;
}

#endif // OCR_CONFIG_H
