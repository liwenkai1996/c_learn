#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR服务测试用例
用于测试 /home/work/liwenkai/c_learn/ocr_main.py 启动的OCR服务
"""

import requests
import json
import time
import os
from typing import Dict, Any

# OCR服务配置
OCR_SERVICE_URL = "http://localhost:1578"
OCR_API_ENDPOINT = f"{OCR_SERVICE_URL}/api/ocr"
HEALTH_ENDPOINT = f"{OCR_SERVICE_URL}/api/health"

class OCRServiceTester:
    """OCR服务测试类"""
    
    def __init__(self, base_url: str = OCR_SERVICE_URL):
        self.base_url = base_url
        self.ocr_endpoint = f"{base_url}/api/ocr"
        self.health_endpoint = f"{base_url}/api/health"
    
    def check_service_health(self) -> bool:
        """检查服务是否正常运行"""
        try:
            response = requests.get(self.health_endpoint, timeout=5)
            if response.status_code == 200:
                result = response.json()
                print(f"✅ 服务健康检查通过: {result['message']}")
                return True
            else:
                print(f"❌ 服务健康检查失败: HTTP {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"❌ 无法连接到OCR服务: {e}")
            return False
    
    def test_ocr_single_image(self, image_path: str, **kwargs) -> Dict[str, Any]:
        """测试单张图片OCR识别"""
        print(f"\n🔍 开始测试图片: {image_path}")
        
        # 构建请求数据
        request_data = {
            "image_path": image_path,
            **kwargs
        }
        
        try:
            start_time = time.time()
            response = requests.post(
                self.ocr_endpoint,
                json=request_data,
                timeout=30
            )
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ OCR识别成功!")
                print(f"⏱️  耗时: {end_time - start_time:.2f}秒")
                
                # 调试：打印实际响应格式
                print(f"🔍 响应格式: {result}")
                
                if result.get('body') and result['body'].get('text_count') is not None:
                    print(f"📊 识别到 {result['body']['text_count']} 个文本")
                    
                    # 显示识别结果
                    if result['body'].get('results'):
                        print("📝 识别结果:")
                        for i, item in enumerate(result['body']['results'], 1):
                            print(f"  {i}. 文本: '{item['text']}' (置信度: {item['score']:.3f})")
                    else:
                        print("📝 未识别到任何文本")
                else:
                    print("📝 响应数据格式异常")
                    print(f"   实际响应: {result}")
                
                return result
            else:
                print(f"❌ OCR识别失败: HTTP {response.status_code}")
                try:
                    error_info = response.json()
                    print(f"   错误信息: {error_info.get('message', '未知错误')}")
                except:
                    print(f"   响应内容: {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"❌ 请求失败: {e}")
            return None
    


def main():
    """主测试函数"""
    print("🚀 OCR服务测试开始")
    print("=" * 50)
    
    # 创建测试器
    tester = OCRServiceTester()
    
   
    print("\n2️⃣ 测试单张图片OCR识别")
    test_image = "/home/work/liwenkai/c_learn/图片-SM- (122).jpg"
    tester.test_ocr_single_image(test_image)
    
    
    print("\n" + "=" * 50)
    print("✅ OCR服务测试完成")



    



if __name__ == "__main__":
    # 运行主测试
    main()

