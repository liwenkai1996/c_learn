import requests
import os
import json
from typing import List, Dict, Any, Optional

class DETRInferenceAPI:
    def __init__(self, base_url: str = "http://10.0.20.200:7815"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.timeout = 30
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """通用请求方法"""
        url = f"{self.base_url}{endpoint}"
        try:
            if method.upper() == 'GET':
                response = self.session.get(url, **kwargs)
            elif method.upper() == 'POST':
                response = self.session.post(url, **kwargs)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            return response
        except requests.exceptions.RequestException as e:
            raise Exception(f"Request failed: {e}")
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查接口"""
        endpoint = "/api/detr/health"
        response = self._make_request('GET', endpoint)
        
        print(f"Health Check - Status Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Health Check - Response: {result}")
            return result
        else:
            print(f"Health Check - Error: {response.text}")
            return {"error": response.text, "status_code": response.status_code}
    
    def get_status(self) -> Dict[str, Any]:
        """获取服务状态接口"""
        endpoint = "/api/detr/status"
        response = self._make_request('GET', endpoint)
        
        print(f"Status - Status Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Status - Response: {result}")
            return result
        else:
            print(f"Status - Error: {response.text}")
            return {"error": response.text, "status_code": response.status_code}
    
    def single_inference(self, image_path: str, confidence_threshold: float = 0.3) -> Dict[str, Any]:
        """单张图片推理接口"""
        endpoint = "/api/detr/inference"
        
        # 检查文件是否存在
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        try:
            with open(image_path, 'rb') as image_file:
                files = {'image': image_file}
                data = {'confidence_threshold': confidence_threshold}
                
                response = self._make_request('POST', endpoint, files=files, data=data)
                
                print(f"Single Inference - Status Code: {response.status_code}")
                if response.status_code == 200:
                    result = response.json()
                    print(f"Single Inference - Response: {result}")
                    return result
                else:
                    print(f"Single Inference - Error: {response.text}")
                    return {"error": response.text, "status_code": response.status_code}
                    
        except Exception as e:
            raise Exception(f"Single inference failed: {e}")
    
    def batch_inference(self, image_paths: List[str], confidence_threshold: float = 0.3) -> Dict[str, Any]:
        """批量图片推理接口"""
        endpoint = "/api/detr/batch_inference"
        
        # 检查所有文件是否存在
        for image_path in image_paths:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
        
        try:
            files = []
            for i, image_path in enumerate(image_paths):
                with open(image_path, 'rb') as image_file:
                    files.append(('images', (os.path.basename(image_path), image_file.read(), 'image/jpeg')))
            
            data = {'confidence_threshold': confidence_threshold}
            
            response = self._make_request('POST', endpoint, files=files, data=data)
            
            print(f"Batch Inference - Status Code: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"Batch Inference - Response: {result}")
                return result
            else:
                print(f"Batch Inference - Error: {response.text}")
                return {"error": response.text, "status_code": response.status_code}
                
        except Exception as e:
            raise Exception(f"Batch inference failed: {e}")


def main():
    """主函数示例"""
    # 创建API客户端
    api_client = DETRInferenceAPI()
    
    # 测试图片路径
    test_image = "/home/work/liwenkai/c_learn/test/DJI_20250326134948_0002_V.JPG"
    test_images = [
        "/home/work/liwenkai/c_learn/test/DJI_20250326134948_0002_V.JPG",
        "/home/work/liwenkai/c_learn/test/DJI_20250326135023_0004_V.JPG",
        "/home/work/liwenkai/c_learn/test/DJI_20250326140133_0046_V.JPG",
    ]
    
    try:
        # # 1. 健康检查
        # print("=== Health Check ===")
        # health_result = api_client.health_check()
        
        # # 2. 获取状态
        # print("\n=== Get Status ===")
        # status_result = api_client.get_status()
        
        # # 3. 单张图片推理
        # print("\n=== Single Inference ===")
        # if os.path.exists(test_image):
        #     single_result = api_client.single_inference(test_image, confidence_threshold=0.3)
        
        # 4. 批量图片推理
        print("\n=== Batch Inference ===")
        if len(test_images) > 0:
            batch_result = api_client.batch_inference(test_images, confidence_threshold=0.3)
        
    except Exception as e:
        print(f"Error in main: {e}")


if __name__ == "__main__":
    main()