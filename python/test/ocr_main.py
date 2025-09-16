import logging
import os
import uvicorn
from fastapi import FastAPI
from paddleocr import PaddleOCR
import cv2
import requests
import numpy as np
from pydantic import BaseModel
import gc
# import torch

LOGGING_NAME = 'ocr_task'
fmt = '%(asctime)s - %(levelname)s - %(message)s'
LOGGER = logging.getLogger(LOGGING_NAME)
LOGGER.setLevel(logging.DEBUG)

log_path = os.path.abspath('./logger_ocr.txt')
file_handler = logging.FileHandler(log_path, encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(fmt))
LOGGER.addHandler(file_handler)

consoleLog = logging.StreamHandler()
consoleLog.setLevel(logging.DEBUG)
consoleLog.setFormatter(logging.Formatter(fmt))
LOGGER.addHandler(consoleLog)


OCR_CONFIG = {
    'use_doc_orientation_classify': False,
    'use_doc_unwarping': False,
    'use_textline_orientation': True,
    'text_det_unclip_ratio': 1.5,
    'text_det_box_thresh': 0.6,
    'text_det_thresh': 0.3,
    'text_rec_score_thresh': 0,
}

OCR_SCORE_THRESH = 0.3
CUT_SIZE = 1920
OVERLAP_RATIO = 0.1
SERVER_HOST = '0.0.0.0'
SERVER_PORT = 1578

IS_HARD = True
HARD_PATH = "/mnt/nfs_105/txy/images"

# 简化的请求模型
class OCRRequest(BaseModel):
    image_path: str
    cut_size: int = CUT_SIZE
    overlap_ratio: float = OVERLAP_RATIO
    force_crop: bool = False


def url_to_cv2_image(url):
    resp = requests.get(url, timeout=10)
    img_array = np.frombuffer(resp.content, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img

class OcrService:
    def __init__(self):
        self.ocr = PaddleOCR(**OCR_CONFIG)
        self.app = FastAPI()
        LOGGER.info("OCR模型已加载并常驻内存")


    @staticmethod
    def crop_image_with_overlap(img, cut_size=CUT_SIZE, overlap_ratio=OVERLAP_RATIO):
        h, w = img.shape[:2]
        overlap = int(cut_size * overlap_ratio)
        crops = []
        y_starts = list(range(0, h, cut_size - overlap))
        x_starts = list(range(0, w, cut_size - overlap))
        if y_starts and y_starts[-1] + cut_size < h:
            y_starts.append(h - cut_size)
        if x_starts and x_starts[-1] + cut_size < w:
            x_starts.append(w - cut_size)
        for y0 in y_starts:
            for x0 in x_starts:
                y1 = min(y0 + cut_size, h)
                x1 = min(x0 + cut_size, w)
                crop_img = img[y0:y1, x0:x1]
                crops.append(((y0, y1, x0, x1), crop_img))
        return crops

    @staticmethod
    def parse_ocr_result(result):
        out = []
        for res in result:
            texts = res['rec_texts']
            scores = res['rec_scores']
            for text, score in zip(texts, scores):
                if score > OCR_SCORE_THRESH:
                    out.append({'text': text, 'score': float(score)})
        return out


    def ocr_classify(self, image_path, cut_size=CUT_SIZE, overlap_ratio=OVERLAP_RATIO, force_crop=False):
        """OCR识别主函数"""
        img = None
        crops = None
        crop_imgs = None
        
        try:
            # 判断image_path是url还是本地路径
            if isinstance(image_path, str) and (image_path.startswith('http://') or image_path.startswith('https://')):
                img = url_to_cv2_image(image_path)
                LOGGER.info(f"Downloaded image from URL: {image_path}")
            elif isinstance(image_path, str):
                img = cv2.imread(image_path)
                if img is None:
                    raise ValueError(f"无法读取图片: {image_path}")
                LOGGER.info(f"Loaded image from local: {image_path}")
            else:
                raise ValueError("image_path必须是字符串类型")
                
            h, w = img.shape[:2]
            LOGGER.info(f"Image size: {w}x{h}")
            ocr_results = []
            
            if force_crop and (h >= cut_size or w >= cut_size):
                LOGGER.info(f"图片过大，进行切分处理: {cut_size}x{cut_size}")
                crops = self.crop_image_with_overlap(img, cut_size, overlap_ratio)
                
                # 分批处理，避免一次性加载太多图像到显存
                batch_size = 4  # 根据显存大小调整
                for i in range(0, len(crops), batch_size):
                    batch_crops = crops[i:i+batch_size]
                    crop_imgs = [crop_img for bbox, crop_img in batch_crops]
                    
                    try:
                        results = self.ocr.predict(input=crop_imgs)
                        for res in results:
                            ocr_results.extend(self.parse_ocr_result([res]))
                    finally:
                        # 清理批次图像数据
                        if crop_imgs:
                            del crop_imgs
                            crop_imgs = None

            else:
                LOGGER.info(f"直接处理整张图片")
                
                # 按比例缩放到cut_size尺寸
                h, w = img.shape[:2]
                if h > cut_size or w > cut_size:
                    # 计算缩放比例，保持宽高比
                    scale = min(cut_size / w, cut_size / h)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    
                    LOGGER.info(f"图片尺寸 {w}x{h} 缩放到 {new_w}x{new_h} (缩放比例: {scale:.3f})")
                    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    
                    # 释放原始图像
                    del img
                    img = img_resized
                
                result = self.ocr.predict(input=img)
                ocr_results.extend(self.parse_ocr_result(result))

            LOGGER.info(f"识别到 {len(ocr_results)} 个文本")
            return ocr_results
            
        finally:
            # 确保所有图像数据被释放，但保持模型常驻
            if img is not None:
                del img
                img = None
            if crops is not None:
                del crops
                crops = None
            if crop_imgs is not None:
                del crop_imgs
                crop_imgs = None
            


    # 同步方法
    def test(self):
        # image_path = "/mnt/nfs_105/txy/images/图片-SM- (11).bmp"
        image_path = "/mnt/nfs_105/txy/images/图片-FM- (2).bmp"
        # image_path = "https://smartfly.lilingtech.cn/media/test/dalian/DCIM/DJI_202504221630_010_%E4%B8%AD%E4%BA%A4%E4%B8%80%E8%88%AA%E5%B1%80%E4%B8%89%E5%85%AC%E5%8F%B8%E5%A4%A7%E8%BF%9E%E9%87%91%E5%B7%9E%E6%B9%BE%E5%9B%BD%E9%99%85%E6%9C%BA%E5%9C%BA-%E6%96%B0%E5%BB%BA%E9%9D%A2%E7%8A%B6%E8%88%AA%E7%BA%BF3/DJI_20250422164746_0006_V.JPG"
        results = self.ocr_classify(image_path)
        for item in results:
            print(item)

    def start(self):
        """启动OCR服务"""
        uvicorn.run(app=self.app, host=SERVER_HOST, port=SERVER_PORT)

    def register_routes(self):
        """注册API路由"""
        @self.app.post("/api/ocr")
        async def ocr_task(request: OCRRequest):
            """OCR识别接口"""
            try:
                
                if(IS_HARD):  #采用硬编码
                    filename = os.path.basename(request.image_path)
                    request.image_path = os.path.join(HARD_PATH, filename) 
                LOGGER.info(f'收到OCR请求: {request.image_path}')
                
                results = self.ocr_classify(
                    image_path=request.image_path,
                    cut_size=request.cut_size,
                    overlap_ratio=request.overlap_ratio,
                    force_crop=request.force_crop
                )
                LOGGER.info(f'识别完成，共识别到 {len(results)} 个文本')
                return {
                    "code": 200,
                    "message": "成功",
                    "body": {
                        "image_path": request.image_path,
                        "text_count": len(results),
                        "results": results
                    }
                }
            except Exception as e:
                LOGGER.error(f'OCR识别失败: {e}')
                return {
                    "code": 4004,
                    "message": f"OCR识别失败: {str(e)}",
                    "body": None
                }

        @self.app.get("/api/health")
        async def health_check():
            """健康检查接口"""
            return {"code": 200, "message": "服务正常", "body": None}




if __name__ == '__main__':
    ocr_service = OcrService()
    ocr_service.register_routes()
    
    # 测试模式
    # ocr_service.test()
    
    # 启动服务
    ocr_service.start()
    # nohup python /home/work/liwenkai/c_learn/ocr_main.py > logger_ocr.txt 2>&1&disown
    # exit