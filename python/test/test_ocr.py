from paddleocr import PaddleOCR
import cv2
threshold = 0.3
cut_size = 1920
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=True,
    text_det_unclip_ratio = 1.5,
    text_det_box_thresh = 0.6,
    text_det_thresh = 0.3,
    text_rec_score_thresh = 0,

)


def ocr_classify(image):
    """
    入参: image (str) 图片路径
    返回: [{'text':..., 'type':...}, ...]  type=0车牌, type=1横幅
    """
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    # if  (h >= cut_size or w >= cut_size):
    #     # 按比例缩放图像
    #     max_dim = max(h, w)
    #     if max_dim > cut_size:
    #         ratio = cut_size / max_dim
    #         new_w = int(w * ratio)
    #         new_h = int(h * ratio)
    #         resize_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    #         print(f"缩放后图像尺寸: {resize_img.shape}")
    #     else:
    #         resize_img = img.copy()
    #         print(f"不缩放图像尺寸: {resize_img.shape}")
    
    max_dim = max(h, w)    
    ratio = cut_size / max_dim 
    new_w = int(w * ratio)
    new_h = int(h * ratio)
    resize_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    
    
    result = ocr.predict(input=resize_img, text_det_limit_side_len=736,text_det_limit_type='max')
    out = []
    for res in result:
        texts = res['rec_texts']
        scores = res['rec_scores']
        for text, score in zip(texts, scores):
            if score > threshold:
                out.append({'text': text})
    return result[0],out

# 示例调用
if __name__ == "__main__":
    # image_path = "/home/work/liwenkai/yxgame/txy/图片-SM- (4).jpg"
    
    # image_path = "/home/work/liwenkai/c_learn/det_resize_0.jpg"
    image_path = "/mnt/nfs_105/txy/images/图片-SM- (122).jpg"
    # image_path = "/mnt/nfs_105/txy/images/图片-SM- (7).jpg"
    # image_path ="/home/work/huangzhi/paddlex_workspace/3.jpg"

    import time
    startTime= time.time()
    img,results = ocr_classify(image_path)
    endTime = time.time()
    cost = endTime - startTime
    print(cost)
    img.save_to_img('/home/work/liwenkai/c_learn/ocr_result11.jpg')
    # for item in results:
    #     print(item)
    