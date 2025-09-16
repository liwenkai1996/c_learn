import cv2
import numpy as np
import os

def order_points(pts):
    """
    对四个点进行排序，使其按照左上、右上、右下、左下的顺序排列
    """
    rect = np.zeros((4, 2), dtype="float32")
    
    # 计算每个点的和与差
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # 左上角点（和最小）
    rect[2] = pts[np.argmax(s)]  # 右下角点（和最大）
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # 右上角点（差最小）
    rect[3] = pts[np.argmax(diff)]  # 左下角点（差最大）
    
    return rect

def four_point_transform(image, pts):
    """
    对图像进行四点透视变换
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # 计算新图像的宽度
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    # 计算新图像的高度
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # 构建目标点
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    
    # 计算透视变换矩阵并应用
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return warped

def detect_document_edge_enhanced(image, debug=False):
    """
    使用边缘增强方法检测文档边界（基于用户反馈的最佳效果）
    """
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 边缘增强
    kernel_sharpen = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(gray, -1, kernel_sharpen)
    
    # 双边滤波保持边缘
    bilateral = cv2.bilateralFilter(sharpened, 9, 75, 75)
    
    # 边缘检测 - 使用用户反馈的最佳参数
    edged = cv2.Canny(bilateral, 75, 200)
    
    if debug:
        # 保存中间处理结果
        cv2.imwrite("debug_edge_enhanced_gray.jpg", gray)
        cv2.imwrite("debug_edge_enhanced_sharpened.jpg", sharpened)
        cv2.imwrite("debug_edge_enhanced_bilateral.jpg", bilateral)
        cv2.imwrite("debug_edge_enhanced_final.jpg", edged)
        print("保存边缘增强处理过程图像")
    
    # 查找轮廓
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # 按面积排序
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # 检查前10个最大的轮廓
    for contour in contours[:10]:
        # 计算轮廓周长
        peri = cv2.arcLength(contour, True)
        # 近似轮廓 - 稍微放宽条件
        approx = cv2.approxPolyDP(contour, 0.03 * peri, True)
        
        # 如果轮廓有四个点
        if len(approx) == 4:
            # 计算轮廓的矩形度
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            rect_area = w * h
            rect_score = area / rect_area if rect_area > 0 else 0
            
            # 计算长宽比
            aspect_ratio = w / h if h > 0 else 0
            ideal_ratio = 1.4  # A4纸比例
            ratio_score = 1.0 / (1.0 + abs(aspect_ratio - ideal_ratio))
            
            # 综合评分
            total_score = rect_score * 0.6 + ratio_score * 0.4
            
            # 降低阈值，更容易找到文档
            if total_score > 0.3:  # 原来可能是0.5或更高
                if debug:
                    print(f"边缘增强方法找到轮廓，评分: {total_score:.3f}")
                return approx.reshape(4, 2)
    
    return None

def detect_document_improved(image, debug=False, method="auto"):
    """
    改进的文档检测函数，支持指定检测方法
    
    参数:
        image: 输入图像
        debug: 是否显示调试信息
        method: 检测方法 ("auto", "edge_enhanced", "multi_method")
    """
    
    if method == "edge_enhanced":
        # 直接使用边缘增强方法
        return detect_document_edge_enhanced(image, debug)
    
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 多种预处理方法
    methods = []
    
    # 方法1: 原始方法
    blurred1 = cv2.GaussianBlur(gray, (5, 5), 0)
    edged1 = cv2.Canny(blurred1, 75, 200)
    methods.append(("原始方法", edged1))
    
    # 方法2: 自适应阈值
    blurred2 = cv2.GaussianBlur(gray, (7, 7), 0)
    edged2 = cv2.Canny(blurred2, 50, 150)
    methods.append(("自适应阈值", edged2))
    
    # 方法3: 形态学操作
    kernel = np.ones((3, 3), np.uint8)
    edged3 = cv2.morphologyEx(edged1, cv2.MORPH_CLOSE, kernel)
    methods.append(("形态学操作", edged3))
    
    # 方法4: 边缘增强（用户反馈的最佳方法）
    kernel_sharpen = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(gray, -1, kernel_sharpen)
    blurred4 = cv2.GaussianBlur(sharpened, (5, 5), 0)
    edged4 = cv2.Canny(blurred4, 75, 200)
    methods.append(("边缘增强", edged4))
    
    if debug:
        # 保存所有方法的边缘检测结果
        for i, (name, edged) in enumerate(methods):
            debug_filename = f"debug_{name}_{i}.jpg"
            cv2.imwrite(debug_filename, edged)
            print(f"保存调试图像: {debug_filename}")
    
    # 优先尝试边缘增强方法
    corners = detect_document_edge_enhanced(image, debug=False)
    if corners is not None:
        if debug:
            print("边缘增强方法成功检测到文档")
        return corners
    
    # 如果边缘增强失败，尝试其他方法
    best_contour = None
    best_score = 0
    
    for method_name, edged in methods:
        # 查找轮廓
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            continue
            
        # 按面积排序
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # 只检查前5个最大的轮廓
        for contour in contours[:5]:
            # 计算轮廓周长
            peri = cv2.arcLength(contour, True)
            # 近似轮廓
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            # 如果轮廓有四个点
            if len(approx) == 4:
                # 计算轮廓的矩形度（矩形度越高越好）
                area = cv2.contourArea(contour)
                x, y, w, h = cv2.boundingRect(contour)
                rect_area = w * h
                rect_score = area / rect_area if rect_area > 0 else 0
                
                # 计算长宽比（接近1:1.4的A4比例更好）
                aspect_ratio = w / h if h > 0 else 0
                ideal_ratio = 1.4  # A4纸比例
                ratio_score = 1.0 / (1.0 + abs(aspect_ratio - ideal_ratio))
                
                # 综合评分
                total_score = rect_score * 0.7 + ratio_score * 0.3
                
                if total_score > best_score:
                    best_score = total_score
                    best_contour = approx
                    if debug:
                        print(f"{method_name} 找到更好的轮廓，评分: {total_score:.3f}")
    
    if best_contour is not None:
        return best_contour.reshape(4, 2)
    
    return None

def auto_correct_document(image_path, output_path=None, debug=False, save_debug_images=False, method="auto"):
    """
    自动校正文档图片（改进版）
    
    参数:
        image_path (str): 输入图像路径
        output_path (str, optional): 输出图像路径，如果为None则自动生成
        debug (bool): 是否显示调试信息
        save_debug_images (bool): 是否保存调试图像
        method (str): 检测方法 ("auto", "edge_enhanced", "multi_method")
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return None
    
    print(f"图像尺寸: {image.shape}")
    print(f"使用检测方法: {method}")
    
    # 检测文档角点
    corners = detect_document_improved(image, debug=debug, method=method)
    
    if corners is None:
        print("未检测到文档边界，请确保图像中包含清晰的文档")
        print("建议:")
        print("1. 确保文档与背景有足够的对比度")
        print("2. 确保文档边缘清晰可见")
        print("3. 避免强反光和阴影")
        print("4. 尝试调整拍摄角度")
        return None
    
    print(f"检测到文档角点: {corners}")
    
    # 在图像上绘制检测到的角点
    debug_image = image.copy()
    for i, (x, y) in enumerate(corners):
        cv2.circle(debug_image, (int(x), int(y)), 10, (0, 255, 0), -1)
        cv2.putText(debug_image, str(i), (int(x)+15, int(y)-15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # 绘制轮廓线
    corners_int = corners.astype(int)
    cv2.polylines(debug_image, [corners_int], True, (255, 0, 0), 3)
    
    if save_debug_images:
        debug_path = image_path.replace('.', '_debug.')
        cv2.imwrite(debug_path, debug_image)
        print(f"调试图像已保存到: {debug_path}")
    
    if debug:
        # 保存检测结果图像
        debug_result_path = image_path.replace('.', '_detection_result.')
        cv2.imwrite(debug_result_path, debug_image)
        print(f"保存检测结果图像: {debug_result_path}")
    
    # 进行透视变换
    corrected = four_point_transform(image, corners)
    
    # 保存结果
    if output_path is None:
        base_name = os.path.splitext(image_path)[0]
        output_path = f"{base_name}_corrected.jpg"
    
    cv2.imwrite(output_path, corrected)
    print(f"校正后的图像已保存到: {output_path}")
    print(f"校正后图像尺寸: {corrected.shape}")
    
    return corrected

def manual_correct_with_coordinates(image_path, corners, output_path=None):
    """
    使用指定的坐标进行文档校正（替代交互式校正）
    
    参数:
        image_path (str): 输入图像路径
        corners (list): 四个角点坐标，格式为 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        output_path (str, optional): 输出图像路径，如果为None则自动生成
    
    返回:
        numpy.ndarray: 校正后的图像数组，如果失败返回None
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return None
    
    # 在图像上绘制指定的角点
    debug_image = image.copy()
    for i, (x, y) in enumerate(corners):
        cv2.circle(debug_image, (int(x), int(y)), 10, (0, 255, 0), -1)
        cv2.putText(debug_image, str(i), (int(x)+15, int(y)-15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # 绘制轮廓线
    corners_int = np.array(corners, dtype=int)
    cv2.polylines(debug_image, [corners_int], True, (255, 0, 0), 3)
    
    # 保存调试图像
    debug_path = image_path.replace('.', '_manual_corners.')
    cv2.imwrite(debug_path, debug_image)
    print(f"保存手动角点图像: {debug_path}")
    
    # 转换为numpy数组
    corners = np.array(corners, dtype="float32")
    
    # 进行透视变换
    corrected = four_point_transform(image, corners)
    
    # 保存结果
    if output_path is None:
        base_name = os.path.splitext(image_path)[0]
        output_path = f"{base_name}_manual_corrected.jpg"
    
    cv2.imwrite(output_path, corrected)
    print(f"手动校正完成，结果保存到: {output_path}")
    
    return corrected

def test_document_detection(image_path):
    """
    测试文档检测功能
    """
    print(f"测试文档检测: {image_path}")
    auto_correct_document(image_path, debug=True, save_debug_images=True)

def test_edge_enhanced_method(image_path):
    """
    专门测试边缘增强检测方法
    """
    print(f"测试边缘增强检测方法: {image_path}")
    auto_correct_document(image_path, debug=True, save_debug_images=True, method="edge_enhanced")

def analyze_image_for_corners(image_path):
    """
    分析图像，帮助用户确定角点坐标
    保存多个处理步骤的图像，方便查看
    """
    print(f"分析图像: {image_path}")
    
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return None
    
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 保存灰度图
    gray_path = image_path.replace('.', '_gray.')
    cv2.imwrite(gray_path, gray)
    print(f"保存灰度图: {gray_path}")
    
    # 多种边缘检测方法
    methods = [
        ("gaussian_5x5_canny_75_200", cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 75, 200)),
        ("gaussian_7x7_canny_50_150", cv2.Canny(cv2.GaussianBlur(gray, (7, 7), 0), 50, 150)),
        ("gaussian_3x3_canny_100_200", cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 100, 200)),
        ("bilateral_canny_75_200", cv2.Canny(cv2.bilateralFilter(gray, 9, 75, 75), 75, 200)),
    ]
    
    # 保存各种边缘检测结果
    for name, edged in methods:
        edged_path = image_path.replace('.', f'_{name}.')
        cv2.imwrite(edged_path, edged)
        print(f"保存边缘检测结果: {edged_path}")
    
    # 尝试检测轮廓
    for name, edged in methods:
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 按面积排序
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            # 保存前5个最大轮廓
            contour_image = image.copy()
            for i, contour in enumerate(contours[:5]):
                # 绘制轮廓
                cv2.drawContours(contour_image, [contour], -1, (0, 255, 0), 2)
                
                # 尝试近似为四边形
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                
                if len(approx) == 4:
                    # 绘制四边形
                    cv2.polylines(contour_image, [approx], True, (255, 0, 0), 3)
                    
                    # 绘制角点
                    for j, point in enumerate(approx):
                        x, y = point[0]
                        cv2.circle(contour_image, (x, y), 8, (0, 0, 255), -1)
                        cv2.putText(contour_image, str(j), (x+10, y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # 保存轮廓分析结果
            contour_path = image_path.replace('.', f'_contours_{name}.')
            cv2.imwrite(contour_path, contour_image)
            print(f"保存轮廓分析结果: {contour_path}")
    
    print("图像分析完成！请查看保存的图片文件来了解图像处理过程。")
    print("如果自动检测失败，您可以使用 manual_correct_with_coordinates() 函数手动指定角点。")

if __name__ == "__main__":
    # 示例用法
    print("改进的文档校正工具")
    print("=" * 50)
    print("1. 自动校正: auto_correct_document('image.jpg')")
    print("2. 边缘增强校正: auto_correct_document('image.jpg', method='edge_enhanced')")
    print("3. 手动校正: manual_correct_with_coordinates('image.jpg', [[x1,y1],[x2,y2],[x3,y3],[x4,y4]])")
    print("4. 测试检测: test_document_detection('image.jpg')")
    print("5. 测试边缘增强: test_edge_enhanced_method('image.jpg')")
    print("6. 图像分析: analyze_image_for_corners('image.jpg')")
    print("=" * 50)
