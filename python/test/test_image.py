import cv2
import numpy as np
import os
import glob
'''文档方向校正'''

def rotate_image_keep_size(image, angle):
    """
    旋转图像但保持原始尺寸不变
    """
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    
    # 获取旋转矩阵
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # 应用旋转变换
    rotated = cv2.warpAffine(image, rotation_matrix, (width, height), 
                            flags=cv2.INTER_CUBIC, 
                            borderMode=cv2.BORDER_CONSTANT, 
                            borderValue=(255, 255, 255))  # 白色背景填充
    
    return rotated


def simple_angle_detection(image_path, debug_folder="debug_output"):
    """
    简单的角度检测方法（基于霍夫变换）
    """
    os.makedirs(debug_folder, exist_ok=True)
    
    print(f"\n🔍 开始角度检测: {image_path}")
    
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ 无法读取图像: {image_path}")
        return 0
    
    print(f"✅ 图像读取成功，尺寸: {img.shape}")
    
    # 保存原始图像
    cv2.imwrite(os.path.join(debug_folder, "01_原始图像.jpg"), img)
    print(f"💾 保存: 01_原始图像.jpg")
    
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(debug_folder, "02_灰度图.jpg"), gray)
    print(f"💾 保存: 02_灰度图.jpg")
    
    # 高斯模糊
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    cv2.imwrite(os.path.join(debug_folder, "03_高斯模糊.jpg"), blurred)
    print(f"💾 保存: 03_高斯模糊.jpg")
    
    # Canny边缘检测
    edges = cv2.Canny(blurred, 30, 100, apertureSize=3)
    cv2.imwrite(os.path.join(debug_folder, "04_Canny边缘.jpg"), edges)
    print(f"💾 保存: 04_Canny边缘.jpg")
    
    # 形态学闭运算
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(os.path.join(debug_folder, "05_形态学闭运算.jpg"), edges)
    print(f"💾 保存: 05_形态学闭运算.jpg")
    
    # 使用霍夫变换检测直线
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
    
    if lines is None:
        print(f"⚠️ 霍夫变换未检测到直线")
        return 0
    
    print(f"✅ 霍夫变换检测到 {len(lines)} 条直线")
    
    # 在边缘图上绘制检测到的直线
    lines_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    angles = []
    for i, line in enumerate(lines):
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        angles.append(angle)
        
        # 绘制直线
        cv2.line(lines_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # 标注角度
        cv2.putText(lines_img, f"{angle:.1f}°", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    cv2.imwrite(os.path.join(debug_folder, "06_霍夫直线检测.jpg"), lines_img)
    print(f"💾 保存: 06_霍夫直线检测.jpg")
    
    if len(angles) == 0:
        print(f"⚠️ 没有有效的角度数据")
        return 0
    
    # 角度统计信息
    angles_array = np.array(angles)
    
    print(f"📊 角度统计:")
    print(f"   - 所有角度:{angles_array}")
    print(f"   - 最小角度: {angles_array.min():.2f}°")
    print(f"   - 最大角度: {angles_array.max():.2f}°")
    print(f"   - 平均角度: {angles_array.mean():.2f}°")
    print(f"   - 中位数角度: {np.median(angles_array):.2f}°")
    print(f"   - 标准差: {angles_array.std():.2f}°")
    
    # 使用中位数角度
    median_angle = np.median(angles)
    print(f"🎯 最终选择角度: {median_angle:.2f} 度")
    
    return median_angle

# ===== 新增：无 OCR 的方向判定，仅决定用 angle 还是 180+angle =====

def _bin_otsu_invert(gray):
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return 255 - th  # 让墨迹为白色


def _orientation_score(gray):
    h, w = gray.shape
    bw = _bin_otsu_invert(gray)
    score = 0.0
    
    # 规则1：顶部/底部密度对比（权重降低）
    band = max(1, h // 10)
    top_density = bw[:band, :].mean() / 255.0
    bot_density = bw[h - band:, :].mean() / 255.0
    score += 0.3 * (top_density - bot_density)
    
    # 规则2：内容重心位置（上半部分内容多 = 正向）
    row_density = bw.mean(axis=1) / 255.0
    total_content = row_density.sum()
    if total_content > 0:
        # 计算内容重心
        weighted_y = sum(i * density for i, density in enumerate(row_density))
        center_of_mass = weighted_y / total_content
        # 重心越靠上，分数越高
        score += 0.5 * (0.5 - center_of_mass / h)
    
    # 规则3：上下半部分内容比例
    mid = h // 2
    top_content = row_density[:mid].sum()
    bot_content = row_density[mid:].sum()
    total = top_content + bot_content
    if total > 0:
        # 上半部分内容占比越高，分数越高
        score += 0.4 * (top_content - bot_content) / total
    
    # 规则4：长水平线位置（标题、段落分隔线等）
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=120, minLineLength=int(0.3 * w), maxLineGap=15)
    if lines is not None:
        ys = []
        tan8 = np.tan(np.radians(8))
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            dx, dy = x2 - x1, y2 - y1
            if dx == 0:
                continue
            slope = abs(dy / dx)
            length = np.hypot(dx, dy)
            if slope < tan8 and length >= 0.3 * w:
                ys.append((y1 + y2) * 0.5)
        if ys:
            median_y = float(np.median(ys))
            # 长线越靠上，分数越高
            score += 0.2 * (0.5 - median_y / h)
    
    # 规则5：内容开始位置（从上往下扫描第一个有内容的区域）
    content_threshold = max(0.05, np.percentile(row_density, 70))
    first_content_row = next((i for i, v in enumerate(row_density) if v > content_threshold), h // 2)
    # 内容开始位置越靠上，分数越高
    score += 0.3 * (0.5 - first_content_row / h)
    
    return float(score)


def decide_final_angle_no_ocr(image, angle):
    """
    仅决定最终旋转角：返回 angle 或 180+angle。
    依据无 OCR 的方向分数，选择更像“正向”的那个。
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    s0 = _orientation_score(gray)
    s180 = _orientation_score(cv2.rotate(gray, cv2.ROTATE_180))
    return angle if s0 <= s180 else (angle + 180)


def correct_document(image_path, angle_threshold, output_path="corrected_doc.jpg"):
    """
    文档校正方法（仅使用霍夫变换检测角度 + 无OCR方向判定）
    """
    # 检测文档角度
    angle = simple_angle_detection(image_path)
    
    # 读取原始图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return False
    
    height, width = img.shape[:2]
    print(f"原始图像尺寸: {width} x {height}")
    
    # 计算最终旋转角：angle 或 180+angle
    final_angle = decide_final_angle_no_ocr(img, angle)
    
    # 只有霍夫变换角度足够大时才进行校正
    if abs(angle) % 180 > angle_threshold: 
        corrected = rotate_image_keep_size(img, final_angle)
        cv2.imwrite(output_path, corrected)
        print(f"最终旋转角 {final_angle:.2f}°（源角 {angle:.2f}°），保存为 {output_path}")
    else:
        cv2.imwrite(output_path, img)
        print(f"旋转角很小（{final_angle:.2f}°），保存原始图像为 {output_path}")
    
    return True


def batch_correct_documents(input_folder, output_folder,angle_threshold):
    """
    批量处理文件夹中的图片文件
    """
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 支持的图片格式
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG','*.bmp','*.tif']
    
    # 获取所有图片文件
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(input_folder, ext)))
    
    if not image_files:
        print(f"在 {input_folder} 中没有找到图片文件")
        return
    
    print(f"找到 {len(image_files)} 个图片文件，开始批量处理...")
    
    success_count = 0
    for i, image_path in enumerate(image_files, 1):
        try:

            # 获取文件名（不含路径）
            filename = os.path.basename(image_path)
            name, ext = os.path.splitext(filename)
            # if(filename != "图片-SM- (32).png"):
            #     continue
            # 构建输出路径
            output_path = os.path.join(output_folder, f"{name}_corrected{ext}")
            
            print(f"\n[{i}/{len(image_files)}] 处理: {filename}")
            
            # 检测文档角度
            angle = simple_angle_detection(image_path)
            
            # 读取原始图像
            img = cv2.imread(image_path)
            if img is None:
                print(f"  ❌ 无法读取图像: {image_path}")
                continue
            
            height, width = img.shape[:2]
            print(f"  📐 图像尺寸: {width} x {height}")
            
            # 计算最终旋转角：angle 或 180+angle
            final_angle = decide_final_angle_no_ocr(img, angle)
            
            # 只有角度足够大时才进行校正
            if abs(final_angle) % 180 > angle_threshold:
                corrected = rotate_image_keep_size(img, final_angle)
                cv2.imwrite(output_path, corrected)
                print(f"  ✅ 旋转 {final_angle:.2f}°，保存为: {output_path}")
            else:
                cv2.imwrite(output_path, img)
                print(f"  ✅ 角度很小（{final_angle:.2f}°），保存原图为: {output_path}")
            
            success_count += 1
            
        except Exception as e:
            print(f"  ❌ 处理失败: {str(e)}")
            continue
    
    print(f"\n🎉 批量处理完成！成功处理 {success_count}/{len(image_files)} 个文件")


def resize_image_fixed_long_side(image_path, target_long_side=1024):
    """
    调整图片尺寸：长边缩放到固定尺寸，短边按比例缩小
    
    Args:
        image_path: 图片路径
        target_long_side: 目标长边尺寸，默认1024
    
    Returns:
        调整后的图片路径
    """
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return image_path
    
    height, width = img.shape[:2]
    print(f"原始图像尺寸: {width} x {height}")
    
    # 确定长边和短边
    if width >= height:
        long_side = width
        short_side = height
        is_landscape = True
    else:
        long_side = height
        short_side = width
        is_landscape = False
    
    # 计算缩放比例
    scale = target_long_side / long_side
    new_short_side = int(short_side * scale)
    
    # 调整尺寸
    if is_landscape:
        new_width = target_long_side
        new_height = new_short_side
    else:
        new_width = new_short_side
        new_height = target_long_side
    
    # 使用INTER_AREA插值方法，适合图像缩小
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # 生成新的文件名
    base_name = os.path.splitext(image_path)[0]
    extension = os.path.splitext(image_path)[1]
    resized_path = f"{base_name}_resized{extension}"
    
    # 保存调整后的图片
    cv2.imwrite(resized_path, resized_img)
    print(f"调整后图像尺寸: {new_width} x {new_height}")
    print(f"调整后图像保存为: {resized_path}")
    
    return resized_path


if __name__ == "__main__":
    angle_threshold = 7.0  #角度阈值，单位度
    # 单个文件测试
    # image_path = "/home/work/liwenkai/yxgame/txy/图片-SM- (5).jpg"
    image_path = "/home/work/liwenkai/yxgame/txy/图片-SM- (32).png"
    # image_path = "/home/work/liwenkai/yxgame/txy/图片-SM- (33).png"
    # image_path = "/home/work/liwenkai/yxgame/txy/图片-SM- (39).png"
    # image_path = "/home/work/liwenkai/yxgame/txy/图片-SM- (41).png"
    # image_path = "/home/work/liwenkai/yxgame/txy/图片-SM- (71).png"   #有问题
    # image_path = "/home/work/liwenkai/yxgame/txy/图片-SM- (75).jpg"
    # image_path = "/home/work/liwenkai/yxgame/txy/图片-SM- (76).jpg"
    # image_path = "/home/work/liwenkai/yxgame/txy/图片-SM- (115).jpg"   #有问题
    
    # 先调整图片尺寸，长边缩放到1024像素
    # resized_image_path = resize_image_fixed_long_side(image_path, target_long_side=2000)
    
    # # 使用调整后的图片进行文档校正
    # correct_document(resized_image_path, angle_threshold, "corrected_final.jpg")
    
    # 批量处理
    input_folder = "/home/work/liwenkai/yxgame/secret_text"  # 输入文件夹路径
    output_folder = "/home/work/liwenkai/c_learn/corrected_images2"  # 输出文件夹路径
    batch_correct_documents(input_folder, output_folder,angle_threshold)
    
    
    # 图片-SM- (75).jpg
    # 图片-SM- (115).jpg
    # 图片-SM- (76).jpg
    # 图片-SM- (71).png
    # 图片-SM- (41).png
    # 图片-SM- (39).png
    # 图片-SM- (33).png
    # 图片-SM- (32).png