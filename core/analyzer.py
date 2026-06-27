"""
图像分析模块
SEM颗粒识别的核心算法实现

功能:
- 模型安全加载（阅后即焚）
- YOLOv8图像分割
- OCR自动比例尺提取
- 边缘过滤 + 完整度筛选
- 粒径统计分析
"""

import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"D:\\Tesseract-OCR\\tesseract.exe"

import cv2
import numpy as np
import tempfile
import os
import math
import random
from typing import List, Optional, Tuple

# YOLOv8 模型支持
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    YOLO = None


# =========================================================
# 全局变量 - 模型单例
# =========================================================
_model_instance: Optional[YOLO] = None
_encrypted_model_path: Optional[str] = None


# =========================================================
# A. 模型安全加载（阅后即焚）
# =========================================================

def load_model_securely(encrypted_path: str) -> YOLO:
    """
    安全加载加密的YOLOv8模型（阅后即焚机制）

    工作流程:
    1. 解密 .encrypted 模型文件得到 bytes
    2. 写入 tempfile.NamedTemporaryFile
    3. 用 YOLO(temp_file.name) 加载
    4. 【关键】加载完立即 os.remove 临时文件

    Args:
        encrypted_path: 加密模型文件路径 (.encrypted)

    Returns:
        加载好的 YOLO 模型对象

    Raises:
        FileNotFoundError: 加密模型文件不存在
        ImportError: ultralytics 未安装
    """
    if not ULTRALYTICS_AVAILABLE:
        raise ImportError("需要安装 ultralytics: pip install ultralytics")

    if not os.path.exists(encrypted_path):
        raise FileNotFoundError(f"加密模型不存在: {encrypted_path}")

    from core.security import decrypt_data

    print(f"[安全加载] 读取加密模型: {encrypted_path}")

    # Step 1: 读取加密的模型字节
    with open(encrypted_path, "r", encoding="utf-8") as f:
        encrypted_model_str = f.read()

    # Step 2: 解密得到原始模型字节
    model_bytes = decrypt_data(encrypted_model_str)
    print(f"[安全加载] 解密完成，模型大小: {len(model_bytes) / (1024*1024):.2f} MB")

    # Step 3: 创建临时文件
    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pt",
        prefix="yolo_"
    )
    temp_file_path = temp_file.name

    try:
        # 写入临时文件
        temp_file.write(model_bytes)
        temp_file.close()
        print(f"[安全加载] 写入临时文件: {temp_file_path}")

        # Step 4: 加载模型
        print("[安全加载] YOLO加载中...")
        model = YOLO(temp_file_path)

        # ================================================
        # 【关键安全步骤】阅后即焚：删除临时文件
        # ================================================
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            print(f"[安全加载] ✓ 临时文件已销毁: {temp_file_path}")

        print("[安全加载] ✓ 模型加载成功（阅后即焚机制已激活）")
        return model

    except Exception as e:
        # 异常时也要清理临时文件
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise e


def get_model() -> Optional[YOLO]:
    """获取已加载的模型实例"""
    return _model_instance


def initialize_model(encrypted_path: str) -> YOLO:
    """
    初始化模型单例

    Args:
        encrypted_path: 加密模型文件路径

    Returns:
        YOLO 模型对象
    """
    global _model_instance, _encrypted_model_path

    if _model_instance is None:
        _model_instance = load_model_securely(encrypted_path)
        _encrypted_model_path = encrypted_path

    return _model_instance


# =========================================================
# B. 分析算法
# =========================================================

def analyze_particles(
    image_bytes: bytes,
    scale_ratio: float = None,
    use_mock: bool = True
) -> dict:
    """
    分析SEM图像中的颗粒

    流程:
    1. cv2.imdecode 读取图片
    2. 自动提取比例尺（如果未提供）
    3. 调用 YOLO 获取 masks
    4. 循环处理每个 mask:
       - cv2.contourArea 计算实际面积
       - cv2.minEnclosingCircle 获取外接圆及直径
       - 判定完整度: actual_area / (PI * r^2) > 0.8
       - 判定位置: 任何像素点不得触碰图像四壁
    5. 统计数据: 计算平均直径(nm)、D50(nm)、总数

    Args:
        image_bytes: 图像的原始字节数据
        scale_ratio: 像素到纳米的转换比例 (1像素 = X nm)，可选
                      如果为 None 或 <= 0，则自动从图片OCR提取
        use_mock: 是否使用模拟数据（无模型时）

    Returns:
        包含分析结果的字典:
        {
            "total_count": int,
            "average_nm": float,
            "d50_nm": float,
            "std_dev": float,
            "min_nm": float,
            "max_nm": float,
            "scale_ratio": float,      # 实际使用的比例尺
            "scale_source": str,       # "user_provided" 或 "auto_extracted"
            "particle_data": [...]
        }
    """
    # Step 1: 解码图像
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("无法解码图像数据")

    height, width = image.shape[:2]
    print(f"[分析] 图像尺寸: {width}x{height}")

    # Step 2: 自动提取比例尺（如果未提供）
    if scale_ratio is None or scale_ratio <= 0:
        print("[分析] 未提供比例尺，尝试自动提取...")
        try:
            from core.scale_reader import extract_scale_ratio, ScaleExtractionError

            extracted_ratio = extract_scale_ratio(image)
            scale_ratio = extracted_ratio
            scale_source = "auto_extracted"
            print(f"[分析] ✓ OCR自动提取比例尺: 1像素 = {scale_ratio:.4f} nm")

        except ScaleExtractionError as e:
            print(f"[分析] ✗ 自动提取失败: {e}")
            raise ValueError(
                f"无法自动提取比例尺: {e}\n"
                "请手动提供 scale_ratio 参数"
            )
    else:
        scale_source = "user_provided"
        print(f"[分析] 使用用户提供的比例尺: 1像素 = {scale_ratio:.4f} nm")

    # Step 3: 获取轮廓
    if use_mock or _model_instance is None:
        print("[分析] 使用模拟数据模式")
        contours = _generate_mock_contours(width, height, num=30)
    else:
        print("[分析] 使用YOLOv8模型分割")
        contours = _extract_contours_from_yolo(_model_instance, image)

    print(f"[分析] 检测到轮廓数量: {len(contours)}")

    # Step 3: 过滤和处理每个轮廓
    valid_particles = []
    border_tolerance = 5
    min_completeness = 0.8

    for i, contour in enumerate(contours):
        # 3.1 边缘检测：任何像素点触碰图像四壁则丢弃
        if _is_on_image_border(contour, width, height, border_tolerance):
            continue

        # 3.2 计算实际面积
        actual_area = cv2.contourArea(contour)

        # 3.3 计算质心
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
        else:
            cx, cy = 0, 0

        # 3.4 计算轮廓上所有点到质心的平均距离（作为半径）
        distances = []
        for point in contour:
            px, py = point[0]
            dist = np.sqrt((px - cx)**2 + (py - cy)**2)
            distances.append(dist)
        radius = np.mean(distances)

        # 3.5 计算理论面积（外接圆面积）
        theoretical_area = math.pi * radius * radius

        # 3.6 计算完整度
        completeness = actual_area / theoretical_area if theoretical_area > 0 else 0

        # 3.7 完整度过滤：丢弃 < 0.8 的颗粒
        if completeness < min_completeness:
            continue

        # 3.8 转换为物理尺寸
        diameter_pixels = 2 * radius
        diameter_nm = diameter_pixels * scale_ratio

        valid_particles.append({
            "id": len(valid_particles) + 1,
            "x": float(cx),
            "y": float(cy),
            "radius_pixels": float(radius),
            "diameter_nm": float(diameter_nm),
            "completeness": round(completeness, 4),
            "segments": contour.reshape(-1, 2).tolist()  #多边形
        })

    print(f"[分析] 有效颗粒数量: {len(valid_particles)}")

    # Step 4: 统计计算
    stats = _calculate_statistics(valid_particles)

        # Step 5: 在原图上绘制标注
    annotated_image = image.copy()
    for p in valid_particles:
        # 画轮廓（绿色）
        contour_points = np.array(p["segments"], dtype=np.int32).reshape((-1, 1, 2))
        cv2.drawContours(annotated_image, [contour_points], -1, (0, 255, 0), 2)
        
        # 画质心（红色圆点）
        cx, cy = int(p["x"]), int(p["y"])
        cv2.circle(annotated_image, (cx, cy), 3, (0, 0, 255), -1)
        
        # 标注编号
        cv2.putText(annotated_image, str(p["id"]), (cx + 8, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # 转 Base64
    import base64
    '''
    _, buffer = cv2.imencode('.png', annotated_image)
    annotated_base64 = base64.b64encode(buffer).decode('utf-8')
    '''

    _, buffer = cv2.imencode('.jpg', annotated_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    annotated_base64 = base64.b64encode(buffer).decode('utf-8')

    print(f"[分析] 使用用户提供的比例尺: 1像素 = {scale_ratio:.4f} nm")

    return {
        "total_count": stats["total_count"],
        "average_nm": stats["average_nm"],
        "d50_nm": stats["d50_nm"],
        "std_dev": stats["std_dev"],
        "min_nm": stats["min_nm"],
        "max_nm": stats["max_nm"],
        "scale_ratio": round(scale_ratio, 6),  # 实际使用的比例尺
        "scale_source": scale_source,          # 来源: user_provided 或 auto_extracted
        "particle_data": valid_particles,
        "annotated_image_base64": annotated_base64,
    }


def _extract_contours_from_yolo(model: YOLO, image: np.ndarray) -> List:
    """
    使用YOLOv8模型提取轮廓

    Args:
        model: 加载好的YOLO模型
        image: BGR格式图像（原始尺寸）

    Returns:
        轮廓列表（坐标已映射回原图尺寸）
    """
    contours = []
    orig_h, orig_w = image.shape[:2]

    # YOLOv8 推理
    results = model(image, verbose=False)

    for r in results:
        if r.masks is not None:
            masks = r.masks.data.cpu().numpy()  # shape: (N, H, W)，H和W是模型内部尺寸
            mask_h, mask_w = masks.shape[1], masks.shape[2]

            # 计算缩放比例：原图尺寸 / mask尺寸
            scale_x = orig_w / mask_w
            scale_y = orig_h / mask_h

            for mask in masks:
                # 转换为二值图（mask 内部尺寸）
                binary_mask = (mask * 255).astype(np.uint8)

                # 提取轮廓（此时坐标是 mask 尺寸下的）
                found_contours, _ = cv2.findContours(
                    binary_mask,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )

                if found_contours:
                    # 取最大轮廓
                    largest = max(found_contours, key=cv2.contourArea)
                    if cv2.contourArea(largest) > 100:  # 过滤过小的轮廓
                        # 关键修复：把轮廓坐标从 mask 尺寸映射回原图尺寸
                        largest = largest.astype(np.float32)
                        largest[:, :, 0] *= scale_x  # X 坐标缩放
                        largest[:, :, 1] *= scale_y  # Y 坐标缩放
                        largest = largest.astype(np.int32)
                        contours.append(largest)

    return contours


def _generate_mock_contours(width: int, height: int, num: int = 30) -> List:
    """
    生成模拟轮廓（用于测试）

    Args:
        width: 图像宽度
        height: 图像高度
        num: 模拟颗粒数量

    Returns:
        模拟轮廓列表
    """
    contours = []
    margin = 100  # 边缘留白

    for _ in range(num):
        cx = random.randint(margin, width - margin)
        cy = random.randint(margin, height - margin)
        radius = random.uniform(20, 50)

        # 创建带扰动的圆形轮廓
        angles = np.linspace(0, 2 * math.pi, 32, endpoint=False)
        points = []
        for angle in angles:
            r = radius * random.uniform(0.8, 1.0)
            px = cx + r * math.cos(angle)
            py = cy + r * math.sin(angle)
            points.append([[px, py]])

        contour = np.array(points, dtype=np.float32)
        contours.append(contour)

    return contours


def _is_on_image_border(
    contour,
    img_width: int,
    img_height: int,
    tolerance: int = 5
) -> bool:
    """
    检测轮廓是否触碰图像边缘

    Args:
        contour: OpenCV轮廓
        img_width: 图像宽度
        img_height: 图像高度
        tolerance: 容差像素数

    Returns:
        True 如果轮廓触碰边界
    """
    # 方法1：检查边界矩形
    x, y, w, h = cv2.boundingRect(contour)
    if x <= tolerance or y <= tolerance:
        return True
    if x + w >= img_width - tolerance or y + h >= img_height - tolerance:
        return True

    # 方法2：检查轮廓上的点（更精确）
    pts = contour.reshape(-1, 2)
    if (pts[:, 0] <= tolerance).any() or (pts[:, 0] >= img_width - tolerance).any():
        return True
    if (pts[:, 1] <= tolerance).any() or (pts[:, 1] >= img_height - tolerance).any():
        return True

    return False


def _calculate_statistics(particles: List[dict]) -> dict:
    """
    计算颗粒统计信息

    Args:
        particles: 有效颗粒列表

    Returns:
        统计结果字典
    """
    if not particles:
        return {
            "total_count": 0,
            "average_nm": 0.0,
            "d50_nm": 0.0,
            "std_dev": 0.0,
            "min_nm": 0.0,
            "max_nm": 0.0
        }

    diameters = [p["diameter_nm"] for p in particles]
    diameters_sorted = sorted(diameters)

    total_count = len(diameters)
    average_nm = float(np.mean(diameters))
    std_dev = float(np.std(diameters))
    min_nm = float(min(diameters))
    max_nm = float(max(diameters))

    # D50: 中位径
    median_idx = total_count // 2
    d50_nm = diameters_sorted[median_idx]

    return {
        "total_count": total_count,
        "average_nm": round(average_nm, 2),
        "d50_nm": round(d50_nm, 2),
        "std_dev": round(std_dev, 2),
        "min_nm": round(min_nm, 2),
        "max_nm": round(max_nm, 2)
    }
