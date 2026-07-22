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


def get_model_path() -> Optional[str]:
    """获取当前已加载模型的路径"""
    return _encrypted_model_path


def load_model_from_path(model_path: str) -> YOLO:
    """
    按文件类型加载模型

    - .encrypted: 走阅后即焚解密流程
    - .pt / .onnx 等: 直接由 YOLO 加载明文权重
    """
    if not ULTRALYTICS_AVAILABLE:
        raise ImportError("需要安装 ultralytics: pip install ultralytics")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    if model_path.lower().endswith(".encrypted"):
        return load_model_securely(model_path)

    print(f"[模型加载] 加载明文权重: {model_path}")
    model = YOLO(model_path)
    print("[模型加载] ✓ 模型加载成功")
    return model


def initialize_model(model_path: str, force: bool = False) -> YOLO:
    """
    初始化/切换模型单例

    Args:
        model_path: 模型文件路径（.encrypted / .pt / .onnx）
        force: True 时强制重新加载（用于前端切换模型）

    Returns:
        YOLO 模型对象
    """
    global _model_instance, _encrypted_model_path

    need_reload = (
        force
        or _model_instance is None
        or _encrypted_model_path != model_path
    )

    if need_reload:
        _model_instance = load_model_from_path(model_path)
        _encrypted_model_path = model_path

    return _model_instance


# =========================================================
# B. 分析算法
# =========================================================

def analyze_particles(
    image_bytes: bytes,
    scale_ratio: float = None,
    use_mock: bool = True,
    completeness_threshold: float = 0.8,
    edge_tolerance: int = 5,
    min_area: float = 100,
    max_area=None,
    exclude_edge_particles: bool = True
) -> dict:
    """
    分析SEM图像中的颗粒（V2.1.1：参数真实生效）

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
        completeness_threshold: 完整度阈值 [0, 1]，默认 0.8
        edge_tolerance: 边缘容差 px，默认 5
        min_area: 最小像素面积阈值，默认 100
        max_area: 最大像素面积阈值，None 表示不限制
        exclude_edge_particles: 是否排除边缘颗粒，默认 True

    Returns:
        包含分析结果的字典（同时返回 filter_params 用于可复现追溯）:
        {
            "total_count": int,            # == valid_count 兼容旧前端
            "valid_count": int,
            "excluded_count": int,
            "total_detected": int,          # == valid_count + excluded_count
            "average_nm": float,
            "d50_nm": float,
            "std_dev": float,
            "min_nm": float,
            "max_nm": float,
            "scale_ratio": float,
            "scale_source": str,
            "particle_data": [...],
            "excluded_particles": [...],
            "exclusion_stats": {...},
            "filter_params": {...},         # 实际使用的过滤参数（可复现）
            "annotated_image_base64": str
        }
    """
    # Step 1: 解码图像
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("无法解码图像数据")

    height, width = image.shape[:2]
    print(f"[分析] 图像尺寸: {width}x{height}")
    print(f"[分析] 过滤参数: completeness>={completeness_threshold}, edge_tol={edge_tolerance}, "
          f"min_area={min_area}, max_area={max_area}, exclude_edge={exclude_edge_particles}")

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

    # Step 3: 过滤和处理每个轮廓（V2.1.1 真实参数生效）
    valid_particles = []
    excluded_particles = []
    edge_tol = int(edge_tolerance)
    min_area_threshold = float(min_area)
    max_area_threshold = float(max_area) if max_area not in (None, 0) else None

    for i, contour in enumerate(contours):
        # 3.0 计算面积 / 质心 / 包围盒 / 边缘判定（必须在过滤前全部算好）
        actual_area = float(cv2.contourArea(contour))

        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
        else:
            cx, cy = 0.0, 0.0

        bx, by, bw, bh = cv2.boundingRect(contour)
        # touches_edge 判定：包围盒任一边在容差内
        touches_edge = (
            bx <= edge_tol
            or by <= edge_tol
            or (bx + bw) >= width - edge_tol
            or (by + bh) >= height - edge_tol
        )

        # 3.x 过滤顺序：small_area → large_area → edge → low_completeness
        exclude_reason = None

        if actual_area < min_area_threshold:
            exclude_reason = "small_area"
        elif max_area_threshold is not None and actual_area > max_area_threshold:
            exclude_reason = "large_area"
        elif exclude_edge_particles and touches_edge:
            exclude_reason = "edge"
        else:
            # 完整度计算
            distances = []
            for point in contour:
                px, py = point[0]
                distances.append(math.sqrt((px - cx) ** 2 + (py - cy) ** 2))
            radius = float(np.mean(distances)) if distances else 0.0
            theoretical_area = math.pi * radius * radius
            completeness = actual_area / theoretical_area if theoretical_area > 0 else 0

            if completeness < completeness_threshold:
                exclude_reason = "low_completeness"

        if exclude_reason:
            excluded_particles.append(_build_excluded_record_v211(
                contour, scale_ratio, exclude_reason,
                actual_area=actual_area, cx=cx, cy=cy,
                bbox=(bx, by, bw, bh), touches_edge=touches_edge
            ))
            continue

        # 通过过滤：构造有效颗粒
        # 此处再次计算 radius/completeness（上面分支可能没算；radius 仅用于完整度几何）
        distances = []
        for point in contour:
            px, py = point[0]
            distances.append(math.sqrt((px - cx) ** 2 + (py - cy) ** 2))
        radius = float(np.mean(distances)) if distances else 0.0
        theoretical_area = math.pi * radius * radius
        completeness = actual_area / theoretical_area if theoretical_area > 0 else 0

        diameter_pixels = 2 * radius
        diameter_nm = diameter_pixels * scale_ratio
        avg_diameter_nm = float(diameter_nm)
        eq_diameter_px = 2 * math.sqrt(actual_area / math.pi) if actual_area > 0 else 0.0
        eq_diameter_nm = float(eq_diameter_px * scale_ratio)

        perimeter = cv2.arcLength(contour, True)
        # 圆度 = 4πA / P²（完美圆为 1）
        roundness = (4 * math.pi * actual_area / (perimeter ** 2)) if perimeter > 0 else None

        valid_particles.append({
            "id": len(valid_particles) + 1,
            "x": float(cx),
            "y": float(cy),
            "area_pixels": round(actual_area, 3),
            "diameter_pixels": float(diameter_pixels),
            "avg_diameter_nm": round(avg_diameter_nm, 3),
            "eq_diameter_nm": round(eq_diameter_nm, 3),
            "diameter_nm": round(float(diameter_nm), 3),
            "completeness": round(completeness, 4),
            "roundness": round(float(roundness), 4) if roundness is not None else None,
            "circularity": round(float(roundness), 4) if roundness is not None else None,
            "bbox": [int(bx), int(by), int(bw), int(bh)],
            "touches_edge": bool(touches_edge),
            "confidence": None,
            "status": "valid",
            "exclude_reason": None,
            "segments": contour.reshape(-1, 2).tolist()
        })

    print(f"[分析] 有效颗粒数量: {len(valid_particles)}")

    # 给排除颗粒补 id
    for idx, p in enumerate(excluded_particles, start=len(valid_particles) + 1):
        p["id"] = idx

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

    # 引入统一异常避免循环依赖
    from core.stats import build_exclusion_stats

    valid_count = stats["total_count"]
    excluded_count = len(excluded_particles)
    total_detected = valid_count + excluded_count
    return {
        # 统计口径：直径(nm)
        "total_count": valid_count,   # == valid_count，兼容旧前端
        "average_nm": stats["average_nm"],
        "average_diameter_nm": stats["average_diameter_nm"],
        "d50_nm": stats["d50_nm"],
        "std_dev": stats["std_dev"],
        "min_nm": stats["min_nm"],
        "max_nm": stats["max_nm"],
        "min_diameter_nm": stats["min_diameter_nm"],
        "max_diameter_nm": stats["max_diameter_nm"],
        "avg_roundness": stats["avg_roundness"],
        "scale_ratio": round(scale_ratio, 6),
        "scale_source": scale_source,
        "particle_data": valid_particles,
        "annotated_image_base64": annotated_base64,
        # V2.1 新增字段
        "success": True,
        "total_detected": total_detected,
        "valid_count": valid_count,
        "excluded_count": excluded_count,
        "excluded_particles": excluded_particles,
        "exclusion_stats": build_exclusion_stats(excluded_particles),
        # V2.1.1 新增：实际使用的过滤参数（可复现追溯）
        "filter_params": {
            "completeness_threshold": float(completeness_threshold),
            "edge_tolerance": int(edge_tolerance),
            "min_area": float(min_area),
            "max_area": (float(max_area) if max_area not in (None, 0) else None),
            "exclude_edge_particles": bool(exclude_edge_particles)
        }
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


def _build_excluded_record(
    contour,
    scale_ratio: float,
    reason: str,
    extra: dict = None
) -> dict:
    """
    构造排除颗粒记录（V2.1 基础版）
    """
    extra = extra or {}
    area = float(cv2.contourArea(contour))

    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
    else:
        cx, cy = 0.0, 0.0

    distances = []
    for point in contour:
        px, py = point[0]
        distances.append(math.sqrt((px - cx) ** 2 + (py - cy) ** 2))
    radius = float(np.mean(distances)) if distances else 0.0
    diameter_pixels = 2 * radius
    diameter_nm = diameter_pixels * scale_ratio
    avg_diameter_nm = diameter_nm
    eq_diameter_px = 2 * math.sqrt(area / math.pi) if area > 0 else 0.0
    eq_diameter_nm = eq_diameter_px * scale_ratio

    bx, by, bw, bh = cv2.boundingRect(contour)
    perimeter = cv2.arcLength(contour, True)
    roundness = (4 * math.pi * area / (perimeter ** 2)) if perimeter > 0 else None

    return {
        "x": float(cx),
        "y": float(cy),
        "area_pixels": round(area, 3),
        "diameter_pixels": round(diameter_pixels, 3),
        "avg_diameter_nm": round(float(avg_diameter_nm), 3),
        "eq_diameter_nm": round(float(eq_diameter_nm), 3),
        "diameter_nm": round(float(diameter_nm), 3),
        "completeness": extra.get("completeness"),
        "roundness": round(float(roundness), 4) if roundness is not None else None,
        "circularity": round(float(roundness), 4) if roundness is not None else None,
        "bbox": [int(bx), int(by), int(bw), int(bh)],
        "touches_edge": bool(extra.get("touches_edge", False)),
        "confidence": None,
        "status": "excluded",
        "exclude_reason": reason,
        "segments": contour.reshape(-1, 2).tolist() if contour is not None else []
    }


def _build_excluded_record_v211(
    contour,
    scale_ratio: float,
    reason: str,
    actual_area: float = None,
    cx: float = None,
    cy: float = None,
    bbox=None,
    touches_edge: bool = False
) -> dict:
    """
    V2.1.1 排除颗粒记录（在主循环中预计算字段，避免重复计算）
    """
    if actual_area is None:
        actual_area = float(cv2.contourArea(contour))
    if cx is None or cy is None:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
        else:
            cx, cy = 0.0, 0.0
    if bbox is None:
        bx, by, bw, bh = cv2.boundingRect(contour)
    else:
        bx, by, bw, bh = bbox

    distances = []
    for point in contour:
        px, py = point[0]
        distances.append(math.sqrt((px - cx) ** 2 + (py - cy) ** 2))
    radius = float(np.mean(distances)) if distances else 0.0
    diameter_pixels = 2 * radius
    diameter_nm = diameter_pixels * scale_ratio
    avg_diameter_nm = diameter_nm
    eq_diameter_px = 2 * math.sqrt(actual_area / math.pi) if actual_area > 0 else 0.0
    eq_diameter_nm = eq_diameter_px * scale_ratio

    perimeter = cv2.arcLength(contour, True)
    roundness = (4 * math.pi * actual_area / (perimeter ** 2)) if perimeter > 0 else None

    # 完整度（用于 low_completeness 排除颗粒的记录）
    if radius > 0:
        theoretical_area = math.pi * radius * radius
        completeness = actual_area / theoretical_area if theoretical_area > 0 else 0
    else:
        completeness = None

    return {
        "x": float(cx),
        "y": float(cy),
        "area_pixels": round(actual_area, 3),
        "diameter_pixels": round(diameter_pixels, 3),
        "avg_diameter_nm": round(float(avg_diameter_nm), 3),
        "eq_diameter_nm": round(float(eq_diameter_nm), 3),
        "diameter_nm": round(float(diameter_nm), 3),
        "completeness": round(float(completeness), 4) if completeness is not None else None,
        "roundness": round(float(roundness), 4) if roundness is not None else None,
        "circularity": round(float(roundness), 4) if roundness is not None else None,
        "bbox": [int(bx), int(by), int(bw), int(bh)],
        "touches_edge": bool(touches_edge),
        "confidence": None,
        "status": "excluded",
        "exclude_reason": reason,
        "segments": contour.reshape(-1, 2).tolist() if contour is not None else []
    }


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
            "average_diameter_nm": 0.0,
            "d50_nm": 0.0,
            "std_dev": 0.0,
            "min_nm": 0.0,
            "max_nm": 0.0,
            "min_diameter_nm": 0.0,
            "max_diameter_nm": 0.0,
            "avg_roundness": 0.0
        }

    # 优先使用平均直径；兼容旧半径字段（×2）
    diameters = []
    for p in particles:
        if p.get("avg_diameter_nm") is not None:
            diameters.append(float(p["avg_diameter_nm"]))
        elif p.get("diameter_nm") is not None:
            diameters.append(float(p["diameter_nm"]))
        elif p.get("avg_radius_nm") is not None:
            diameters.append(float(p["avg_radius_nm"]) * 2.0)
    diameters_sorted = sorted(diameters)

    total_count = len(diameters)
    average_diameter_nm = float(np.mean(diameters)) if diameters else 0.0
    std_dev = float(np.std(diameters)) if diameters else 0.0
    min_diameter_nm = float(min(diameters)) if diameters else 0.0
    max_diameter_nm = float(max(diameters)) if diameters else 0.0
    median_idx = total_count // 2 if total_count else 0
    d50_nm = diameters_sorted[median_idx] if diameters_sorted else 0.0

    roundness_vals = []
    for p in particles:
        if isinstance(p.get("roundness"), (int, float)):
            roundness_vals.append(float(p["roundness"]))
        elif isinstance(p.get("circularity"), (int, float)):
            roundness_vals.append(float(p["circularity"]))
    avg_roundness = float(np.mean(roundness_vals)) if roundness_vals else 0.0

    return {
        "total_count": total_count,
        "average_nm": round(average_diameter_nm, 2),
        "average_diameter_nm": round(average_diameter_nm, 2),
        "d50_nm": round(d50_nm, 2),
        "std_dev": round(std_dev, 2),
        "min_nm": round(min_diameter_nm, 2),
        "max_nm": round(max_diameter_nm, 2),
        "min_diameter_nm": round(min_diameter_nm, 2),
        "max_diameter_nm": round(max_diameter_nm, 2),
        "avg_roundness": round(avg_roundness, 4)
    }
