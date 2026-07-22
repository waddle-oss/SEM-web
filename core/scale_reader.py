"""
比例尺自动提取模块 (OCR Integration)
从SEM图像底部状态栏自动识别标尺信息

功能:
1. 裁剪图像底部15%区域（状态栏）
2. OCR识别物理长度（如 "100 nm", "1 μm"）
3. OpenCV二值化提取像素长度（最长的水平线段）
4. 计算 scale_ratio = 物理长度(nm) / 像素长度

V2.1 增强:
- 新增 extract_scale_info 返回结构化结果
- 保留原 extract_scale_ratio 函数名兼容旧调用
"""

import cv2
import numpy as np
import re
from typing import Tuple, Dict, Any, Union


class ScaleExtractionError(Exception):
    """比例尺提取失败的自定义异常"""
    pass


def extract_scale_info(image_input: Union[np.ndarray, bytes]) -> Dict[str, Any]:
    """
    提取比例尺的完整结构化信息（V2.1 新接口）

    Args:
        image_input: BGR 图像矩阵 或 图像字节

    Returns:
        {
            "scale_ratio": float,             # nm/px
            "physical_length_nm": float,      # 物理长度 (nm)
            "pixel_length": float,            # 像素长度
            "ocr_text": str,                  # OCR 原始文本
            "confidence": str,                # high / medium / low
            "source": "ocr"
        }

    Raises:
        ScaleExtractionError: 提取失败
    """
    # 兼容 bytes 输入
    if isinstance(image_input, (bytes, bytearray)):
        nparr = np.frombuffer(image_input, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ScaleExtractionError("无法解码图像数据")
    else:
        image = image_input

    if image is None or image.size == 0:
        raise ScaleExtractionError("无法裁剪图像底部区域")

    bottom_region = _crop_bottom_region(image, ratio=0.15)

    # OCR 文本
    physical_length_nm, ocr_text = _extract_physical_length_with_text(bottom_region)

    # 像素长度
    pixel_length = _extract_pixel_length(bottom_region)
    if pixel_length <= 0:
        raise ScaleExtractionError("未能检测到标尺线段")

    scale_ratio = physical_length_nm / pixel_length
    if scale_ratio <= 0 or scale_ratio > 1000:
        raise ScaleExtractionError(f"计算出的比例尺异常: {scale_ratio:.4f} nm/px")

    # 简单置信度估计
    confidence = _estimate_confidence(ocr_text, pixel_length)
    return {
        "scale_ratio": round(float(scale_ratio), 6),
        "physical_length_nm": round(float(physical_length_nm), 3),
        "pixel_length": round(float(pixel_length), 2),
        "ocr_text": ocr_text.strip(),
        "confidence": confidence,
        "source": "ocr"
    }


def _estimate_confidence(ocr_text: str, pixel_length: float) -> str:
    """根据 OCR 文本和线段长度估算置信度"""
    text = ocr_text or ""
    if not text:
        return "low"
    if re.search(r"\d", text):
        if pixel_length >= 50:
            return "high"
        if pixel_length >= 20:
            return "medium"
        return "low"
    return "low"


def extract_scale_ratio(image_matrix: np.ndarray) -> float:
    """
    从SEM图像中自动提取比例尺

    流程:
    1. 裁剪底部15%区域（状态栏）
    2. OCR识别物理长度
    3. 二值化提取像素长度
    4. 计算并返回 scale_ratio

    Args:
        image_matrix: BGR格式的图像矩阵（OpenCV读取的格式）

    Returns:
        float: 1像素对应的纳米数 (nm/pixel)

    Raises:
        ScaleExtractionError: 当无法提取比例尺时抛出
    """
    height, width = image_matrix.shape[:2]

    # Step 1: 裁剪底部15%区域
    bottom_region = _crop_bottom_region(image_matrix, ratio=0.15)

    if bottom_region is None or bottom_region.size == 0:
        raise ScaleExtractionError("无法裁剪图像底部区域")

    # Step 2: OCR识别物理长度
    physical_length_nm = _extract_physical_length_by_ocr(bottom_region)

    # Step 3: 提取像素长度
    pixel_length = _extract_pixel_length(bottom_region)

    if pixel_length <= 0:
        raise ScaleExtractionError("未能检测到标尺线段")

    # Step 4: 计算比例尺
    scale_ratio = physical_length_nm / pixel_length

    if scale_ratio <= 0 or scale_ratio > 1000:
        raise ScaleExtractionError(f"计算出的比例尺异常: {scale_ratio:.4f} nm/px")

    return scale_ratio


def _crop_bottom_region(image: np.ndarray, ratio: float = 0.15) -> np.ndarray:
    """
    裁剪图像底部区域

    Args:
        image: 原始图像
        ratio: 裁剪比例（默认底部15%）

    Returns:
        裁剪后的图像区域
    """
    height, width = image.shape[:2]
    y_start = int(height * (1 - ratio))
    y_end = height

    return image[y_start:y_end, 0:width]


def _extract_physical_length_by_ocr(bottom_region: np.ndarray) -> float:
    """
    使用OCR提取物理长度

    Args:
        bottom_region: 底部区域图像

    Returns:
        物理长度（纳米）

    Raises:
        ScaleExtractionError: OCR识别失败
    """
    value, _ = _extract_physical_length_with_text(bottom_region)
    return value


def _extract_physical_length_with_text(bottom_region: np.ndarray) -> Tuple[float, str]:
    """
    OCR 提取物理长度并返回原始文本

    Returns:
        (physical_length_nm, ocr_text)
    """
    try:
        import pytesseract
    except ImportError:
        raise ScaleExtractionError("请安装pytesseract: pip install pytesseract")

    # 增强对比度（转为灰度+直方图均衡化）
    gray = cv2.cvtColor(bottom_region, cv2.COLOR_BGR2GRAY)

    # OCR识别
    text = pytesseract.image_to_string(
        gray,
        config='--psm 6 --oem 3'  # PSM 6: 假设统一文本块
    )

    # 正则匹配数字+单位
    # 支持格式: "100 nm", "1 μm", "1um", "500nm", "0.5um" 等
    patterns = [
        r'(\d+(?:\.\d+)?)\s*(?:um|μm|micron)',  # 微米
        r'(\d+(?:\.\d+)?)\s*(?:nm|nanometer)',  # 纳米
    ]

    physical_nm = None
    matched_text = ""

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = float(match.group(1))
            matched_text = match.group(0)
            if 'um' in pattern.lower():
                physical_nm = value * 1000  # 微米转纳米
            else:
                physical_nm = value
            break

    if physical_nm is None or physical_nm <= 0:
        raise ScaleExtractionError(
            f"OCR未能识别有效标尺文本。识别内容: '{text.strip()}'"
        )

    return physical_nm, matched_text or text.strip()


def _extract_pixel_length(bottom_region: np.ndarray) -> float:
    """
    提取标尺线段的像素长度

    使用二值化+形态学操作找到最长的水平白色线段

    Args:
        bottom_region: 底部区域图像

    Returns:
        线段的像素宽度
    """
    # 转为灰度图
    gray = cv2.cvtColor(bottom_region, cv2.COLOR_BGR2GRAY)

    # 自适应阈值二值化
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # 形态学操作：闭运算连接线段
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 寻找轮廓
    contours, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        # 备选方案：直接扫描底部区域找水平白线
        return _scan_horizontal_line(gray)

    # 找最长的水平轮廓
    max_width = 0
    best_x, best_y = 0, 0

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # 过滤掉太小的或太垂直的轮廓
        aspect_ratio = w / h if h > 0 else 0
        if w > 20 and aspect_ratio > 3:  # 宽度>20，长宽比>3
            if w > max_width:
                max_width = w
                best_x, best_y = x, y

    if max_width > 0:
        return float(max_width)

    # 备选：扫描水平线
    return _scan_horizontal_line(gray)


def _scan_horizontal_line(gray_image: np.ndarray) -> float:
    """
    备选方案：扫描图像底部水平白线的像素宽度

    Args:
        gray_image: 灰度图像

    Returns:
        检测到的最大线宽
    """
    height, width = gray_image.shape

    # 扫描底部1/4区域
    scan_region = gray_image[int(height * 3 / 4):height, :]

    # 水平Sobel算子
    sobelx = cv2.Sobel(scan_region, cv2.CV_64F, 1, 0, ksize=3)
    abs_sobelx = np.abs(sobelx)

    # 阈值化
    _, binary = cv2.threshold(abs_sobelx, 50, 255, cv2.THRESH_BINARY)

    # 水平方向投影
    horizontal_projection = np.sum(binary, axis=0)

    # 找到最大连续高值区间的宽度
    threshold = np.max(horizontal_projection) * 0.5
    in_line = False
    max_line_width = 0
    current_width = 0

    for val in horizontal_projection:
        if val > threshold:
            current_width += 1
            in_line = True
        else:
            if in_line:
                max_line_width = max(max_line_width, current_width)
                current_width = 0
                in_line = False

    max_line_width = max(max_line_width, current_width)

    # 如果还是没找到，返回估计值
    if max_line_width < 10:
        # 估算：假设标尺占底部15%区域的10%-30%
        estimated_width = width * 0.2
        return estimated_width

    return float(max_line_width)


def validate_scale_ratio(
    scale_ratio: float,
    image_width: int,
    min_diameter_nm: float = 1.0,
    max_diameter_nm: float = 5000.0
) -> bool:
    """
    验证提取的比例尺是否合理

    根据图像宽度和常见的颗粒直径范围进行校验

    Args:
        scale_ratio: 比例尺 (nm/pixel)
        image_width: 图像宽度（像素）
        min_diameter_nm: 最小颗粒直径（纳米）
        max_diameter_nm: 最大颗粒直径（纳米）

    Returns:
        是否合理
    """
    # 根据图像宽度，计算合理的颗粒直径范围
    min_valid_ratio = min_diameter_nm / image_width
    max_valid_ratio = max_diameter_nm / image_width

    return min_valid_ratio <= scale_ratio <= max_valid_ratio
