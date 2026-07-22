"""
文件校验模块
统一图像文件大小、格式校验
"""

import os
from typing import Tuple, Optional

ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
MAX_IMAGE_SIZE = 50 * 1024 * 1024  # 50MB


def validate_image_file(filename: str, image_bytes: bytes) -> Tuple[bool, str]:
    """
    校验图像文件

    Args:
        filename: 文件名
        image_bytes: 图像字节

    Returns:
        (is_valid, error_message)；通过时 error_message 为空串
    """
    if not filename:
        return False, "文件名不能为空"

    ext = os.path.splitext(filename.lower())[1]
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        return False, f"不支持的图像格式: {ext or '(无后缀)'}"

    if not image_bytes:
        return False, "图像文件为空"

    if len(image_bytes) > MAX_IMAGE_SIZE:
        size_mb = len(image_bytes) / (1024 * 1024)
        return False, f"图像文件过大 ({size_mb:.1f}MB)，最大 50MB"

    return True, ""


def get_image_format(filename: str) -> str:
    """从文件名提取格式字符串（不带点号）"""
    ext = os.path.splitext(filename.lower())[1]
    return ext.lstrip(".") if ext else "unknown"


# =========================================================
# V2.1.1 过滤参数默认值与校验
# =========================================================

DEFAULT_COMPLETENESS_THRESHOLD = 0.8
DEFAULT_EDGE_TOLERANCE = 5
DEFAULT_MIN_AREA = 100.0
DEFAULT_MAX_AREA = None  # None 表示不限制
DEFAULT_EXCLUDE_EDGE_PARTICLES = True


def validate_filter_params(
    completeness_threshold: float,
    edge_tolerance: int,
    min_area: float,
    max_area: Optional[float],
    exclude_edge_particles: bool
) -> dict:
    """
    校验过滤参数是否合法

    Args:
        completeness_threshold: 完整度阈值 [0, 1]
        edge_tolerance: 边缘容差 px [0, 200]
        min_area: 最小面积 px² (>= 0)
        max_area: 最大面积 px² (None 或 > min_area)
        exclude_edge_particles: 是否排除边缘颗粒

    Returns:
        {"valid": bool, "errors": [str, ...]}
    """
    errors = []

    try:
        ct = float(completeness_threshold)
        if ct < 0 or ct > 1:
            errors.append("completeness_threshold 必须在 0 到 1 之间")
    except (TypeError, ValueError):
        errors.append("completeness_threshold 不是合法数字")

    try:
        et = int(edge_tolerance)
        if et < 0 or et > 200:
            errors.append("edge_tolerance 必须在 0 到 200 之间")
    except (TypeError, ValueError):
        errors.append("edge_tolerance 不是合法整数")

    try:
        ma = float(min_area) if min_area not in (None, "") else 0.0
        if ma < 0:
            errors.append("min_area 必须 >= 0")
    except (TypeError, ValueError):
        errors.append("min_area 不是合法数字")
        ma = 0.0

    if max_area not in (None, "", 0):
        try:
            mx = float(max_area)
            if mx <= ma:
                errors.append("max_area 必须大于 min_area")
        except (TypeError, ValueError):
            errors.append("max_area 不是合法数字")

    if not isinstance(exclude_edge_particles, bool):
        # FastAPI 在 Form 中会传字符串
        if str(exclude_edge_particles).lower() not in ("true", "false", "1", "0"):
            errors.append("exclude_edge_particles 必须为 true/false")

    return {"valid": len(errors) == 0, "errors": errors}


def normalize_filter_params(
    completeness_threshold=None,
    edge_tolerance=None,
    min_area=None,
    max_area=None,
    exclude_edge_particles=None
) -> dict:
    """
    把传入参数归一化为默认值 + 覆盖

    返回 dict，可直接传入 analyze_particles
    """
    return {
        "completeness_threshold": (
            float(completeness_threshold) if completeness_threshold not in (None, "")
            else DEFAULT_COMPLETENESS_THRESHOLD
        ),
        "edge_tolerance": (
            int(edge_tolerance) if edge_tolerance not in (None, "")
            else DEFAULT_EDGE_TOLERANCE
        ),
        "min_area": (
            float(min_area) if min_area not in (None, "")
            else DEFAULT_MIN_AREA
        ),
        "max_area": (
            float(max_area) if max_area not in (None, "", 0)
            else DEFAULT_MAX_AREA
        ),
        "exclude_edge_particles": (
            bool(exclude_edge_particles) if isinstance(exclude_edge_particles, bool)
            else str(exclude_edge_particles).lower() in ("true", "1")
            if exclude_edge_particles is not None
            else DEFAULT_EXCLUDE_EDGE_PARTICLES
        )
    }