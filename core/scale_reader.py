"""
比例尺自动提取模块

设计原则：
1. 禁止整图 OCR 直接读出 nm/px
2. 先定位比例尺区域（优先底部），再 OCR 标尺文字（如 "100 nm"）
3. 用图像处理测量标尺线像素长度
4. nm/px = physical_length_nm / pixel_length
"""

from __future__ import annotations

import cv2
import numpy as np
import re
from typing import Any, Dict, List, Optional, Tuple, Union


class ScaleExtractionError(Exception):
    """比例尺提取失败的自定义异常"""

    def __init__(self, message: str, debug: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.debug = debug


# 常见 SEM 标尺文字：数字 + 单位（绝不把 OCR 结果当成 nm/px）
_SCALE_PATTERNS = [
    # 微米优先（避免把 "1 μm" 误当成 nm）
    (re.compile(r"(\d+(?:\.\d+)?)\s*(?:μm|µm|um|micron)s?\b", re.IGNORECASE), 1000.0),
    (re.compile(r"(\d+(?:\.\d+)?)\s*(?:nm|nanometers?)\b", re.IGNORECASE), 1.0),
]


# 自动标尺线检测：当前 OCR 模式改由用户两点标注提供像素长度，默认禁用自动检线。
# 保留 _detect_scale_bar 等实现，便于日后开关恢复。
USE_AUTO_BAR_DETECTION = False


def extract_scale_info(
    image_input: Union[np.ndarray, bytes],
    debug: bool = False,
    pixel_length: Optional[float] = None,
    physical_length_nm: Optional[float] = None,
    bar_points: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    提取比例尺结构化信息。

    推荐路径（OCR 模式）：
      前端先手动两点标注得到 pixel_length，再调用本函数只做 OCR 文字识别。
      nm/px = physical_length_nm / pixel_length

    Args:
        image_input: 图像
        debug: 是否返回调试信息
        pixel_length: 用户标注的标尺线像素长度（优先）
        physical_length_nm: 可选，OCR 失败时由用户手动填入的物理长度(nm)
        bar_points: 可选，用户标注端点 {"x1","y1","x2","y2"}
    """
    image = _decode_image(image_input)
    height, width = image.shape[:2]

    # ---- 新路径：用户提供标尺线像素长度，OCR 只负责文字 ----
    if pixel_length is not None:
        return _extract_with_manual_bar(
            image,
            pixel_length=float(pixel_length),
            physical_length_nm=physical_length_nm,
            bar_points=bar_points,
            debug=debug,
        )

    # ---- 旧路径：自动检线（默认禁用）----
    if not USE_AUTO_BAR_DETECTION:
        raise ScaleExtractionError(
            "已禁用自动标尺线检测，请先在图像上标注标尺线两端，再识别文字"
        )

    return _extract_with_auto_bar(image, debug=debug)


def _extract_with_manual_bar(
    image: np.ndarray,
    pixel_length: float,
    physical_length_nm: Optional[float],
    bar_points: Optional[Dict[str, float]],
    debug: bool,
) -> Dict[str, Any]:
    """OCR 文字 + 用户标注的像素长度 → nm/px。"""
    if pixel_length <= 0:
        raise ScaleExtractionError("标尺像素长度必须大于 0")

    height, width = image.shape[:2]
    debug_log: Dict[str, Any] = {
        "image_size": {"width": width, "height": height},
        "mode": "manual_bar + ocr_text",
        "pixel_length_user": pixel_length,
        "regions_tried": [],
        "selected_strategy": None,
        "calculation": None,
        "note": "Bar length from user annotation; OCR only reads scale label text",
    }

    matched_text = ""
    ocr_raw = ""
    physical_nm: Optional[float] = None
    region_used: Optional[Dict[str, Any]] = None
    last_error = "OCR未能识别有效标尺文字"

    for region_spec in _iter_search_regions(height, width):
        crop = image[
            region_spec["y"] : region_spec["y"] + region_spec["h"],
            region_spec["x"] : region_spec["x"] + region_spec["w"],
        ]
        if crop.size == 0:
            continue

        attempt: Dict[str, Any] = {
            "strategy": region_spec["strategy"],
            "region": {
                "x": region_spec["x"],
                "y": region_spec["y"],
                "w": region_spec["w"],
                "h": region_spec["h"],
            },
            "ocr_raw_text": "",
            "ocr_matched": None,
            "ok": False,
            "error": None,
        }
        try:
            phys, matched, raw = _ocr_scale_label(crop)
            attempt["ocr_raw_text"] = raw
            attempt["ocr_matched"] = matched
            attempt["physical_length_nm"] = phys
            attempt["ok"] = True
            debug_log["regions_tried"].append(attempt)
            debug_log["selected_strategy"] = region_spec["strategy"]
            physical_nm = phys
            matched_text = matched
            ocr_raw = raw
            region_used = attempt["region"]
            region_used["strategy"] = region_spec["strategy"]
            break
        except ScaleExtractionError as e:
            attempt["error"] = str(e)
            attempt["ocr_raw_text"] = str(e)
            debug_log["regions_tried"].append(attempt)
            last_error = str(e)

    # OCR 失败：允许调用方传入 physical_length_nm 兜底
    if physical_nm is None:
        if physical_length_nm is not None and float(physical_length_nm) > 0:
            physical_nm = float(physical_length_nm)
            matched_text = matched_text or f"{physical_nm:g} nm (manual)"
            ocr_raw = ocr_raw or "(ocr failed; physical length from user)"
            debug_log["selected_strategy"] = "user_physical_length"
        else:
            raise ScaleExtractionError(
                f"{last_error}。请手动输入物理长度（nm）后重试",
                debug=debug_log if debug else None,
            )

    scale_ratio = physical_nm / pixel_length
    if scale_ratio <= 0 or scale_ratio > 1000:
        raise ScaleExtractionError(f"计算出的比例尺异常: {scale_ratio:.6f} nm/px")

    confidence = _estimate_confidence(matched_text, pixel_length, ocr_raw)
    calculation = f"{physical_nm:g} / {pixel_length:.2f} = {scale_ratio:.6f}"
    debug_log["calculation"] = calculation

    bar = None
    if bar_points and all(k in bar_points for k in ("x1", "y1", "x2", "y2")):
        bar = {
            "x1": int(bar_points["x1"]),
            "y1": int(bar_points["y1"]),
            "x2": int(bar_points["x2"]),
            "y2": int(bar_points["y2"]),
            "length_px": round(float(pixel_length), 2),
            "source": "user_annotation",
        }
    else:
        bar = {
            "x1": 0,
            "y1": 0,
            "x2": int(round(pixel_length)),
            "y2": 0,
            "length_px": round(float(pixel_length), 2),
            "source": "user_annotation",
        }

    result = {
        "scale_ratio": round(float(scale_ratio), 6),
        "physical_length_nm": round(float(physical_nm), 3),
        "pixel_length": round(float(pixel_length), 2),
        "ocr_text": matched_text,
        "ocr_raw_text": (ocr_raw or "").strip(),
        "confidence": confidence,
        "source": "ocr",
        "region": region_used,
        "bar": bar,
    }
    if debug:
        result["debug"] = debug_log
    return result


def _extract_with_auto_bar(image: np.ndarray, debug: bool = False) -> Dict[str, Any]:
    """
    【已禁用默认入口】旧版：区域 OCR + 自动检测标尺线。
    代码保留，仅当 USE_AUTO_BAR_DETECTION=True 时由 extract_scale_info 调用。
    """
    height, width = image.shape[:2]

    debug_log: Dict[str, Any] = {
        "image_size": {"width": width, "height": height},
        "regions_tried": [],
        "selected_strategy": None,
        "calculation": None,
        "note": "OCR only reads scale labels; nm/px is computed as physical_nm / bar_px",
        "mode": "auto_bar (legacy)",
    }

    last_error = "未能在比例尺区域中识别到有效标尺文字与线段"
    best_partial: Optional[Dict[str, Any]] = None

    for region_spec in _iter_search_regions(height, width):
        crop = image[
            region_spec["y"] : region_spec["y"] + region_spec["h"],
            region_spec["x"] : region_spec["x"] + region_spec["w"],
        ]
        if crop.size == 0:
            continue

        attempt: Dict[str, Any] = {
            "strategy": region_spec["strategy"],
            "region": {
                "x": region_spec["x"],
                "y": region_spec["y"],
                "w": region_spec["w"],
                "h": region_spec["h"],
            },
            "ocr_raw_text": "",
            "ocr_matched": None,
            "physical_length_nm": None,
            "bar": None,
            "ok": False,
            "error": None,
        }

        try:
            physical_nm, matched_text, ocr_raw = _ocr_scale_label(crop)
            attempt["ocr_raw_text"] = ocr_raw
            attempt["ocr_matched"] = matched_text
            attempt["physical_length_nm"] = physical_nm

            # 自动检线（保留代码，默认入口已关闭）
            bar_local = _detect_scale_bar(crop)
            if bar_local is None:
                raise ScaleExtractionError("该区域未检测到有效标尺横线")

            bar_global = {
                "x1": region_spec["x"] + bar_local["x1"],
                "y1": region_spec["y"] + bar_local["y1"],
                "x2": region_spec["x"] + bar_local["x2"],
                "y2": region_spec["y"] + bar_local["y2"],
                "length_px": bar_local["length_px"],
            }
            attempt["bar"] = bar_global

            pixel_length = float(bar_local["length_px"])
            if pixel_length < 10:
                raise ScaleExtractionError(f"标尺线过短: {pixel_length:.1f} px")

            scale_ratio = physical_nm / pixel_length
            if scale_ratio <= 0 or scale_ratio > 1000:
                raise ScaleExtractionError(f"计算出的比例尺异常: {scale_ratio:.6f} nm/px")

            confidence = _estimate_confidence(matched_text, pixel_length, ocr_raw)
            calculation = f"{physical_nm:g} / {pixel_length:.2f} = {scale_ratio:.6f}"

            attempt["ok"] = True
            attempt["scale_ratio"] = scale_ratio
            attempt["confidence"] = confidence
            attempt["calculation"] = calculation
            debug_log["regions_tried"].append(attempt)
            debug_log["selected_strategy"] = region_spec["strategy"]
            debug_log["calculation"] = calculation

            result = {
                "scale_ratio": round(float(scale_ratio), 6),
                "physical_length_nm": round(float(physical_nm), 3),
                "pixel_length": round(float(pixel_length), 2),
                "ocr_text": matched_text,
                "ocr_raw_text": (ocr_raw or "").strip(),
                "confidence": confidence,
                "source": "ocr",
                "region": {
                    "x": int(region_spec["x"]),
                    "y": int(region_spec["y"]),
                    "w": int(region_spec["w"]),
                    "h": int(region_spec["h"]),
                    "strategy": region_spec["strategy"],
                },
                "bar": {
                    "x1": int(bar_global["x1"]),
                    "y1": int(bar_global["y1"]),
                    "x2": int(bar_global["x2"]),
                    "y2": int(bar_global["y2"]),
                    "length_px": round(float(pixel_length), 2),
                },
            }
            if debug:
                result["debug"] = debug_log
            return result

        except ScaleExtractionError as e:
            attempt["error"] = str(e)
            debug_log["regions_tried"].append(attempt)
            last_error = str(e)
            if attempt.get("physical_length_nm") and (
                best_partial is None
                or (attempt.get("bar") and not best_partial.get("bar"))
            ):
                best_partial = attempt
            continue

    detail = last_error
    if best_partial and best_partial.get("ocr_matched"):
        detail = (
            f"{last_error}；已识别到文字 '{best_partial['ocr_matched']}'，"
            f"但未能可靠测量标尺线像素长度"
        )
    raise ScaleExtractionError(detail, debug=debug_log if debug else None)


def extract_scale_ratio(image_matrix: np.ndarray) -> float:
    """兼容旧接口：只返回 nm/px。"""
    return float(extract_scale_info(image_matrix, debug=False)["scale_ratio"])


def validate_scale_ratio(
    scale_ratio: float,
    image_width: int,
    min_diameter_nm: float = 1.0,
    max_diameter_nm: float = 5000.0,
) -> bool:
    """粗略校验比例尺是否落在常见粒径范围内。"""
    if image_width <= 0:
        return False
    min_valid_ratio = min_diameter_nm / image_width
    max_valid_ratio = max_diameter_nm / image_width
    return min_valid_ratio <= scale_ratio <= max_valid_ratio


# ---------------------------------------------------------------------------
# 内部实现
# ---------------------------------------------------------------------------

def _decode_image(image_input: Union[np.ndarray, bytes]) -> np.ndarray:
    if isinstance(image_input, (bytes, bytearray)):
        nparr = np.frombuffer(image_input, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ScaleExtractionError("无法解码图像数据")
        return image
    if image_input is None or getattr(image_input, "size", 0) == 0:
        raise ScaleExtractionError("无效图像输入")
    return image_input


def _iter_search_regions(height: int, width: int) -> List[Dict[str, Any]]:
    """
    按优先级生成比例尺搜索区域（全图坐标）。
    默认优先底部 20%；失败后再扩大范围与角落。
    """
    regions: List[Dict[str, Any]] = []

    def add_bottom(ratio: float, name: str) -> None:
        h = max(8, int(height * ratio))
        regions.append({
            "strategy": name,
            "x": 0,
            "y": height - h,
            "w": width,
            "h": h,
        })

    add_bottom(0.20, "bottom_20")
    add_bottom(0.30, "bottom_30")
    add_bottom(0.40, "bottom_40")

    # 底部左右角（部分 SEM 标尺在角落）
    corner_h = max(8, int(height * 0.22))
    corner_w = max(8, int(width * 0.45))
    regions.append({
        "strategy": "bottom_left_corner",
        "x": 0,
        "y": height - corner_h,
        "w": corner_w,
        "h": corner_h,
    })
    regions.append({
        "strategy": "bottom_right_corner",
        "x": width - corner_w,
        "y": height - corner_h,
        "w": corner_w,
        "h": corner_h,
    })

    return regions


def _ocr_scale_label(region: np.ndarray) -> Tuple[float, str, str]:
    """
    仅识别标尺文字（物理长度 + 单位），不输出 nm/px。

    Returns:
        (physical_length_nm, matched_text, raw_ocr_text)
    """
    try:
        import pytesseract
    except ImportError:
        raise ScaleExtractionError("请安装 pytesseract: pip install pytesseract")

    variants = _build_ocr_variants(region)
    configs = [
        "--psm 6 --oem 3",   # 假设块状文本
        "--psm 7 --oem 3",   # 单行
        "--psm 11 --oem 3",  # 稀疏文本
    ]

    best: Optional[Tuple[float, str, str, int]] = None
    all_raw: List[str] = []

    for gray in variants:
        for cfg in configs:
            try:
                text = pytesseract.image_to_string(gray, config=cfg) or ""
            except Exception:
                continue
            text = text.strip()
            if text:
                all_raw.append(text)
            parsed = _parse_scale_label(text)
            if parsed is None:
                continue
            physical_nm, matched, score = parsed
            if best is None or score > best[3]:
                best = (physical_nm, matched, text, score)

    if best is None:
        joined = " | ".join(all_raw[:3]) if all_raw else "(empty)"
        raise ScaleExtractionError(
            f"OCR未能识别有效标尺文字（期望如 '100 nm' / '1 μm'）。原始内容: '{joined}'"
        )

    physical_nm, matched, raw, _ = best
    return physical_nm, matched, raw


def _build_ocr_variants(region: np.ndarray) -> List[np.ndarray]:
    """生成若干增强灰度图，提高底部小字 OCR 成功率。"""
    if region.ndim == 3:
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    else:
        gray = region.copy()

    # 放大：底部状态栏文字通常很小
    h, w = gray.shape[:2]
    scale = 3 if max(h, w) < 800 else 2
    up = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    up = cv2.GaussianBlur(up, (3, 3), 0)
    eq = cv2.equalizeHist(up)

    _, otsu = cv2.threshold(eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_inv = cv2.bitwise_not(otsu)
    adap = cv2.adaptiveThreshold(
        eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 8
    )

    # 去重：按内容哈希简单去重
    variants: List[np.ndarray] = []
    seen = set()
    for img in (up, eq, otsu, otsu_inv, adap):
        key = hash(img.tobytes()[:: max(1, img.size // 5000)])
        if key in seen:
            continue
        seen.add(key)
        variants.append(img)
    return variants


def _parse_scale_label(text: str) -> Optional[Tuple[float, str, int]]:
    """
    从 OCR 文本解析标尺标注。

    Returns:
        (physical_nm, matched_text, score) 或 None
    """
    if not text:
        return None

    # 规范化常见 OCR 误识
    normalized = (
        text.replace("µ", "μ")
        .replace("u m", "um")
        .replace("n m", "nm")
        .replace("1Jm", "1μm")
        .replace("1 jm", "1 μm")
    )

    candidates: List[Tuple[float, str, int]] = []
    for pattern, unit_to_nm in _SCALE_PATTERNS:
        for match in pattern.finditer(normalized):
            value = float(match.group(1))
            if value <= 0:
                continue
            physical_nm = value * unit_to_nm
            # 常见 SEM 标尺：1~10000 nm 量级更可信
            score = 10
            if 1 <= physical_nm <= 10000:
                score += 5
            if re.search(r"\b(nm|μm|um)\b", match.group(0), re.IGNORECASE):
                score += 3
            # 偏好整数常见刻度
            if value in {1, 2, 5, 10, 20, 50, 100, 200, 500, 1000}:
                score += 4
            candidates.append((physical_nm, match.group(0).strip(), score))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates[0]


def _detect_scale_bar(region: np.ndarray) -> Optional[Dict[str, float]]:
    """
    在比例尺区域内检测水平标尺线。

    Returns:
        {"x1","y1","x2","y2","length_px"} 相对 region 左上角；失败返回 None。
        绝不使用“按宽度比例估算”的伪长度。
    """
    if region.ndim == 3:
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    else:
        gray = region

    h, w = gray.shape[:2]
    candidates: List[Dict[str, float]] = []

    # 方案 A：轮廓法（亮线 / 暗线各试一次）
    for invert in (False, True):
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if invert:
            binary = cv2.bitwise_not(binary)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, w // 40), 2))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, bw, bh = cv2.boundingRect(contour)
            if bw < max(20, int(w * 0.03)):
                continue
            if bh <= 0:
                continue
            aspect = bw / bh
            if aspect < 4:
                continue
            # 用该行上的连续前景像素精修长度
            refined = _refine_horizontal_run(binary, y, bh, x, bw)
            length = refined["length_px"] if refined else float(bw)
            if length < 20:
                continue
            candidates.append({
                "x1": float(refined["x1"] if refined else x),
                "y1": float(y + bh / 2.0),
                "x2": float(refined["x2"] if refined else x + bw),
                "y2": float(y + bh / 2.0),
                "length_px": float(length),
                "score": float(length * min(aspect, 50)),
            })

    # 方案 B：逐行扫描最长水平亮/暗段
    scanned = _scan_longest_horizontal_run(gray)
    if scanned is not None:
        scanned["score"] = scanned["length_px"] * 8.0
        candidates.append(scanned)

    if not candidates:
        return None

    candidates.sort(key=lambda c: c["score"], reverse=True)
    best = candidates[0]
    return {
        "x1": best["x1"],
        "y1": best["y1"],
        "x2": best["x2"],
        "y2": best["y2"],
        "length_px": best["length_px"],
    }


def _refine_horizontal_run(
    binary: np.ndarray,
    y: int,
    bh: int,
    x: int,
    bw: int,
) -> Optional[Dict[str, float]]:
    """在包围盒附近取中位行，测量最长连续前景段。"""
    h, w = binary.shape[:2]
    y0 = max(0, y)
    y1 = min(h, y + max(bh, 1))
    row = binary[y0:y1, :].max(axis=0)
    return _longest_run_in_row(row, min_val=127)


def _scan_longest_horizontal_run(gray: np.ndarray) -> Optional[Dict[str, float]]:
    """扫描灰度图每一行，找最长连续高对比水平段。"""
    h, w = gray.shape[:2]
    # 只扫中下部，SEM 标尺通常靠近文字下方
    y_start = int(h * 0.35)
    best: Optional[Dict[str, float]] = None

    for invert in (False, True):
        work = 255 - gray if invert else gray
        # 相对阈值：偏亮像素
        thr = max(40, int(np.percentile(work, 75)))
        for yi in range(y_start, h):
            row = work[yi]
            mask = (row >= thr).astype(np.uint8) * 255
            run = _longest_run_in_row(mask, min_val=127)
            if run is None:
                continue
            if run["length_px"] < max(20, int(w * 0.04)):
                continue
            if best is None or run["length_px"] > best["length_px"]:
                best = {
                    "x1": run["x1"],
                    "y1": float(yi),
                    "x2": run["x2"],
                    "y2": float(yi),
                    "length_px": run["length_px"],
                }
    return best


def _longest_run_in_row(row: np.ndarray, min_val: int = 127) -> Optional[Dict[str, float]]:
    """一维数组上找最长连续超阈值区间。"""
    best_len = 0
    best_start = 0
    cur_len = 0
    cur_start = 0
    for i, v in enumerate(row.tolist()):
        if v >= min_val:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_len = cur_len
                best_start = cur_start
        else:
            cur_len = 0
    if best_len < 10:
        return None
    return {
        "x1": float(best_start),
        "x2": float(best_start + best_len - 1),
        "length_px": float(best_len),
    }


def _estimate_confidence(matched_text: str, pixel_length: float, ocr_raw: str) -> str:
    text = matched_text or ""
    if not text or not re.search(r"\d", text):
        return "low"
    unit_ok = bool(re.search(r"(nm|μm|um|micron)", text, re.IGNORECASE))
    if unit_ok and pixel_length >= 50 and len((ocr_raw or "").strip()) > 0:
        return "high"
    if unit_ok and pixel_length >= 20:
        return "medium"
    return "low"
