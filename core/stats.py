"""
统计指标增强模块
从 particle_data 中提取 D10 / D50 / D90、平均值、标准差等
"""

import math
from typing import List, Dict, Any, Optional


def percentile(values: List[float], p: float) -> Optional[float]:
    """
    线性插值百分位数

    Args:
        values: 数值列表（无需预先排序）
        p: 百分位 [0, 1]

    Returns:
        百分位数值；空列表返回 None
    """
    if not values:
        return None

    vals = sorted(values)
    if len(vals) == 1:
        return float(vals[0])

    index = (len(vals) - 1) * p
    lower = math.floor(index)
    upper = math.ceil(index)

    if lower == upper:
        return float(vals[lower])

    return float(vals[lower] + (vals[upper] - vals[lower]) * (index - lower))


def build_exclusion_stats(excluded_particles: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    统计排除原因

    排除原因枚举：edge / low_completeness / small_area / large_area / unknown
    """
    base = {
        "edge": 0,
        "low_completeness": 0,
        "small_area": 0,
        "large_area": 0,
        "unknown": 0
    }
    for p in excluded_particles:
        reason = p.get("exclude_reason") or "unknown"
        base[reason] = base.get(reason, 0) + 1
    return base


def calculate_enhanced_stats(
    particle_data: List[Dict[str, Any]],
    excluded_particles: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    从颗粒数据中提取完整统计指标

    Args:
        particle_data: 有效颗粒列表
        excluded_particles: 排除颗粒列表（可为 None）

    Returns:
        包含 valid_count / excluded_count / d10 / d50 / d90 / 平均值 / 标准差等的字典
    """
    excluded_particles = excluded_particles or []

    valid = [
        p for p in particle_data
        if (
            (p.get("diameter_nm") is not None or p.get("avg_diameter_nm") is not None)
            and p.get("status", "valid") == "valid"
        )
    ]
    diameters = []
    for p in valid:
        if isinstance(p.get("avg_diameter_nm"), (int, float)):
            diameters.append(float(p["avg_diameter_nm"]))
        elif isinstance(p.get("diameter_nm"), (int, float)):
            diameters.append(float(p["diameter_nm"]))
        elif isinstance(p.get("avg_radius_nm"), (int, float)):
            diameters.append(float(p["avg_radius_nm"]) * 2.0)

    if not diameters:
        return {
            "valid_count": 0,
            "excluded_count": len(excluded_particles),
            "average_nm": None,
            "d10_nm": None,
            "d50_nm": None,
            "d90_nm": None,
            "std_dev": None,
            "min_nm": None,
            "max_nm": None,
            "exclusion_stats": build_exclusion_stats(excluded_particles)
        }

    avg = sum(diameters) / len(diameters)
    variance = sum((x - avg) ** 2 for x in diameters) / len(diameters)
    std_dev = math.sqrt(variance)

    return {
        "valid_count": len(valid),
        "excluded_count": len(excluded_particles),
        "average_nm": round(avg, 3),
        "d10_nm": round(percentile(diameters, 0.10), 3),
        "d50_nm": round(percentile(diameters, 0.50), 3),
        "d90_nm": round(percentile(diameters, 0.90), 3),
        "std_dev": round(std_dev, 3),
        "min_nm": round(min(diameters), 3),
        "max_nm": round(max(diameters), 3),
        "exclusion_stats": build_exclusion_stats(excluded_particles)
    }


def merge_stats_into_result(base_result: Dict[str, Any], enhanced_stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    把增强统计合并到分析结果中，同时保留旧字段

    Args:
        base_result: analyze_particles 原始返回
        enhanced_stats: calculate_enhanced_stats 返回

    Returns:
        合并后的完整结果
    """
    out = dict(base_result)

    # 旧字段 - 保留向后兼容
    # total_count / average_nm / d50_nm / std_dev / min_nm / max_nm / scale_ratio / scale_source
    if enhanced_stats.get("average_nm") is not None:
        out["average_nm"] = enhanced_stats["average_nm"]
    if enhanced_stats.get("d50_nm") is not None:
        out["d50_nm"] = enhanced_stats["d50_nm"]
    if enhanced_stats.get("std_dev") is not None:
        out["std_dev"] = enhanced_stats["std_dev"]
    if enhanced_stats.get("min_nm") is not None:
        out["min_nm"] = enhanced_stats["min_nm"]
    if enhanced_stats.get("max_nm") is not None:
        out["max_nm"] = enhanced_stats["max_nm"]

    # 新增顶层字段
    out["success"] = True
    out["valid_count"] = enhanced_stats["valid_count"]
    out["excluded_count"] = enhanced_stats["excluded_count"]
    out["d10_nm"] = enhanced_stats["d10_nm"]
    out["d90_nm"] = enhanced_stats["d90_nm"]
    out["exclusion_stats"] = enhanced_stats["exclusion_stats"]
    out["excluded_particles"] = out.get("excluded_particles", [])

    return out