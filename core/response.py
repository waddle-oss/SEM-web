"""
统一响应模块
所有接口的成功响应、错误响应使用统一格式
"""

from typing import Any, Optional, Dict
from fastapi.responses import JSONResponse


# 标准错误码
ERROR_CODES = {
    "MODEL_NOT_LOADED",
    "INVALID_IMAGE_FORMAT",
    "IMAGE_TOO_LARGE",
    "OCR_FAILED",
    "INVALID_SCALE_RATIO",
    "ANALYSIS_FAILED",
    "DECRYPT_FAILED",
    "INVALID_REQUEST",
    "PERMISSION_DENIED",
    "INTERNAL_ERROR",
    "FILE_EMPTY",
    "FILE_NOT_FOUND",
    "INVALID_FILTER_PARAMS"
}


def success_response(data: Any = None, message: str = "success", extra: Optional[Dict] = None):
    """
    生成成功响应字典

    Args:
        data: 业务数据
        message: 提示信息
        extra: 顶层额外字段

    Returns:
        dict
    """
    payload = {
        "success": True,
        "message": message,
        "data": data
    }
    if extra:
        payload.update(extra)
    return payload


def error_response(
    code: str,
    message: str,
    detail: Optional[str] = None,
    http_status: int = 400
) -> JSONResponse:
    """
    生成错误响应 JSONResponse

    Args:
        code: 错误码（必须来自 ERROR_CODES）
        message: 用户友好提示
        detail: 详细错误（内部日志用）
        http_status: HTTP 状态码

    Returns:
        JSONResponse
    """
    if code not in ERROR_CODES:
        code = "INTERNAL_ERROR"

    return JSONResponse(
        status_code=http_status,
        content={
            "success": False,
            "code": code,
            "message": message,
            "detail": detail
        }
    )


def safe_error(e: Exception, default_code: str = "INTERNAL_ERROR", default_message: str = "服务器内部错误", http_status: int = 500) -> JSONResponse:
    """
    把异常转换为统一错误响应
    """
    return error_response(
        code=default_code,
        message=default_message,
        detail=str(e),
        http_status=http_status
    )