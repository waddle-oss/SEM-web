"""
SEM颗粒智能分析安全接口 - FastAPI服务入口
工业级加密版本 V2.0

功能:
- POST /analyze_secure: 接收加密图片，返回加密结果
- 所有数据传输使用AES加密
- YOLOv8模型磁盘加密存储
- OCR自动比例尺提取（可选手动指定）
"""

import uvicorn
import json
import os
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from core.security import encrypt_data, decrypt_data
from core.analyzer import (
    analyze_particles,
    initialize_model,
    get_model,
    get_model_path,
    ULTRALYTICS_AVAILABLE
)
from core.response import success_response, error_response, safe_error
from core.validation import validate_image_file, get_image_format, validate_filter_params, normalize_filter_params
from core.scale_reader import ScaleExtractionError, extract_scale_info
import pymysql


# =========================================================
# 应用配置 (V2.1.1)
# =========================================================
APP_VERSION = "2.1.1"
APP_EDITION = "Parameter Real Integration Edition"
PLAIN_API_ENABLED = True
DEFAULT_ENCRYPTED_MODEL = "best.encrypted"

# 本地实验模型根目录 & 默认实验（优先于 best.encrypted）
LOCAL_MODEL_ROOT = Path(__file__).resolve().parent / "runs" / "segment"
DEFAULT_MODEL_NAME = "exp_10"

# 当前加载的模型名称（用于状态追踪，如 exp_10 / best.encrypted）
_current_model_name: Optional[str] = None


def check_ocr_available():
    """检测 Tesseract OCR 是否可用"""
    try:
        import pytesseract
        version = pytesseract.get_tesseract_version()
        return True, str(version)
    except Exception:
        return False, None


def detect_device():
    """检测运行设备（cpu / cuda / unknown）"""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "unknown"


def _exp_sort_key(exp_dir: Path):
    """自然排序：exp_7 < exp_9 < exp_10"""
    try:
        return int(exp_dir.name.split("_", 1)[1])
    except (IndexError, ValueError):
        return exp_dir.name


def find_weights_file(exp_dir: Path) -> Optional[Path]:
    """在实验目录的 weights/ 下查找 best.pt 或 best.onnx（优先 .pt）"""
    weights_dir = exp_dir / "weights"
    if not weights_dir.exists():
        return None
    for ext in [".pt", ".onnx"]:
        candidate = weights_dir / f"best{ext}"
        if candidate.exists():
            return candidate
    return None


def resolve_startup_model() -> tuple:
    """
    解析启动时默认模型

    优先级:
    1. 环境变量 ENCRYPTED_MODEL_PATH（显式指定）
    2. runs/segment/exp_10/weights/best.{pt,onnx}
    3. 回退 best.encrypted

    Returns:
        (model_name, model_path) 或 (None, None)
    """
    env_path = os.environ.get("ENCRYPTED_MODEL_PATH")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return (p.stem if p.suffix == ".encrypted" else p.parent.parent.name, str(p))
        print(f"[-] 环境变量指定的模型不存在: {env_path}")

    default_exp = LOCAL_MODEL_ROOT / DEFAULT_MODEL_NAME
    weights = find_weights_file(default_exp)
    if weights is not None:
        return (DEFAULT_MODEL_NAME, str(weights))

    fallback = Path(DEFAULT_ENCRYPTED_MODEL)
    if fallback.exists():
        return ("best.encrypted", str(fallback.resolve()))

    return (None, None)


# =========================================================
# FastAPI 生命周期管理
# =========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理

    启动时:
    - 默认加载 exp_10 权重（可被环境变量覆盖）
    - 失败时回退 best.encrypted / 模拟模式

    关闭时:
    - 清理资源
    """
    global _current_model_name

    print("\n" + "=" * 50)
    print("SEM颗粒智能分析系统 - 启动中")
    print("=" * 50)

    model_name, model_path = resolve_startup_model()

    if model_path:
        print(f"[*] 加载默认模型: {model_name} -> {model_path}")
        try:
            initialize_model(model_path, force=True)
            _current_model_name = model_name
            print(f"[+] 模型加载成功: {model_name}")
        except Exception as e:
            print(f"[-] 模型加载失败: {e}")
            print("    系统将以模拟模式运行")
            _current_model_name = None
    else:
        print(f"[-] 未找到可用模型（默认 {DEFAULT_MODEL_NAME} 或 {DEFAULT_ENCRYPTED_MODEL}）")
        print("    系统将以模拟模式运行")
        _current_model_name = None

    print("=" * 50 + "\n")

    yield

    # 关闭时清理
    model = get_model()
    if model is not None:
        print("[*] 模型资源已释放")
    print("[*] SEM颗粒智能分析系统 - 已停止\n")


# 创建 FastAPI 应用
app = FastAPI(
    title="SEM颗粒智能分析安全接口",
    description="V2.1 工业级加密版 - 增强统计、统一响应、状态可观测",
    version=APP_VERSION,
    lifespan=lifespan
)

# 允许前端跨域访问（开发环境）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# 健康检查接口
# =========================================================

@app.get("/")
async def root():
    """根路径 - 服务状态"""
    model = get_model()
    return {
        "service": "SEM颗粒智能分析安全接口",
        "version": "2.0.0",
        "status": "running",
        "model_loaded": model is not None,
        "mode": "secure" if model is not None else "mock"
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    model = get_model()
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "ultralytics_available": ULTRALYTICS_AVAILABLE
    }

# =========================================================
# 模型管理接口（多模型选择）
# =========================================================

@app.get("/api/models/list")
async def list_models():
    """
    扫描本地模型目录，返回可用模型列表
    前端下拉框使用；标记默认模型与当前已加载模型
    """
    try:
        models = []
        model_root = Path(LOCAL_MODEL_ROOT)

        if not model_root.exists():
            return {
                "success": False,
                "message": f"模型目录不存在: {LOCAL_MODEL_ROOT}",
                "data": [],
                "default_model": DEFAULT_MODEL_NAME,
                "current_model": _current_model_name
            }

        for exp_dir in sorted(model_root.glob("exp_*"), key=_exp_sort_key):
            model_file = find_weights_file(exp_dir)
            if model_file is None:
                continue
            models.append({
                "name": exp_dir.name,
                "path": str(model_file),
                "size": model_file.stat().st_size,
                "format": model_file.suffix[1:],
                "modified": model_file.stat().st_mtime,
                "is_default": exp_dir.name == DEFAULT_MODEL_NAME,
                "is_current": exp_dir.name == _current_model_name
            })

        return {
            "success": True,
            "data": models,
            "count": len(models),
            "default_model": DEFAULT_MODEL_NAME,
            "current_model": _current_model_name
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"扫描模型目录失败: {str(e)}",
            "data": [],
            "default_model": DEFAULT_MODEL_NAME,
            "current_model": _current_model_name
        }


@app.post("/api/models/load")
async def load_model(request: dict):
    """
    加载指定的模型（支持前端自由切换）
    请求体: {"model_name": "exp_10"}
    """
    global _current_model_name

    model_name = request.get("model_name")

    if not model_name:
        return {"success": False, "message": "请指定模型名称"}

    try:
        model_root = Path(LOCAL_MODEL_ROOT)
        exp_dir = model_root / model_name

        if not exp_dir.exists() or not exp_dir.is_dir():
            return {
                "success": False,
                "message": f"模型目录不存在: {exp_dir}"
            }

        model_file = find_weights_file(exp_dir)
        if model_file is None:
            return {
                "success": False,
                "message": f"模型文件不存在: {exp_dir}/weights/best.pt 或 best.onnx"
            }

        # force=True 允许从已加载模型切换到另一个
        initialize_model(str(model_file), force=True)
        _current_model_name = model_name

        return {
            "success": True,
            "data": {
                "model_name": model_name,
                "model_path": str(model_file),
                "format": model_file.suffix[1:],
                "is_default": model_name == DEFAULT_MODEL_NAME
            },
            "message": f"模型 {model_name} 加载成功"
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"模型加载失败: {str(e)}"
        }

# =========================================================
# V2.1 系统状态接口
# =========================================================

@app.get("/api/system/status")
async def system_status():
    """
    V2.1 系统状态接口

    供前端顶部状态栏读取，包含：
    - 后端版本
    - 模型加载状态、类型、设备
    - OCR 可用性
    - 运行模式、明文接口开关
    """
    model = get_model()
    ocr_ok, ocr_ver = check_ocr_available()
    device = detect_device()

    current_path = get_model_path()
    encrypted_fallback = os.environ.get("ENCRYPTED_MODEL_PATH", DEFAULT_ENCRYPTED_MODEL)
    display_path = current_path or encrypted_fallback
    model_exists = os.path.exists(display_path) if display_path else False
    is_encrypted = bool(display_path and str(display_path).lower().endswith(".encrypted"))

    data = {
        "version": APP_VERSION,
        "backend_edition": APP_EDITION,
        "service": "SEM Particle Intelligent Analysis System",
        "model": {
            "loaded": model is not None,
            "name": _current_model_name,
            "type": "YOLOv8n-seg" if model is not None else None,
            "path": display_path,
            "encrypted": is_encrypted,
            "exists_on_disk": model_exists,
            "device": device,
            "default_name": DEFAULT_MODEL_NAME
        },
        "ocr": {
            "available": ocr_ok,
            "engine": "tesseract" if ocr_ok else None,
            "version": ocr_ver,
            "status": "ready" if ocr_ok else "unavailable"
        },
        "runtime": {
            "mode": "development",
            "mock_enabled": True,           # 当前代码层面 mock 可用
            "plain_api_enabled": PLAIN_API_ENABLED,
            "ultralytics_available": ULTRALYTICS_AVAILABLE
        }
    }
    return success_response(data=data, message="系统运行正常")


# =========================================================
# V2.1 OCR 比例尺预识别接口
# =========================================================

@app.post("/api/scale/extract")
async def extract_scale_endpoint(file: UploadFile = File(...)):
    """
    V2.1 比例尺预识别接口

    流程：
    1. 上传图像
    2. 校验文件
    3. 调用 extract_scale_info 返回结构化结果
    4. 前端展示识别值，用户点击确认后再调用 analyze 接口
    """
    try:
        image_bytes = await file.read()

        # 文件校验
        ok, err = validate_image_file(file.filename, image_bytes)
        if not ok:
            return error_response(
                code="INVALID_IMAGE_FORMAT" if "格式" in err else "FILE_EMPTY",
                message=err,
                http_status=400
            )

        # 调用 OCR
        scale_info = extract_scale_info(image_bytes)
        return success_response(
            data=scale_info,
            message="比例尺识别成功，请确认后开始分析"
        )
    except ScaleExtractionError as e:
        return error_response(
            code="OCR_FAILED",
            message="OCR 未能识别比例尺，请手动输入 nm/px",
            detail=str(e),
            http_status=400
        )
    except Exception as e:
        return safe_error(
            e,
            default_code="OCR_FAILED",
            default_message="OCR 识别失败，请手动输入比例尺",
            http_status=500
        )

# =========================================================
# 登录接口
# =========================================================

@app.post("/api/login")
async def login(request: dict):
    """
    用户登录验证
    接收 {"username": "xxx", "password": "xxx"}
    """
    username = request.get("username", "").strip()
    password = request.get("password", "")
    
    if not username or not password:
        return {"success": False, "message": "用户名和密码不能为空"}
    
    try:
        conn = pymysql.connect(
            host="110.42.196.53",
            port=3306,
            user="root",
            password="4ce100f909a8e4f3",
            database="sem_users",
            charset="utf8mb4"
        )
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        cursor.execute("SELECT username, password, role FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if user and user["password"] == password:
            return {
                "success": True,
                "username": user["username"],
                "role": user["role"]
            }
        else:
            return {"success": False, "message": "用户名或密码错误"}
    except Exception as e:
        return {"success": False, "message": f"服务器错误: {str(e)}"}


# =========================================================
# 注册接口
# =========================================================

@app.post("/api/register")
async def register(request: dict):
    """
    用户注册
    接收 {"username": "xxx", "password": "xxx", "invite_code": "xxx"}
    """
    username = request.get("username", "").strip()
    password = request.get("password", "")
    invite_code = request.get("invite_code", "").strip()
    
    if not username or not password:
        return {"success": False, "message": "用户名和密码不能为空"}
    if not invite_code:
        return {"success": False, "message": "邀请码不能为空"}
    if len(username) < 3:
        return {"success": False, "message": "用户名至少3个字符"}
    if len(password) < 6:
        return {"success": False, "message": "密码至少6个字符"}
    
    try:
        conn = pymysql.connect(
            host="110.42.196.53",
            port=3306,
            user="root",
            password="4ce100f909a8e4f3",
            database="sem_users",
            charset="utf8mb4"
        )
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        
        # 验证邀请码
        cursor.execute("SELECT * FROM invite_codes WHERE code = %s AND used_by IS NULL", (invite_code,))
        code_record = cursor.fetchone()
        if not code_record:
            cursor.close()
            conn.close()
            return {"success": False, "message": "邀请码无效或已被使用"}
        
        # 检查用户名是否已存在
        cursor.execute("SELECT id FROM users WHERE username = %s", (username,))
        if cursor.fetchone():
            cursor.close()
            conn.close()
            return {"success": False, "message": "用户名已存在"}
        
        # 注册用户
        cursor.execute("INSERT INTO users (username, password, role) VALUES (%s, %s, 'user')", (username, password))
        
        # 标记邀请码已使用
        cursor.execute("UPDATE invite_codes SET used_by = %s WHERE code = %s", (username, invite_code))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return {"success": True, "message": "注册成功"}
    except Exception as e:
        return {"success": False, "message": f"服务器错误: {str(e)}"}

# =========================================================
# 管理员接口 - 获取用户列表（分页）
# =========================================================

@app.post("/api/admin/users")
async def get_users(request: dict):
    """
    管理员获取用户列表（分页）
    请求体: {"username": "admin", "page": 1, "limit": 5}
    """
    username = request.get("username", "").strip()
    page = request.get("page", 1)
    limit = request.get("limit", 5)
    
    if not username:
        return {"success": False, "message": "用户名不能为空"}
    
    try:
        page = int(page)
        limit = int(limit)
        if page < 1:
            page = 1
        if limit < 1 or limit > 100:
            limit = 5
    except:
        page = 1
        limit = 5
    
    offset = (page - 1) * limit
    
    try:
        conn = pymysql.connect(
            host="110.42.196.53",
            port=3306,
            user="root",
            password="4ce100f909a8e4f3",
            database="sem_users",
            charset="utf8mb4"
        )
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        
        # 验证当前用户是否为管理员
        cursor.execute("SELECT role FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()
        
        if not user:
            cursor.close()
            conn.close()
            return {"success": False, "message": "用户不存在"}
        
        if user["role"] != "admin":
            cursor.close()
            conn.close()
            return {"success": False, "message": "权限不足，需要管理员身份"}
        
        # 查询总记录数
        cursor.execute("SELECT COUNT(*) as total FROM users")
        total = cursor.fetchone()["total"]
        
        # 查询当前页数据
        cursor.execute(
            "SELECT id, username, role, created_at FROM users ORDER BY id LIMIT %s OFFSET %s",
            (limit, offset)
        )
        users = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        return {
            "success": True,
            "data": users,
            "total": total,
            "page": page,
            "limit": limit,
            "total_pages": (total + limit - 1) // limit
        }
    except Exception as e:
        return {"success": False, "message": f"服务器错误: {str(e)}"}

# =========================================================
# 管理员接口 - 生成邀请码
# =========================================================

@app.post("/api/admin/generate_invite")
async def generate_invite_code(request: dict):
    """
    管理员生成邀请码
    请求体: {"username": "admin", "count": 1}
    """
    username = request.get("username", "").strip()
    count = request.get("count", 1)
    
    if not username:
        return {"success": False, "message": "用户名不能为空"}
    
    try:
        count = int(count)
        if count < 1:
            count = 1
        if count > 10:
            count = 10  # 一次最多生成10个
    except:
        count = 1
    
    try:
        conn = pymysql.connect(
            host="110.42.196.53",
            port=3306,
            user="root",
            password="4ce100f909a8e4f3",
            database="sem_users",
            charset="utf8mb4"
        )
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        
        # 验证是否为管理员
        cursor.execute("SELECT role FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()
        if not user or user["role"] != "admin":
            cursor.close()
            conn.close()
            return {"success": False, "message": "权限不足，需要管理员身份"}
        
        import random
        import string
        
        generated_codes = []
        for _ in range(count):
            # 生成6-12位随机字母+数字
            length = random.randint(6, 12)
            code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))
            
            # 检查是否已存在（极小概率冲突，但以防万一）
            cursor.execute("SELECT id FROM invite_codes WHERE code = %s", (code,))
            if cursor.fetchone():
                # 如果冲突，重新生成一次
                code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))
            
            # 插入数据库
            cursor.execute("INSERT INTO invite_codes (code, used_by) VALUES (%s, NULL)", (code,))
            generated_codes.append(code)
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return {
            "success": True,
            "message": f"成功生成 {len(generated_codes)} 个邀请码",
            "codes": generated_codes
        }
    except Exception as e:
        return {"success": False, "message": f"服务器错误: {str(e)}"}


# =========================================================
# 管理员接口 - 获取所有邀请码列表（分页）
# =========================================================

@app.post("/api/admin/invite_codes")
async def get_invite_codes(request: dict):
    """
    管理员获取邀请码列表（分页）
    请求体: {"username": "admin", "page": 1, "limit": 10}
    """
    username = request.get("username", "").strip()
    page = request.get("page", 1)
    limit = request.get("limit", 10)
    
    if not username:
        return {"success": False, "message": "用户名不能为空"}
    
    try:
        page = int(page)
        limit = int(limit)
        if page < 1:
            page = 1
        if limit < 1 or limit > 50:
            limit = 10
    except:
        page = 1
        limit = 10
    
    offset = (page - 1) * limit
    
    try:
        conn = pymysql.connect(
            host="110.42.196.53",
            port=3306,
            user="root",
            password="4ce100f909a8e4f3",
            database="sem_users",
            charset="utf8mb4"
        )
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        
        # 验证是否为管理员
        cursor.execute("SELECT role FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()
        if not user or user["role"] != "admin":
            cursor.close()
            conn.close()
            return {"success": False, "message": "权限不足，需要管理员身份"}
        
        # 查询总记录数
        cursor.execute("SELECT COUNT(*) as total FROM invite_codes")
        total = cursor.fetchone()["total"]
        
        # 查询当前页数据
        cursor.execute(
            "SELECT id, code, used_by, created_at FROM invite_codes ORDER BY id DESC LIMIT %s OFFSET %s",
            (limit, offset)
        )
        codes = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        return {
            "success": True,
            "data": codes,
            "total": total,
            "page": page,
            "limit": limit,
            "total_pages": (total + limit - 1) // limit
        }
    except Exception as e:
        return {"success": False, "message": f"服务器错误: {str(e)}"}

# =========================================================
# 核心分析接口
# =========================================================

@app.post("/api/analyze_secure")
async def analyze_secure(
    file: UploadFile = File(..., description="加密后的SEM图像"),
    scale_ratio: float = Form(None, description="1像素对应的纳米数（可选，不提供则自动OCR提取）"),
    scale_source: str = Form("user_confirmed", description="比例尺来源"),
    completeness_threshold: float = Form(0.8, description="完整度阈值 [0,1]"),
    edge_tolerance: int = Form(5, description="边缘容差 px [0,200]"),
    min_area: float = Form(100, description="最小像素面积"),
    max_area: float = Form(None, description="最大像素面积，None 表示不限制"),
    exclude_edge_particles: str = Form("true", description="是否排除边缘颗粒")
):
    """
    安全分析接口（V2.1.1 参数真实接入）

    完整流程:
    1. 接收加密的图片数据
    2. 使用AES解密图片
    3. 文件校验
    4. 调用分析算法处理图片（自动提取比例尺 + 真实过滤参数）
    5. 补充增强字段 (image_info / performance / d10 / d90 / filter_params)
    6. 将增强结果JSON加密
    7. 返回加密结果

    外层结构保持兼容: {"encrypted_result": "..."}
    """
    # Step 1: 验证参数
    if scale_ratio is not None and scale_ratio <= 0:
        return _error_response("scale_ratio必须大于0")

    # 校验过滤参数
    bool_exclude = str(exclude_edge_particles).lower() in ("true", "1", "yes")
    fp_validation = validate_filter_params(
        completeness_threshold, edge_tolerance, min_area, max_area, bool_exclude
    )
    if not fp_validation["valid"]:
        return error_response(
            code="INVALID_FILTER_PARAMS",
            message="过滤参数不合法",
            detail="; ".join(fp_validation["errors"]),
            http_status=400
        )
    fp = normalize_filter_params(
        completeness_threshold, edge_tolerance, min_area, max_area, bool_exclude
    )

    # Step 2: 读取文件
    try:
        file_bytes = await file.read()
        if not file_bytes:
            return _error_response("上传的文件为空")
    except Exception as e:
        return _error_response(f"读取文件失败: {str(e)}")

    # Step 3: 解密图片
    try:
        decrypted_image = decrypt_data(file_bytes)
    except Exception as e:
        return _error_response(f"图片解密失败: {str(e)}")

    # Step 4: 分析图片
    model = get_model()
    use_mock = model is None

    import time
    start_time = time.perf_counter()
    ocr_time_ms = None

    try:
        if scale_ratio is None or scale_ratio <= 0:
            ocr_start = time.perf_counter()
            scale_info = extract_scale_info(decrypted_image)
            scale_ratio = scale_info["scale_ratio"]
            ocr_time_ms = int((time.perf_counter() - ocr_start) * 1000)
            actual_source = "auto_extracted"
        else:
            actual_source = scale_source if scale_source in {"user_confirmed", "ocr_confirmed", "manual", "auto_extracted"} else "user_confirmed"

        inference_start = time.perf_counter()
        result = analyze_particles(
            image_bytes=decrypted_image,
            scale_ratio=scale_ratio,
            use_mock=use_mock,
            completeness_threshold=fp["completeness_threshold"],
            edge_tolerance=fp["edge_tolerance"],
            min_area=fp["min_area"],
            max_area=fp["max_area"],
            exclude_edge_particles=fp["exclude_edge_particles"]
        )
        inference_time_ms = int((time.perf_counter() - inference_start) * 1000)

        total_time_ms = int((time.perf_counter() - start_time) * 1000)

        # 补充 image_info
        try:
            import cv2
            import numpy as np
            nparr = np.frombuffer(decrypted_image, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            img_h, img_w = (img.shape[:2] if img is not None else (None, None))
        except Exception:
            img_h, img_w = None, None

        image_info = {
            "filename": file.filename or "encrypted_upload",
            "width": int(img_w) if img_w else None,
            "height": int(img_h) if img_h else None,
            "format": "encrypted",
            "size_bytes": len(decrypted_image)
        }

        performance = {
            "ocr_time_ms": ocr_time_ms,
            "inference_time_ms": inference_time_ms,
            "postprocess_time_ms": None,
            "total_time_ms": total_time_ms
        }

        # 合并增强字段
        result["success"] = True
        result["scale_source"] = actual_source
        result["image_info"] = image_info
        result["performance"] = performance

        # 补充 d10 / d50 / d90（直径分位数）
        if "d10_nm" not in result or result.get("d10_nm") is None:
            try:
                from core.stats import percentile as _pct
                diameters = []
                for p in (result.get("particle_data") or []):
                    if isinstance(p.get("avg_diameter_nm"), (int, float)):
                        diameters.append(float(p["avg_diameter_nm"]))
                    elif isinstance(p.get("diameter_nm"), (int, float)):
                        diameters.append(float(p["diameter_nm"]))
                    elif isinstance(p.get("avg_radius_nm"), (int, float)):
                        diameters.append(float(p["avg_radius_nm"]) * 2.0)
                diameters = sorted(diameters)
                result["d10_nm"] = round(_pct(diameters, 0.10), 3) if diameters else None
                result["d50_nm"] = result.get("d50_nm") or (round(_pct(diameters, 0.50), 3) if diameters else None)
                result["d90_nm"] = round(_pct(diameters, 0.90), 3) if diameters else None
            except Exception:
                result["d10_nm"] = None
                result["d90_nm"] = None

        if "valid_count" not in result:
            result["valid_count"] = result.get("total_count", 0)
        if "excluded_count" not in result:
            result["excluded_count"] = len(result.get("excluded_particles", []))
        if "total_detected" not in result:
            result["total_detected"] = result["valid_count"] + result["excluded_count"]

        # V2.1.1：保证 filter_params 在加密内容里
        if "filter_params" not in result:
            result["filter_params"] = fp

    except ValueError as e:
        return _error_response(f"图像处理失败: {str(e)}")
    except Exception as e:
        return _error_response(f"分析出错: {str(e)}")

    # Step 5: 加密结果并返回
    try:
        result_json = json.dumps(result, ensure_ascii=False)
        encrypted_result = encrypt_data(result_json.encode("utf-8"))
    except Exception as e:
        return _error_response(f"结果加密失败: {str(e)}")

    return JSONResponse(content={"encrypted_result": encrypted_result})


@app.post("/api/extract_scale")
async def extract_scale(
    file: UploadFile = File(..., description="SEM图像")
):
    """
    仅提取比例尺（OCR独立接口）— 旧路径

    推荐新接口：POST /api/scale/extract（统一响应格式）
    """
    try:
        file_bytes = await file.read()
        if not file_bytes:
            return error_response(code="FILE_EMPTY", message="文件为空", http_status=400)
    except Exception as e:
        return safe_error(e, default_code="INVALID_REQUEST", default_message=f"读取文件失败: {str(e)}", http_status=400)

    try:
        scale_info = extract_scale_info(file_bytes)
        return success_response(data=scale_info, message="OCR识别成功")
    except ScaleExtractionError as e:
        return error_response(
            code="OCR_FAILED",
            message=str(e),
            http_status=400
        )
    except Exception as e:
        return safe_error(e, default_code="OCR_FAILED", default_message="OCR处理失败", http_status=500)


@app.post("/api/analyze_plain")
async def analyze_plain(
    file: UploadFile = File(...),
    scale_ratio: float = Form(None, description="标尺比例（可选）"),
    scale_source: str = Form("user_confirmed", description="比例尺来源"),
    completeness_threshold: float = Form(0.8, description="完整度阈值 [0,1]"),
    edge_tolerance: int = Form(5, description="边缘容差 px [0,200]"),
    min_area: float = Form(100, description="最小像素面积"),
    max_area: float = Form(None, description="最大像素面积，None 表示不限制"),
    exclude_edge_particles: str = Form("true", description="是否排除边缘颗粒")
):
    """
    明文分析接口（V2.1.1）

    V2.1 增强 + V2.1.1 参数真实接入：
    - 接收过滤参数（completeness_threshold / edge_tolerance / min_area / max_area / exclude_edge_particles）
    - 参数校验失败返回 INVALID_FILTER_PARAMS
    - 传入 analyze_particles 参与真实过滤
    - 返回 filter_params 供前端展示与导出追溯
    """
    if not PLAIN_API_ENABLED:
        return error_response(
            code="PERMISSION_DENIED",
            message="当前环境不允许使用明文分析接口",
            http_status=403
        )

    # 校验 scale_ratio
    if scale_ratio is not None and scale_ratio <= 0:
        return error_response(
            code="INVALID_SCALE_RATIO",
            message="scale_ratio 必须大于 0",
            http_status=400
        )

    # 校验过滤参数
    bool_exclude = str(exclude_edge_particles).lower() in ("true", "1", "yes")
    fp_validation = validate_filter_params(
        completeness_threshold, edge_tolerance, min_area, max_area, bool_exclude
    )
    if not fp_validation["valid"]:
        return error_response(
            code="INVALID_FILTER_PARAMS",
            message="过滤参数不合法",
            detail="; ".join(fp_validation["errors"]),
            http_status=400
        )

    # 归一化参数
    fp = normalize_filter_params(
        completeness_threshold, edge_tolerance, min_area, max_area, bool_exclude
    )

    # 读取文件
    try:
        file_bytes = await file.read()
        if not file_bytes:
            return error_response(code="FILE_EMPTY", message="文件为空", http_status=400)
    except Exception as e:
        return safe_error(e, default_code="INVALID_REQUEST", default_message=f"读取文件失败: {str(e)}", http_status=400)

    # 文件格式校验
    ok, err = validate_image_file(file.filename, file_bytes)
    if not ok:
        return error_response(
            code="INVALID_IMAGE_FORMAT" if "格式" in err else "IMAGE_TOO_LARGE" if "大" in err else "FILE_EMPTY",
            message=err,
            http_status=400
        )

    model = get_model()
    use_mock = model is None

    import time
    start_time = time.perf_counter()
    ocr_time_ms = None

    # 如果未提供 scale_ratio，需要 OCR 自动提取（计时）
    if scale_ratio is None or scale_ratio <= 0:
        try:
            ocr_start = time.perf_counter()
            scale_info = extract_scale_info(file_bytes)
            scale_ratio = scale_info["scale_ratio"]
            ocr_time_ms = int((time.perf_counter() - ocr_start) * 1000)
            actual_source = "auto_extracted"
        except ScaleExtractionError as e:
            return error_response(
                code="OCR_FAILED",
                message=str(e),
                http_status=400
            )
        except Exception as e:
            return safe_error(e, default_code="OCR_FAILED", default_message=f"OCR 处理失败: {str(e)}", http_status=500)
    else:
        actual_source = scale_source if scale_source in {"user_confirmed", "ocr_confirmed", "manual", "auto_extracted"} else "user_confirmed"

    # 推理 + 后处理（V2.1.1：传入过滤参数）
    inference_start = time.perf_counter()
    try:
        result = analyze_particles(
            image_bytes=file_bytes,
            scale_ratio=scale_ratio,
            use_mock=use_mock,
            completeness_threshold=fp["completeness_threshold"],
            edge_tolerance=fp["edge_tolerance"],
            min_area=fp["min_area"],
            max_area=fp["max_area"],
            exclude_edge_particles=fp["exclude_edge_particles"]
        )
    except ValueError as e:
        return error_response(code="INVALID_IMAGE_FORMAT", message=str(e), http_status=400)
    except Exception as e:
        return safe_error(e, default_code="ANALYSIS_FAILED", default_message=f"分析失败: {str(e)}", http_status=500)
    inference_time_ms = int((time.perf_counter() - inference_start) * 1000)

    total_time_ms = int((time.perf_counter() - start_time) * 1000)

    # 推断 image_info
    try:
        import cv2
        import numpy as np
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_h, img_w = (img.shape[:2] if img is not None else (None, None))
    except Exception:
        img_h, img_w = None, None

    image_info = {
        "filename": file.filename or "unknown",
        "width": int(img_w) if img_w else None,
        "height": int(img_h) if img_h else None,
        "format": get_image_format(file.filename or ""),
        "size_bytes": len(file_bytes)
    }

    performance = {
        "ocr_time_ms": ocr_time_ms,
        "inference_time_ms": inference_time_ms,
        "postprocess_time_ms": None,
        "total_time_ms": total_time_ms
    }

    # 合并增强字段
    result["success"] = True
    result["scale_source"] = actual_source
    result["image_info"] = image_info
    result["performance"] = performance

    # 补充 d10 / d50 / d90（直径分位数）
    if "d10_nm" not in result or result.get("d10_nm") is None:
        try:
            from core.stats import percentile as _pct
            diameters = []
            for p in (result.get("particle_data") or []):
                if isinstance(p.get("avg_diameter_nm"), (int, float)):
                    diameters.append(float(p["avg_diameter_nm"]))
                elif isinstance(p.get("diameter_nm"), (int, float)):
                    diameters.append(float(p["diameter_nm"]))
                elif isinstance(p.get("avg_radius_nm"), (int, float)):
                    diameters.append(float(p["avg_radius_nm"]) * 2.0)
            diameters = sorted(diameters)
            result["d10_nm"] = round(_pct(diameters, 0.10), 3) if diameters else None
            result["d50_nm"] = result.get("d50_nm") or (round(_pct(diameters, 0.50), 3) if diameters else None)
            result["d90_nm"] = round(_pct(diameters, 0.90), 3) if diameters else None
        except Exception:
            result["d10_nm"] = None
            result["d90_nm"] = None

    # 补 valid_count / excluded_count / total_detected（如 analyzer 未提供）
    if "valid_count" not in result:
        result["valid_count"] = result.get("total_count", 0)
    if "excluded_count" not in result:
        result["excluded_count"] = len(result.get("excluded_particles", []))
    if "total_detected" not in result:
        result["total_detected"] = result["valid_count"] + result["excluded_count"]

    # V2.1.1：保证 filter_params 在顶层
    if "filter_params" not in result:
        result["filter_params"] = fp

    return JSONResponse(content=result)


# =========================================================
# 辅助函数
# =========================================================

def _error_response(message: str, status_code: int = 400) -> JSONResponse:
    """
    返回加密的错误信息

    Args:
        message: 错误信息
        status_code: HTTP状态码

    Returns:
        JSONResponse 包含加密的错误信息
    """
    error_data = {
        "error": True,
        "message": message
    }

    try:
        encrypted_error = encrypt_data(json.dumps(error_data).encode("utf-8"))
        return JSONResponse(
            content={"encrypted_result": encrypted_error},
            status_code=status_code
        )
    except:
        # 如果连错误信息都加密失败，返回明文
        return JSONResponse(
            content={"error": True, "message": message},
            status_code=status_code
        )


# =========================================================
# 启动入口
# =========================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
