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
    ULTRALYTICS_AVAILABLE
)
import pymysql


# 默认加密模型路径
DEFAULT_ENCRYPTED_MODEL = "best.encrypted"


# =========================================================
# FastAPI 生命周期管理
# =========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理

    启动时:
    - 加载加密模型（阅后即焚机制）

    关闭时:
    - 清理资源
    """
    print("\n" + "=" * 50)
    print("SEM颗粒智能分析系统 - 启动中")
    print("=" * 50)

    # 获取模型路径
    model_path = os.environ.get("ENCRYPTED_MODEL_PATH", DEFAULT_ENCRYPTED_MODEL)

    # 尝试加载加密模型
    if os.path.exists(model_path):
        print(f"[*] 加载加密模型: {model_path}")
        try:
            initialize_model(model_path)
            print("[+] 模型加载成功")
        except Exception as e:
            print(f"[-] 模型加载失败: {e}")
            print("    系统将以模拟模式运行")
    else:
        print(f"[-] 加密模型不存在: {model_path}")
        print("    系统将以模拟模式运行")

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
    description="工业级加密版 - 所有数据传输使用AES加密",
    version="2.0.0",
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
    scale_ratio: float = Form(None, description="1像素对应的纳米数（可选，不提供则自动OCR提取）")
):
    """
    安全分析接口

    完整流程:
    1. 接收加密的图片数据
    2. 使用AES解密图片
    3. 调用分析算法处理图片（自动提取比例尺）
    4. 将分析结果JSON加密
    5. 返回加密结果

    Args:
        file: 加密后的图片文件
        scale_ratio: 标尺比例（1像素 = X纳米），可选
                      - 如果提供：使用指定值
                      - 如果不提供：自动从图片OCR提取

    Returns:
        {"encrypted_result": "Base64加密的分析结果"}
    """
    # Step 1: 验证参数（scale_ratio允许为空或0，会自动提取）
    if scale_ratio is not None and scale_ratio <= 0:
        return _error_response("scale_ratio必须大于0")

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

    try:
        result = analyze_particles(
            image_bytes=decrypted_image,
            scale_ratio=scale_ratio,
            use_mock=use_mock
        )
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


@app.post("/api/analyze_plain")
async def analyze_plain(
    file: UploadFile = File(...),
    scale_ratio: float = Form(None, description="标尺比例（可选）")
):
    """
    明文分析接口（仅用于调试）

    Args:
        file: 原始图片文件（不加密）
        scale_ratio: 标尺比例，可选

    Returns:
        明文分析结果JSON
    """
    if scale_ratio is not None and scale_ratio <= 0:
        raise HTTPException(status_code=400, detail="scale_ratio必须大于0")

    try:
        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="文件为空")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    model = get_model()
    use_mock = model is None

    try:
        result = analyze_particles(
            image_bytes=file_bytes,
            scale_ratio=scale_ratio,
            use_mock=use_mock
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
