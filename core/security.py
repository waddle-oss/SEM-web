"""
安全核心模块
基于 cryptography.fernet 的 AES-128-CBC 加解密实现
"""

from cryptography.fernet import Fernet
import base64


# =========================================================
# Fernet 密钥
# 格式要求：必须是可以被 Fernet 正确解码的 Base64 字符串
# 长度：32 字节的 URL-safe Base64 编码
# =========================================================
# 使用 Fernet.generate_key() 生成的示例密钥：
SECRET_KEY = b'LrVHm8k9_QmXZdD9Kp3nY2xJw5vE7cR4sUo0bFtG8yI='


# 初始化 Fernet 实例
_fernet = Fernet(SECRET_KEY)


def encrypt_data(data: bytes) -> str:
    """
    加密数据，返回Base64编码的密文字符串

    Args:
        data: 要加密的原始字节流

    Returns:
        Base64编码的加密字符串

    Raises:
        TypeError: 当数据类型不是bytes时抛出
    """
    if not isinstance(data, bytes):
        raise TypeError(f"encrypt_data 需要 bytes 类型，输入为: {type(data)}")

    encrypted_bytes = _fernet.encrypt(data)
    return base64.b64encode(encrypted_bytes).decode("utf-8")


def decrypt_data(encrypted_str: str) -> bytes:
    """
    解密数据，返回原始字节流

    Args:
        encrypted_str: Base64编码的加密字符串

    Returns:
        解密后的原始字节流

    Raises:
        ValueError: 当解密失败时抛出
    """
    if not isinstance(encrypted_str, (str, bytes)):
        raise TypeError(f"decrypt_data 需要 str 或 bytes 类型，输入为: {type(encrypted_str)}")

    try:
        # 如果是字符串，先编码为bytes
        if isinstance(encrypted_str, str):
            encrypted_str = encrypted_str.encode("utf-8")

        # Base64解码
        decoded_bytes = base64.b64decode(encrypted_str)

        # Fernet解密
        decrypted_bytes = _fernet.decrypt(decoded_bytes)

        return decrypted_bytes

    except Exception as e:
        raise ValueError(f"解密失败: {str(e)}")


def generate_new_key() -> bytes:
    """
    生成新的Fernet密钥

    Returns:
        新的Fernet密钥（bytes类型）
    """
    return Fernet.generate_key()
