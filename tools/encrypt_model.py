"""
模型加密工具
将YOLOv8模型权重文件(.pt)加密为.encrypted文件

使用方法:
    python tools/encrypt_model.py                # 加密默认模型 best.pt
    python tools/encrypt_model.py model.pt     # 加密指定模型
    python tools/encrypt_model.py model.pt -o custom.encrypted  # 指定输出文件
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.security import encrypt_data


def encrypt_model(input_path: str, output_path: str = None) -> str:
    """
    加密模型文件

    Args:
        input_path: 原始模型文件路径 (.pt)
        output_path: 加密后输出文件路径 (.encrypted)

    Returns:
        加密文件的路径
    """
    input_file = Path(input_path)

    # 验证输入文件
    if not input_file.exists():
        raise FileNotFoundError(f"模型文件不存在: {input_file.absolute()}")

    # 默认输出路径：同目录下同名.encrypted文件
    if output_path is None:
        output_file = input_file.with_suffix(".encrypted")
    else:
        output_file = Path(output_path)

    # 检查输出文件是否存在
    if output_file.exists():
        response = input(f"输出文件已存在: {output_file}，是否覆盖? (y/n): ")
        if response.lower() != 'y':
            print("操作已取消")
            sys.exit(0)

    # 读取模型文件
    print(f"[*] 读取模型: {input_file.name}")
    with open(input_file, "rb") as f:
        model_bytes = f.read()

    file_size_mb = len(model_bytes) / (1024 * 1024)
    print(f"[*] 模型大小: {file_size_mb:.2f} MB")

    # 加密
    print("[*] 加密中...")
    encrypted_data = encrypt_data(model_bytes)

    encrypted_size_mb = len(encrypted_data) / (1024 * 1024)
    print(f"[*] 加密后大小: {encrypted_size_mb:.2f} MB")

    # 保存加密文件
    print(f"[*] 保存加密文件: {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(encrypted_data)

    print(f"[+] 加密完成!")
    print(f"[+] 输出文件: {output_file.absolute()}")

    return str(output_file)


def main():
    """命令行入口"""
    print()
    print("=" * 50)
    print("  YOLOv8 模型加密工具")
    print("=" * 50)
    print()

    # 解析命令行参数
    input_model = "best.pt"  # 默认模型名
    output_model = None

    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg in ["-h", "--help"]:
            print("用法: python tools/encrypt_model.py [输入模型] [-o 输出模型]")
            print("示例: python tools/encrypt_model.py best.pt")
            print("      python tools/encrypt_model.py yolov8n-seg.pt -o custom.encrypted")
            sys.exit(0)
        elif arg == "-o" and i + 1 < len(args):
            output_model = args[i + 1]
        elif not arg.startswith("-"):
            input_model = arg

    # 执行加密
    try:
        encrypt_model(input_model, output_model)
    except FileNotFoundError as e:
        print(f"[错误] {e}")
        print()
        print("请确保:")
        print("  1. 模型文件存在于当前目录")
        print("  2. 或指定完整路径")
        print()
        print("示例: python tools/encrypt_model.py best.pt")
        sys.exit(1)
    except Exception as e:
        print(f"[错误] 加密失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
