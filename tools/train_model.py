"""
YOLOv8 模型训练脚本
针对 SEM 球形颗粒优化的专属训练配置

功能:
- 加载基础模型 yolov8n-seg.pt
- 使用针对球形颗粒的 Transforms 参数进行训练
- 支持命令行参数配置

Transforms 参数说明:
- fliplr=0.5, flipud=0.5: 允许水平和垂直翻转
- degrees=90.0: 允许任意角度旋转
- hsv_v=0.4, hsv_s=0.2: 模拟电镜亮度/对比度变化
- scale=0.5: 允许缩放
- shear=0.0, perspective=0.0: 禁止拉伸和透视（保持正圆）
- mosaic=1.0: 开启马赛克增强
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from ultralytics import YOLO


# =========================================================
# SEM球形颗粒专属 Transforms 参数
# =========================================================

SEM_PARTICLE_TRANSFORMS = {
    # 翻转参数 - 允许水平和垂直翻转
    'fliplr': 0.5,      # 50%概率水平翻转
    'flipud': 0.5,      # 50%概率垂直翻转

    # 旋转参数 - 允许任意角度旋转（球形无方向性）
    'degrees': 90.0,    # 允许 ±90° 旋转

    # 色彩空间参数 - 模拟电镜成像变化
    'hsv_h': 0.0,       # 色调不变（电镜色彩一致）
    'hsv_s': 0.2,       # 饱和度变化 ±20%
    'hsv_v': 0.4,       # 亮度变化 ±40%（电镜曝光差异）

    # 缩放参数
    'scale': 0.5,        # 允许 ±50% 缩放

    # 【关键】禁止拉伸和透视 - 保持正圆形状
    'shear': 0.0,        # 禁止剪切变形
    'perspective': 0.0,  # 禁止透视变换

    # 马赛克增强
    'mosaic': 1.0,       # 开启马赛克（100%使用）
    'mixup': 0.0,        # 关闭mixup（不适合SEM图像）
    'copy_paste': 0.0,   # 关闭copy_paste

    # 模糊和噪声
    #'blur': 0.0,        # 关闭模糊
    #'noise': 0.0,       # 关闭噪声（电镜图像本身有噪声）
}


def train_model(
    data_config: str,
    base_model: str = "yolov8n-seg.pt",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    project: str = "runs/segment",
    name: str = "sem_particles",
    **additional_args
) -> YOLO:
    """
    训练 YOLOv8 分割模型

    Args:
        data_config: 数据集配置文件路径 (.yaml)
        base_model: 基础模型路径
        epochs: 训练轮数
        imgsz: 输入图像尺寸
        batch: 批次大小
        project: 输出项目目录
        name: 实验名称
        **additional_args: 其他训练参数

    Returns:
        训练完成的模型
    """
    print("=" * 60)
    print("  SEM颗粒分析 - YOLOv8 训练脚本")
    print("=" * 60)
    print()

    # 检查基础模型
    if not os.path.exists(base_model):
        print(f"[-] 基础模型不存在: {base_model}")
        print(f"    将自动下载 yolov8n-seg.pt")
        base_model = "yolov8n-seg.pt"

    # 加载模型
    print(f"[*] 加载基础模型: {base_model}")
    model = YOLO(base_model)

    # 构建训练参数（合并默认Transforms和用户参数）
    train_args = {
        # 数据配置
        'data': data_config,

        # 训练轮次和批次
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch,

        # 输出配置
        'project': project,
        'name': name,
        'exist_ok': True,  # 允许覆盖同名实验

        # 优化器配置
        'optimizer': 'AdamW',
        'lr0': 0.00001,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,

        # 数据增强（SEM球形颗粒专属）
        **SEM_PARTICLE_TRANSFORMS,

        # 其他参数
        **additional_args
    }

    print("[*] 训练参数:")
    print("-" * 40)
    for key, value in train_args.items():
        if key in SEM_PARTICLE_TRANSFORMS:
            print(f"    {key:15} = {value:10} [SEM优化]")
        else:
            print(f"    {key:15} = {value}")
    print("-" * 40)
    print()

    # 开始训练
    print("[*] 开始训练...")
    print("[*] 按 Ctrl+C 可中断训练")
    print()

    try:
        results = model.train(**train_args)
        print()
        print("[+] 训练完成!")

        # 获取最佳模型路径
        best_model_path = results.save_dir / "weights/best.pt"
        if best_model_path.exists():
            print(f"[+] 最佳模型: {best_model_path}")

        return model

    except KeyboardInterrupt:
        print()
        print("[-] 训练被用户中断")
        raise
    except Exception as e:
        print()
        print(f"[-] 训练失败: {e}")
        raise


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="SEM球形颗粒 YOLOv8 训练脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python tools/train_model.py --data dataset.yaml
  python tools/train_model.py --data dataset.yaml --epochs 200 --batch 8
  python tools/train_model.py --data dataset.yaml --name my_experiment --imgsz 1024
        """
    )

    parser.add_argument(
        '--data', '-d',
        required=True,
        help='数据集配置文件路径 (.yaml)'
    )

    parser.add_argument(
        '--model', '-m',
        default='yolov8s-seg.pt',
        help='基础模型路径 (默认: yolov8s-seg.pt)'
    )

    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=100,
        help='训练轮数 (默认: 100)'
    )

    parser.add_argument(
        '--imgsz', '-i',
        type=int,
        default=1024,
        help='输入图像尺寸 (默认: 1024)'
    )

    parser.add_argument(
        '--batch', '-b',
        type=int,
        default=16,
        help='批次大小 (默认: 16)'
    )

    parser.add_argument(
        '--project', '-p',
        default='runs/segment',
        help='输出项目目录 (默认: runs/segment)'
    )

    parser.add_argument(
        '--name', '-n',
        default='sem_particles',
        help='实验名称 (默认: sem_particles)'
    )

    parser.add_argument(
        '--device', '-c',
        default='',
        help='训练设备，如 "0" 或 "cpu" (默认: 自动选择)'
    )

    parser.add_argument(
        '--resume', '-r',
        action='store_true',
        help='从上次中断处恢复训练'
    )

    args = parser.parse_args()

    # 构建额外参数
    additional_args = {}
    if args.device:
        additional_args['device'] = args.device
    if args.resume:
        additional_args['resume'] = True

    # 执行训练
    try:
        train_model(
            data_config=args.data,
            base_model=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            project=args.project,
            name=args.name,
            **additional_args
        )
    except Exception as e:
        print(f"\n[-] 训练异常退出: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
