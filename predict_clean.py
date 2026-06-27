"""
SEM颗粒预测脚本 - 干净输出版

功能：
- 加载模型进行预测
- 过滤条件：
  1. 完整度 > 0.75
  2. 不在图像边缘
- 输出干净的结果图（不画圆圈和数字）
"""

import sys
import os
from pathlib import Path

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import argparse

# 过滤参数
MIN_COMPLETENESS = 0.90  # 完整度阈值
EDGE_TOLERANCE = 5       # 边缘容差像素


def load_model(model_path: str):
    """加载模型（直接加载.pt文件）"""
    from ultralytics import YOLO

    # 如果是加密文件，转换为pt路径
    if model_path.endswith(".encrypted"):
        model_path = model_path.replace(".encrypted", ".pt")

    print(f"[*] 加载模型: {model_path}")
    model = YOLO(model_path)
    print("[+] 模型加载成功")

    return model


def calculate_completeness(contour) -> float:
    """计算轮廓完整度"""
    actual_area = cv2.contourArea(contour)

    # 计算质心
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return 0
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]

    # 计算平均半径
    distances = []
    for point in contour:
        px, py = point[0]
        dist = np.sqrt((px - cx)**2 + (py - cy)**2)
        distances.append(dist)
    radius = np.mean(distances)

    # 理论面积
    theoretical_area = np.pi * radius * radius

    return actual_area / theoretical_area if theoretical_area > 0 else 0


def is_on_border(contour, img_width, img_height, tolerance=5) -> bool:
    """检测是否在图像边缘"""
    x, y, w, h = cv2.boundingRect(contour)
    if x <= tolerance or y <= tolerance:
        return True
    if x + w >= img_width - tolerance or y + h >= img_height - tolerance:
        return True

    pts = contour.reshape(-1, 2)
    if (pts[:, 0] <= tolerance).any() or (pts[:, 0] >= img_width - tolerance).any():
        return True
    if (pts[:, 1] <= tolerance).any() or (pts[:, 1] >= img_height - tolerance).any():
        return True

    return False


def predict_and_visualize(model, image_path: str, output_path: str = None):
    """
    预测并可视化结果

    Args:
        model: YOLO模型
        image_path: 输入图像路径
        output_path: 输出图像路径（可选）
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")

    height, width = image.shape[:2]
    print(f"[*] 图像尺寸: {width}x{height}")

    # YOLO预测 - 不画任何标注
    print("[*] 运行预测...")
    results = model.predict(
        image,
        imgsz=640,  # 🔥 加上这一行，与训练时一致
        verbose=False,
        conf=0.25,
        show=False,
        save=False,
        line_width=0,  # 不画边界框
        hide_labels=True,  # 隐藏标签
        hide_conf=True  # 隐藏置信度
    )

    # 收集所有有效颗粒
    valid_contours = []

    for r in results:
        if r.masks is not None:
            masks = r.masks.data.cpu().numpy()
            orig_h, orig_w = image.shape[:2]  # 原图尺寸

            for mask in masks:
                binary_mask = (mask * 255).astype(np.uint8)
                binary_mask = cv2.resize(binary_mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)                

                # 提取轮廓
                contour, _ = cv2.findContours(
                    binary_mask,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )

                if not contour:
                    continue

                largest = max(contour, key=cv2.contourArea)

                # 过滤条件
                if cv2.contourArea(largest) < 100:
                    continue
                if is_on_border(largest, width, height, EDGE_TOLERANCE):
                    continue
                if calculate_completeness(largest) < MIN_COMPLETENESS:
                    continue

                valid_contours.append(largest)

    print(f"[*] 检测到有效颗粒: {len(valid_contours)}")

    # 创建输出图像
    output = image.copy()

    # 填充有效颗粒区域为半透明绿色（可选，显示检测区域）
    overlay = image.copy()
    for contour in valid_contours:
        cv2.fillPoly(overlay, [contour], (0, 255, 0, 50))

    # 混合原图和检测结果
    output = cv2.addWeighted(overlay, 0.3, output, 0.7, 0)

    # 添加统计信息
    stats_text = f"检测到 {len(valid_contours)} 个有效颗粒"
    cv2.putText(
        output,
        stats_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # 保存结果
    if output_path is None:
        name = Path(image_path).stem
        output_path = f"{name}_result.jpg"

    cv2.imwrite(output_path, output)
    print(f"[+] 结果已保存: {output_path}")

    # 显示结果（如果安装了cv2的GUI支持）
    try:
        cv2.imshow("Result", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        print("[*] 无法显示窗口（可能在无GUI环境）")

    return len(valid_contours)


def main():
    parser = argparse.ArgumentParser(description="SEM颗粒预测 - 干净输出")
    parser.add_argument("image", help="输入图像路径")
    parser.add_argument("-m", "--model", default="E:\\circleview\\NT-SEM\\runs\\segment\\exp_9\\weights\\best.pt", help="模型路径")
    parser.add_argument("-o", "--output", help="输出图像路径")
    parser.add_argument("--completeness", type=float, default=0.75, help="完整度阈值")
    parser.add_argument("--edge-tolerance", type=int, default=5, help="边缘容差")

    args = parser.parse_args()

    global MIN_COMPLETENESS, EDGE_TOLERANCE
    MIN_COMPLETENESS = args.completeness
    EDGE_TOLERANCE = args.edge_tolerance

    print("=" * 50)
    print("  SEM颗粒预测 - 干净输出版")
    print(f"  完整度阈值: > {MIN_COMPLETENESS}")
    print(f"  边缘容差: {EDGE_TOLERANCE}px")
    print("=" * 50)
    print()

    # 加载模型
    model = load_model(args.model)

    # 预测
    count = predict_and_visualize(model, args.image, args.output)

    print()
    print(f"[+] 完成！检测到 {count} 个有效颗粒")


if __name__ == "__main__":
    main()
