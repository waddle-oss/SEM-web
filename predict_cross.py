"""
SEM颗粒预测脚本 - 直径标记版

功能：
- 加载模型进行预测
- 过滤条件：
  1. 完整度 > 0.75
  2. 不在图像边缘
- 在每个有效颗粒中心画两条90度相交的直径
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import argparse

MIN_COMPLETENESS = 0.75
EDGE_TOLERANCE = 5


def load_model(model_path: str):
    """加载模型"""
    from ultralytics import YOLO

    if model_path.endswith(".encrypted"):
        model_path = model_path.replace(".encrypted", ".pt")

    print(f"[*] 加载模型: {model_path}")
    model = YOLO(model_path)
    print("[+] 模型加载成功")

    return model


def calculate_completeness(contour) -> float:
    """计算轮廓完整度"""
    actual_area = cv2.contourArea(contour)

    M = cv2.moments(contour)
    if M["m00"] == 0:
        return 0
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]

    distances = []
    for point in contour:
        px, py = point[0]
        dist = np.sqrt((px - cx)**2 + (py - cy)**2)
        distances.append(dist)
    radius = np.mean(distances)

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


def get_particle_center_radius(contour):
    """获取颗粒的中心点与平均径向距离（用于绘制直径十字线）"""
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None, None

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    distances = []
    for point in contour:
        px, py = point[0]
        dist = np.sqrt((px - cx)**2 + (py - cy)**2)
        distances.append(dist)
    mean_r = np.mean(distances)

    return (cx, cy), mean_r


def draw_cross_diameters(img, center, mean_r, line_length_ratio=0.8):
    """
    在颗粒中心画两条90度相交的直径

    Args:
        img: 图像
        center: 圆心坐标 (x, y)
        mean_r: 颗粒平均径向距离
        line_length_ratio: 线段半长占 mean_r 的比例（默认0.8）
    """
    if center is None or mean_r is None:
        return

    cx, cy = center
    half_length = int(mean_r * line_length_ratio)

    color = (0, 255, 0)
    thickness = 2

    cv2.line(img, (cx - half_length, cy), (cx + half_length, cy), color, thickness)
    cv2.line(img, (cx, cy - half_length), (cx, cy + half_length), color, thickness)


def predict_and_visualize(model, image_path: str, output_path: str = None):
    """预测并可视化结果"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")

    height, width = image.shape[:2]
    print(f"[*] 图像尺寸: {width}x{height}")

    print("[*] 运行预测...")
    results = model.predict(
        image,
        imgsz=640,  # 🔥 加上这一行，与训练时一致
        verbose=False,
        conf=0.25,
        show=False,
        save=False,
        line_width=0,
        hide_labels=True,
        hide_conf=True
    )

    valid_contours = []

    for r in results:
        if r.masks is not None:
            masks = r.masks.data.cpu().numpy()
            orig_h, orig_w = image.shape[:2]  # 原图尺寸

            for mask in masks:
                binary_mask = (mask * 255).astype(np.uint8)
                binary_mask = cv2.resize(binary_mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)                

                contour, _ = cv2.findContours(
                    binary_mask,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )

                if not contour:
                    continue

                largest = max(contour, key=cv2.contourArea)

                if cv2.contourArea(largest) < 100:
                    continue
                if is_on_border(largest, width, height, EDGE_TOLERANCE):
                    continue
                if calculate_completeness(largest) < MIN_COMPLETENESS:
                    continue

                valid_contours.append(largest)

    print(f"[*] 检测到有效颗粒: {len(valid_contours)}")

    output = image.copy()

    for contour in valid_contours:
        center, radius = get_particle_center_radius(contour)
        if center and radius:
            draw_cross_diameters(output, center, radius)

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

    if output_path is None:
        name = Path(image_path).stem
        output_path = f"{name}_result.jpg"

    cv2.imwrite(output_path, output)
    print(f"[+] 结果已保存: {output_path}")

    try:
        cv2.imshow("Result", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        print("[*] 无法显示窗口（可能在无GUI环境）")

    return len(valid_contours)


def main():
    parser = argparse.ArgumentParser(description="SEM颗粒预测 - 直径标记版")
    parser.add_argument("image", help="预测图片路径E:\\SEMphoto\\allphoto\\8")
    parser.add_argument("-m", "--model", default="E:\\circleview\\NT-SEM\\runs\\segment\\exp_9\\weights\\best.pt", help="模型E:\\circleview\\NT-SEM\\runs\\segment\\exp_9\\weights\\best.pt")
    parser.add_argument("-o", "--output", help="输出路径E:\\circleview\\NT-SEM\\runs\\segment\\predict4")
    parser.add_argument("--completeness", type=float, default=0.75, help="完整度阈值")
    parser.add_argument("--edge-tolerance", type=int, default=5, help="边缘容差")

    args = parser.parse_args()

    global MIN_COMPLETENESS, EDGE_TOLERANCE
    MIN_COMPLETENESS = args.completeness
    EDGE_TOLERANCE = args.edge_tolerance

    print("=" * 50)
    print("  SEM颗粒预测 - 直径标记版")
    print(f"  完整度阈值: > {MIN_COMPLETENESS}")
    print(f"  边缘容差: {EDGE_TOLERANCE}px")
    print("=" * 50)
    print()

    model = load_model(args.model)
    count = predict_and_visualize(model, args.image, args.output)

    print()
    print(f"[+] 完成！检测到 {count} 个有效颗粒")


if __name__ == "__main__":
    main()
