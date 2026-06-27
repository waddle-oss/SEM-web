import json
import math
import glob
import os


def convert_circle_to_polygon(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    converted_count = 0
    for shape in data['shapes']:
        # 只处理类型为 circle 的标注
        if shape['shape_type'] == 'circle':
            # LabelMe 的 circle 存储格式为: [中心点, 边缘点]
            center = shape['points'][0]
            edge = shape['points'][1]

            cx, cy = center[0], center[1]
            # 计算半径: R = sqrt((x2-x1)^2 + (y2-y1)^2)
            radius = math.sqrt((edge[0] - cx) ** 2 + (edge[1] - cy) ** 2)

            # 生成 16 个多边形边缘点（这个数量对 YOLO 来说足够精准了）
            new_points = []
            num_segments = 16
            for i in range(num_segments):
                angle = math.radians(i * (360 / num_segments))
                px = cx + radius * math.cos(angle)
                py = cy + radius * math.sin(angle)
                new_points.append([px, py])

            # 狸猫换太子：修改类型和坐标点
            shape['shape_type'] = 'polygon'
            shape['points'] = new_points
            converted_count += 1

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    if converted_count > 0:
        print(f"✅ {os.path.basename(json_path)}: 成功转换 {converted_count} 个圆。")


if __name__ == '__main__':
    # 【注意】请确保这个路径是你存放那几张标注好的 JSON 的地方
    target_dir = "E:\图片（原始） (4)\\图片（原始）"
    json_files = glob.glob(os.path.join(target_dir, "*.json"))

    for j_path in json_files:
        convert_circle_to_polygon(j_path)
    print("\n🎉 全部圆转换多边形完成！你现在可以用 LabelMe 重新打开看看，圆已经变成细密的点阵了。")