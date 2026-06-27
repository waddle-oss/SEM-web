import json
import cv2
import os

# 路径配置（改成你的实际路径）
JSON_DIR = r"E:\\SEMphoto\\allphoto"  # 改成你的新数据根目录
OUTPUT_DIR = r"E:\\circleview\\NT-SEM\\my_dataset"

# 创建输出目录
os.makedirs(f"{OUTPUT_DIR}/images/train", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/labels/train", exist_ok=True)

# 使用 os.walk 递归遍历所有子文件夹
for root, dirs, files in os.walk(JSON_DIR):
    for file in files:
        if not file.endswith('.json'):
            continue
        
        json_path = os.path.join(root, file)
        print(f"处理: {json_path}")
        
        # 读取JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 找对应的图片（同名的 .jpg/.jpeg/.png/.tif）
        base_name = file.replace('.json', '')
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.tif']:
            test_path = os.path.join(root, base_name + ext)
            if os.path.exists(test_path):
                img_path = test_path
                break
        
        if img_path is None:
            print(f"  跳过：找不到图片 {base_name} 在 {root}")
            continue
        
        # 读取图片尺寸
        img = cv2.imread(img_path)
        if img is None:
            print(f"  跳过：无法读取图片 {img_path}")
            continue
        
        h, w = img.shape[:2]
        
        # 复制图片到目标目录（保持原名）
        img_ext = os.path.splitext(img_path)[1]
        img_name = base_name + img_ext
        cv2.imwrite(f"{OUTPUT_DIR}/images/train/{img_name}", img)
        
        # 转换标注
        txt_path = f"{OUTPUT_DIR}/labels/train/{base_name}.txt"
        with open(txt_path, 'w') as f_out:
            for shape in data['shapes']:
                points = shape['points']
                norm = []
                for p in points:
                    norm.append(str(p[0] / w))
                    norm.append(str(p[1] / h))
                f_out.write('0 ' + ' '.join(norm) + '\n')
        
        print(f"  完成：{img_name}，共 {len(data['shapes'])} 个颗粒")

print("\n全部转换完成！")
print("图片和标注文件已保存到：", OUTPUT_DIR)