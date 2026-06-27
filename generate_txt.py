import os

# 配置路径
IMAGE_DIR = r"E:\circleview\NT-SEM\my_dataset\images\train"
OUTPUT_DIR = r"E:\circleview\NT-SEM"  # dataset.yaml所在的目录

# 获取所有图片文件的绝对路径
image_paths = []
for f in os.listdir(IMAGE_DIR):
    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
        # 生成绝对路径，注意替换反斜杠为正斜杠
        abs_path = os.path.join(IMAGE_DIR, f).replace("\\", "/")
        image_paths.append(abs_path)

# 简单划分：80% 训练，20% 验证
split_idx = int(len(image_paths) * 0.8)
train_paths = image_paths[:split_idx]
val_paths = image_paths[split_idx:]

# 写入 train.txt
with open(os.path.join(OUTPUT_DIR, "train.txt"), "w") as f:
    f.write("\n".join(train_paths))

# 写入 val.txt
with open(os.path.join(OUTPUT_DIR, "val.txt"), "w") as f:
    f.write("\n".join(val_paths))

print(f"生成完成！训练集 {len(train_paths)} 张，验证集 {len(val_paths)} 张。")