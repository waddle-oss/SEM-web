import os
import random
import shutil

DATASET_DIR = r"E:\SEMphoto\data_exp5"

# 获取所有图片
images = os.listdir(f"{DATASET_DIR}/images/train")
images = [f for f in images if f.endswith(('.jpg', '.png', '.tif', '.tiff'))]

# 随机打乱
random.shuffle(images)

total = len(images)
print(f"总图片数: {total}")

# 44 训练，5 验证，2 测试
train_count = 44
val_count = 5
test_count = 2

train_images = images[:train_count]
val_images = images[train_count:train_count+val_count]
test_images = images[train_count+val_count:train_count+val_count+test_count]

print(f"训练集: {len(train_images)}，验证集: {len(val_images)}，测试集: {len(test_images)}")

# 创建目标文件夹
for subset in ['val', 'test']:
    os.makedirs(f"{DATASET_DIR}/images/{subset}", exist_ok=True)
    os.makedirs(f"{DATASET_DIR}/labels/{subset}", exist_ok=True)

def move_files(img_list, subset):
    for img in img_list:
        # 移动图片
        src_img = os.path.join(DATASET_DIR, 'images/train', img)
        dst_img = os.path.join(DATASET_DIR, 'images', subset, img)
        shutil.move(src_img, dst_img)
        print(f"移动图片: {img} -> {subset}")
        
        # 移动对应的 txt
        txt_name = img.rsplit('.', 1)[0] + '.txt'
        src_txt = os.path.join(DATASET_DIR, 'labels/train', txt_name)
        dst_txt = os.path.join(DATASET_DIR, 'labels', subset, txt_name)
        if os.path.exists(src_txt):
            shutil.move(src_txt, dst_txt)
            print(f"移动标注: {txt_name} -> {subset}")

move_files(val_images, 'val')
move_files(test_images, 'test')

print("\n完成！")