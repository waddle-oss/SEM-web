import os
import json
from PIL import Image


# ===============================
# tif转化成jpg
# ===============================

SOURCE_DIR = r"E:\图片"


# 输出文件夹（自动生成）
OUTPUT_DIR = os.path.join(SOURCE_DIR, "jpg_dataset")


# ===============================
# 开始转换
# ===============================

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


success = 0
failed = 0


print("开始转换...\n")


for filename in os.listdir(SOURCE_DIR):

    # 只处理 tif
    if filename.lower().endswith((".tif", ".tiff")):

        tif_path = os.path.join(
            SOURCE_DIR,
            filename
        )

        name = os.path.splitext(filename)[0]

        jpg_name = name + ".jpg"

        jpg_path = os.path.join(
            OUTPUT_DIR,
            jpg_name
        )


        try:

            # ============
            # TIFF 转 JPG
            # ============

            img = Image.open(tif_path)


            # 处理灰度/16bit TIFF
            if img.mode != "RGB":
                img = img.convert("RGB")


            img.save(
                jpg_path,
                "JPEG",
                quality=95
            )


            # ============
            # 修改 JSON
            # ============

            json_old = os.path.join(
                SOURCE_DIR,
                name + ".json"
            )


            if os.path.exists(json_old):

                with open(
                    json_old,
                    "r",
                    encoding="utf-8"
                ) as f:

                    data = json.load(f)


                # 修改图片名字
                data["imagePath"] = jpg_name


                json_new = os.path.join(
                    OUTPUT_DIR,
                    name + ".json"
                )


                with open(
                    json_new,
                    "w",
                    encoding="utf-8"
                ) as f:

                    json.dump(
                        data,
                        f,
                        ensure_ascii=False,
                        indent=2
                    )


            success += 1

            print(
                f"✔ {filename}  --->  {jpg_name}"
            )


        except Exception as e:

            failed += 1

            print(
                f"✘ {filename} 转换失败:"
            )
            print(e)



print("\n========================")
print("转换完成")
print(f"成功: {success}")
print(f"失败: {failed}")
print("========================")

print("\n输出位置:")
print(OUTPUT_DIR)