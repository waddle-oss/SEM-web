import requests

# 改成你的图片路径
image_path = r"E:\\SEMphoto\\图片2-1.jpg"

with open(image_path, "rb") as f:
    response = requests.post(
        "http://localhost:8000/analyze_plain",
        files={"file": f}
    )

print("状态码:", response.status_code)
print("结果:", response.json())