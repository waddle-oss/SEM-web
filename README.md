# SEM 颗粒智能分析系统

基于 YOLOv8n-seg 实例分割模型的生物图像分析工具，用于自动识别并统计显微图像中颗粒（圆形对象）的数量、粒径分布及形态参数。


## 1. 项目简介

本系统面向生物学科实验场景，提供一站式的图像分析解决方案：

- **核心功能**：自动识别图像中的圆形颗粒，输出数量、面积、周长、圆度等量化指标。
- **应用场景**：生物细胞计数、微球颗粒分析、材料科学中的粒径分布统计等。
- **工作模式**：用户上传图像 → 后端调用分割模型推理 → 返回识别结果与统计数据。


## 2. 技术栈

| 模块 | 技术选型 |
|------|----------|
| **后端框架** | Python + Flask |
| **深度学习模型** | YOLOv8n-seg（实例分割） |
| **前端界面** | 原生 HTML + CSS + JavaScript（单页应用） |
| **数据库** | MySQL 5.7.43 |
| **服务器** | 腾讯云轻量服务器 |
| **反向代理** | Nginx + FRP 内网穿透 |


## 3. 目录结构说明

```
SEM颗粒智能分析系统/
├── core/                      # 核心业务逻辑模块
│   ├── __init__.py
│   ├── analyzer.py            # 图像分析主逻辑
│   ├── scale_reader.py        # 标尺/比例尺识别
│   └── security.py            # 安全相关功能
├── tools/                     # 辅助工具脚本
│   ├── __init__.py
│   ├── circle_to_polygon.py   # 圆转多边形工具
│   ├── encrypt_model.py       # 模型加密脚本
│   ├── plot_results.py        # 结果可视化绘图
│   └── train_model.py         # 模型训练脚本
├── my_dataset/                # 数据集存放目录
├── runs/                      # 训练日志与输出
├── best.encrypted             # 加密后的模型权重文件
├── main.py                    # Flask 应用入口（启动文件）
├── app.py                     # 应用主程序（与main.py功能一致，二选一）
├── predict_clean.py           # 干净图像预测脚本
├── predict_cross.py           # 交叉验证预测
├── predict_cross2.py          # 交叉验证预测（变体）
├── yolov8n-seg.py             # YOLOv8n-seg 模型定义与加载
├── index.html                 # 前端主页面
├── requirements.txt           # Python 依赖清单
├── dataset.yaml               # 数据集配置文件
├── labels.txt                 # 类别标签
├── train.txt / val.txt        # 训练集/验证集划分
├── convert_json_to_txt.py     # JSON 转 TXT 标签格式
├── convert.py                 # 格式转换脚本
├── generate_txt.py            # 标签文件生成
├── split_dataset.py           # 数据集拆分工具
├── test-api.py                # API 接口测试脚本
└── README.md                  # 项目说明文档
```


## 4. 环境准备与安装

### 4.1 Python 虚拟环境

本项目使用 Python 虚拟环境管理依赖，请先创建并激活虚拟环境：

```bash
# 创建虚拟环境（假设使用 venv）
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux / Mac:
source venv/bin/activate
```

### 4.2 安装依赖

```bash
pip install -r requirements.txt
```


## 5. 数据库配置

- **数据库类型**：MySQL 5.7.43
- **数据库名**：`sem_users`
- **数据表结构**：请执行 `database_structure.sql` 文件完成建表：

```bash
mysql -u root -p sem_users < database/database_structure.sql
```


## 6. 启动运行

### 6.1 启动 FRP 内网穿透（本地执行）

由于后端模型部署在本地电脑，需通过 FRP 将本地服务映射到腾讯云服务器：

```bash
D:\frp\frpc.exe -c D:\frp\frpc.toml
frpc.exe -c frpc.toml
```

### 6.2 启动后端服务（本地执行）

```bash
# 确保已激活虚拟环境
python app.py
```

后端服务默认运行在 `http://localhost:5000`，通过 FRP 映射后，云服务器可通过内网地址访问。

### 6.3 访问前端（服务器端）

用户通过浏览器访问腾讯云服务器的 Nginx 代理地址（如 `http://110.42.196.53`），前端页面通过 Nginx 反向代理将 API 请求转发至 FRP 映射的后端端口。


## 7. 使用说明

1. 打开前端页面，点击“上传图像”按钮选择待分析的显微图像。
2. 系统自动调用后端 YOLOv8n-seg 模型进行实例分割。
3. 分析完成后，页面将展示：
   - 识别出的颗粒总数
   - 每个颗粒的轮廓标注
   - 颗粒面积、周长、圆度等统计表格
   - 粒径分布直方图（可选）



