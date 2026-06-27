import pandas as pd
import matplotlib.pyplot as plt
import os

# 改成你的实际路径
csv_path = r"E:\circleview\NT-SEM\runs\segment\exp_9\results.csv"

# 检查文件是否存在
if not os.path.exists(csv_path):
    print(f"❌ 文件不存在: {csv_path}")
    exit()

# 读取 CSV
df = pd.read_csv(csv_path, skip_blank_lines=True)

# 🔥 关键：去除列名中的首尾空格
df.columns = df.columns.str.strip()

print("清理后的列名：", df.columns.tolist())

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建图表
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 图1：mAP50 曲线
axes[0, 0].plot(df['epoch'], df['metrics/mAP50(M)'], label='mAP50', color='blue', linewidth=2)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('mAP50')
axes[0, 0].set_title('Mask mAP50 训练曲线')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend()

# 图2：Loss 曲线
axes[0, 1].plot(df['epoch'], df['train/seg_loss'], label='Train Seg Loss', color='red', linewidth=2)
axes[0, 1].plot(df['epoch'], df['val/seg_loss'], label='Val Seg Loss', color='orange', linewidth=2)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].set_title('分割损失曲线')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend()

# 图3：精确率和召回率
axes[1, 0].plot(df['epoch'], df['metrics/precision(M)'], label='Precision', color='green', linewidth=2)
axes[1, 0].plot(df['epoch'], df['metrics/recall(M)'], label='Recall', color='purple', linewidth=2)
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Score')
axes[1, 0].set_title('精确率与召回率曲线')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend()

# 图4：mAP50-95 曲线
axes[1, 1].plot(df['epoch'], df['metrics/mAP50-95(M)'], label='mAP50-95', color='brown', linewidth=2)
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('mAP50-95')
axes[1, 1].set_title('严格 mAP 曲线')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].legend()

plt.tight_layout()

# 保存图片
output_path = os.path.join(os.path.dirname(csv_path), 'results.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✅ 图表已保存到: {output_path}")

plt.show()