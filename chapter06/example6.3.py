from transformers import pipeline
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt

# 创建分割模型的 pipeline
detector = pipeline(task="image-segmentation")
# 加载图像并进行预测
image_path = "res/pedestrians-crosswalk.jpg"
preds = detector(image_path)
print(*preds, sep="\n")

# 读取原始图像
image = Image.open(image_path)
image_copy = image.copy()  # 保留原图，避免修改
# 用于绘制掩膜的画笔
draw = ImageDraw.Draw(image_copy)

# 假设 preds 中有多个掩膜
for i, pred in enumerate(preds):
    # 获取每个 mask 和其类别
    mask = pred['mask']
    label = pred['label']
    # 将 mask 转换为 NumPy 数组
    mask_array = np.array(mask)
    # 确保掩膜是二值化的（即 0 或 1）
    mask_array = mask_array > 0.5  # 这里假设掩膜值大于 0.5 表示目标区域
    # 创建随机颜色（可以选择任何颜色）
    color = np.random.randint(0, 256, size=3).tolist()
    # 将 mask 逐个绘制在图像上
    for y in range(mask_array.shape[0]):
        for x in range(mask_array.shape[1]):
            if mask_array[y, x]:
                draw.point((x, y), fill=tuple(color))
# 保存结果
image_copy.save("res/masks_drawn_image.png")