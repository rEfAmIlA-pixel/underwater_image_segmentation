"""
实现单张图片的预测功能
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from utils import transform, mask_to_rgb, load_model

# 模型初始化参数
num_classes = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化模型
model = load_model(8, "checkpoints/best_model_3.pt", device)
model.to(device)
model.eval()

# 读取图像
img_path = "data/train/images/n_l_6_.jpg"
img = Image.open(img_path)
print("读取的图像为:", img_path)

# Image对象 -> numpy数组
img_array = np.array(img)
print("数据增强前，原图像形状为:", img_array.shape)
# 数据增强
img_tensor = transform["valid"](image=img_array)["image"]
print("数据增强后，图像形状变为:", img_tensor.shape)

# 在第一维度插入一个维度
img_tensor = img_tensor.unsqueeze(0)
print("在第一维度插入一个维度，图像形状变为:", img_tensor.shape)
print()

# 将图像数据放入指定设备
img_tensor = img_tensor.to(device)
# 得到预测结果 —— 类别得分图
start = time.time()
with torch.no_grad():
    outputs = model(img_tensor)
end = time.time()
print("模型预测用时: {:.4f}".format(end - start))
print()
print("outputs（类别得分图）的形状为:", outputs.shape)

# 类别得分图 -> 类别编码图
preds = outputs.argmax(1)
print("preds（类别编码图）的形状为:", preds.shape)

# 删除第一维度
preds = preds.squeeze()
print("类别编码图删除第一维度后的形状为:", preds.shape)

# tensor -> np.array
preds_array = np.array(preds.cpu())
print("类别编码图由 tensor -> np.array，形状为:", preds_array.shape)
# 将预测图调整为原来大小
original_size = img.size  # PIL 中是 (width, height)
preds_img = Image.fromarray(preds_array.astype(np.uint8))
preds_img_resized = preds_img.resize(original_size, resample=Image.NEAREST)
preds_array = np.array(preds_img_resized)
# 类别编码图 -> RGB掩码图
preds_array = mask_to_rgb(preds_array)
print("类别编码图 -> RGB掩码图，形状为:", preds_array.shape)

# 读取真实标签
target_path = img_path.replace("images", "labels").replace('.jpg', '.bmp')
target = Image.open(target_path)

# 展示结果
plt.subplot(1, 3, 1)
plt.imshow(img)
plt.title("Original drawing")

plt.subplot(1, 3, 2)
plt.imshow(target)
plt.title("Real label")

plt.subplot(1, 3, 3)
plt.imshow(preds_array)
plt.title("Prediction label")

plt.show()
