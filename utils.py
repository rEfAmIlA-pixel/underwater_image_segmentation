"""
transform、rgb_to_mask 与 mask_to_rgb 都是作用单张图片，不要 batch_size
"""

import os
from collections import defaultdict

import albumentations as A
import numpy as np
import torch
from PIL import Image
from albumentations.core.composition import Compose
from albumentations.pytorch import ToTensorV2

from archs.nested_unet import NestedUNet

# 类别与编号映射关系
COLOR2LABEL = {
    (0, 0, 0): 0,  # 背景
    (0, 0, 255): 1,  # 潜水员
    (0, 255, 255): 2,  # 水下机器人
    (0, 255, 0): 3,  # 水下植物
    (255, 0, 0): 4,  # 水下动物
    (255, 255, 0): 5,  # 岩石与碎石
    (255, 0, 255): 6,  # 残骸与废墟
    (255, 255, 255): 7  # 沙地与海床
}

# 数据增强操作
transform = {
    "train":
        Compose([
            A.RandomRotate90(p=0.5),  # 随机旋转90°
            A.HorizontalFlip(p=0.5),  # 随机水平翻转
            A.VerticalFlip(p=0.5),  # 随机垂直翻转
            A.Resize(256, 256),  # 调整图像大小
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),  # 归一化
            ToTensorV2(),  # 转为Tensor
        ]),
    "valid":
        Compose([
            A.Resize(256, 256),  # 调整图像大小
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),  # 归一化
            ToTensorV2(),  # 转为Tensor
        ])
}


def rgb_to_mask(label_image):
    """
    将 RGB 掩码图（H * W * 3）转为 类别编码图（H * W）
    """
    # mask 是np数组格式
    h, w, _ = label_image.shape
    label_mask = np.zeros((h, w), dtype=np.uint8)

    for rgb, idx in COLOR2LABEL.items():
        match = np.all(label_image == rgb, axis=-1)
        label_mask[match] = idx

    return label_mask


def mask_to_rgb(mask):
    """
    将类别编号图（H * W）转换为 RGB 掩码图（H * W * 3）
    """
    # mask 是np数组格式
    h, w = mask.shape
    # rgb_mask 是一个空的三维np矩阵
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)

    # 将每个类别编号映射为对应的RGB值
    for rgb, idx in COLOR2LABEL.items():
        # rgb_mask[mask == idx] 的作用是
        # 选择 rgb_mask 中所有与 mask == idx 对应的 True 的位置
        rgb_mask[mask == idx] = rgb

    return rgb_mask


def calculate_class_weights_from_folder(target_dir, num_classes, clamp_range=(0.5, 5.0)):
    """
    计算每种类别的权重，使用反比频率法。

    :param target_dir: 存储标签的目录路径
    :param num_classes: 总类别数
    :param clamp_range: 限幅
    :return: 类别权重的 Pytorch Tensor
    """
    target_counts = defaultdict(int)
    total_pixels = 0

    for filename in os.listdir(target_dir):
        if filename.endswith(".bmp"):
            print("现在在处理：" + filename)
            target_path = os.path.join(target_dir, filename)
            # 读取到的是RGB三通道图
            target_img = Image.open(target_path)  # (h, w, 3)
            target_array = np.array(target_img)  # (h, w, 3)
            # 转为类别编码图
            mask_array = rgb_to_mask(target_array)  # (h, w)

            for cls in range(num_classes):
                count = np.sum(cls == mask_array)
                target_counts[cls] += count

            total_pixels += target_array.size
            print("total_pixels:", total_pixels)

        print()

    # 计算频率
    freqs = np.array([target_counts[c] / total_pixels if total_pixels > 0 else 0. for c in range(num_classes)])
    # log 平滑的反比频率
    weights = 1. / (np.log(1.02 + freqs) + 1e-6)  # 避免log(0)
    # 归一化
    weights = weights / weights.sum() * num_classes
    # 限幅
    weights = np.clip(weights, *clamp_range)

    return torch.tensor(weights, dtype=torch.float32)


def load_model(num_classes, model_path, device):
    """加载模型"""
    model = NestedUNet(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


if __name__ == "__main__":
    class_weights = calculate_class_weights_from_folder("data/train/labels", 8)
    print("训练集的类别权重为：" + str(class_weights) + "，可以直接将结果放置train文件中。")
