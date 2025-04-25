import os

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader

from utils import rgb_to_mask, mask_to_rgb


class SUIMDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform

        self.image_files = sorted(os.listdir(image_dir))

    def __getitem__(self, idx):
        # 获取图像文件和标签文件的路径
        image_name = self.image_files[idx]
        # print(image_name)
        image_path = os.path.join(self.image_dir, image_name)
        label_path = os.path.join(self.label_dir, image_name.replace(".jpg", ".bmp"))

        # 打开图像和标签文件
        image = Image.open(image_path).convert("RGB")  # 图像都转换为RGB模式
        label = Image.open(label_path)  # 此时的标签为RGB形式

        # RGB彩色形式转为一维类别编号图
        mask = np.array(label)  # PIL.Image对象没有.shape，需先转为np数组形式
        mask = rgb_to_mask(mask)  # 仍为np数组形式

        if self.transform:
            # albumentations 要求传入的类型均为np，并且mask应该是二维的类别编号图
            image = np.array(image)
            augmented = self.transform(image=image, mask=mask)
            image_aug = augmented["image"]
            mask_aug = augmented["mask"].clone().detach().long()

            return image_aug, mask_aug
        else:
            # 没有 transform 抛出异常
            raise ValueError("You must provide a transform function under the albumentations library,"
                             " and this function must include at least ToTensor.")

    def __len__(self):
        return len(self.image_files)


if __name__ == "__main__":
    train_dataset = SUIMDataset(r"data/train/images", r"data/train/labels")
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # 验证数据集图片与标签是否对应上
    img, lab = next(iter(train_dataloader))  # 迭代器
    image = img[0].permute((1, 2, 0)).numpy()  # img[0] 是tensor类型，需转为np
    label = mask_to_rgb(lab[0])  # 类别编号图转为RGB格式图

    plt.subplot(121)
    plt.imshow(image)
    plt.subplot(122)
    plt.imshow(label)

    plt.show()
