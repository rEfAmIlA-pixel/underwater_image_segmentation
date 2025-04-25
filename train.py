import os
import time
import warnings

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from archs.nested_unet import NestedUNet
from config import Config as cfg
from dataset import SUIMDataset
from losses import DiceCELoss
from metrics.iou_meter import IoUMeter
from utils import transform, load_model

# 忽略 albumentations 库的版本请求警告
warnings.filterwarnings("ignore", category=UserWarning, module="albumentations.check_version")


# 数据加载器
def get_dataloader(train_dir, valid_dir, batch_size):
    train_image_dir = os.path.join(train_dir, "images")
    train_label_dir = os.path.join(train_dir, "labels")
    valid_image_dir = os.path.join(valid_dir, "images")
    valid_label_dir = os.path.join(valid_dir, "labels")
    # print(train_image_dir)
    # print(train_label_dir)
    # print(valid_image_dir)
    # print(valid_label_dir)

    train_dataset = SUIMDataset(train_image_dir, train_label_dir, transform["train"])
    valid_dataset = SUIMDataset(valid_image_dir, valid_label_dir, transform["valid"])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, valid_dataloader


# 训练参数设置
suim_class_names = [
    "background",  # 背景
    "human_divers",  # 潜水员
    "robots",  # 水下机器人
    "ruins_and_obstacles",  # 遗迹与障碍物
    "flora",  # 水下植物
    "fish_and_marine_animals",  # 鱼类和海洋动物
    "wrecks_and_artefacts",  # 残骸与人工制品
    "sand_and_rubble"  # 沙子和碎石
]
best_val_loss = float("inf")  # 初始化为无穷大
start_time = time.time()

# 初始化数据加载器
train_dataloader, valid_dataloader = get_dataloader(cfg.train_dir, cfg.valid_dir, cfg.batch_size)

# 模型初始化
model = NestedUNet(cfg.classes, 3, deep_supervision=False)  # 全新模型
# model = load_model(cfg.classes, "checkpoints/final_model.pt", cfg.device)  # 已有模型
model.to(cfg.device)

# 定义损失函数与优化器
criterion = DiceCELoss(
    torch.tensor([0.5000, 1.4952, 0.9674, 1.5045, 1.8114, 0.9474, 0.5000, 0.6206]).to(cfg.device))  # CE 和 Dice 组合
optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# ----------评估标准---------- #
iou_meter = IoUMeter(cfg.classes, suim_class_names)

# 训练循环
for epoch in range(cfg.epochs):
    model.train()
    running_loss = 0.

    # 训练阶段
    for i, (inputs, targets) in enumerate(train_dataloader):
        inputs, targets = inputs.to(cfg.device), targets.to(cfg.device)

        # 清除梯度
        optimizer.zero_grad()

        # 前向传播
        outputs = model(inputs)

        # 计算损失
        loss = criterion(outputs, targets)  # targets 是类别编码图

        # 反向传播
        loss.backward()

        # 更新权重
        optimizer.step()

        # 累加损失（跟踪和观察模型的学习进度和效果）
        running_loss += loss.item()

        # 每隔 10 batch 打印一次损失
        if (i + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{cfg.epochs}], Step [{i + 1}/{len(train_dataloader)}], Loss: {loss.item():.4f}")

    # 每个 epoch 结束打印平均损失（len(train_dataloader)计算的是批次数）
    print(f"Epoch [{epoch + 1}/{cfg.epochs}] Training Loss: {running_loss / len(train_dataloader):.4f}")

    # 验证阶段（训练完一个 epoch 验证一次）
    model.eval()
    val_loss = 0.

    # ----------评估标准---------- #
    iou_meter.reset()

    with torch.no_grad():  # 不构建计算图
        for inputs, targets in valid_dataloader:
            # 数据放入指定设备
            inputs, targets = inputs.to(cfg.device), targets.to(cfg.device)

            # 预测结果
            outputs = model(inputs)

            # 计算损失值并累加
            loss = criterion(outputs, targets)
            val_loss += loss.item()

            # ----------评估标准---------- #
            iou_meter.update(outputs, targets)

    # 打印验证集损失
    avg_val_loss = val_loss / len(valid_dataloader)
    print(f"Validation Loss after Epoch {epoch + 1}: {avg_val_loss:.4f}")

    # 学习率衰减
    scheduler.step(avg_val_loss)

    # 保存最优模型
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), cfg.best_model_path)
        print(f"Saved best model with validation loss: {best_val_loss:.4f}")
    print()

    # ----------评估标准---------- #
    iou_meter.compute_per_class_iou_over_single_epoch()
    # 只保存最后一个 epoch 的 per_class_over_single_epoch 柱状图
    if epoch + 1 == cfg.epochs:
        iou_meter.plot_per_class_iou_over_single_epoch(title="Per-Class IoU over Final Epoch",
                                                       save_path=os.path.join(cfg.save_metrics_dir,
                                                                              "per_class_iou_over_final_epoch"))

# 训练结束后保存最终模型
torch.save(model.state_dict(), cfg.final_model_path)
print("Training complete. Final model saved.")
print()

# ----------评估标准---------- #
iou_meter.plot_per_class_iou_over_epochs(save_path=os.path.join(cfg.save_metrics_dir, "per_class_iou_over_epochs"))
iou_meter.plot_mean_iou_over_epochs(save_path=os.path.join(cfg.save_metrics_dir, "mean_iou_over_epochs"))

# 结束时间
end_time = time.time()
duration = end_time - start_time
print("It costs {:.0f}m {:.0f}s".format(duration // 60, duration % 60))
