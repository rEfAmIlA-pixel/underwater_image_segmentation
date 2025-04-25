"""
* 期望传进来的参数类型
    * targets: 类别编码图 (b, h, w) 类型为torch.int64
    * preds:   类别得分图 (b, c, h, w) 类型为torch.float32
* 函数会自动将参数转为
    * targets: 类别编码图 (b, h, w) 类型为torch.int64
    * preds:   类别编码图 (b, h, w) 类型为torch.int64
"""
import matplotlib.pyplot as plt
import numpy as np
import torch


def score2mask(preds):
    """将类别得分图（logits）转为类别编码图"""
    return preds.argmax(dim=1)


class IoUMeter:
    def __init__(self, num_classes, class_names=None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class {i}" for i in range(self.num_classes)]

        # 每个 epoch 中各类 iou（list(float)）
        self.per_class_iou_over_single_epoch = []
        # 所有 epoch 中各类的 IoU（list(list)）
        self.per_class_iou_over_epochs = []
        # 所有 epoch 的 mIoU（list(float)）
        self.mean_iou_over_epochs = []

    def reset(self):
        """重置所有累计统计信息"""
        self.total_inter = [0 for _ in range(self.num_classes)]
        self.total_union = [0 for _ in range(self.num_classes)]

    def update(self, preds, targets):
        """
        更新统计信息
        :param preds: 模型输出 (logits)，形状 (B, C, H, W)
        :param targets: 真实标签，形状 (B, H, W)
        """
        preds = score2mask(preds)
        if isinstance(preds, torch.Tensor):
            preds = preds.cpu()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu()

        for cls in range(self.num_classes):
            pred_cls = (preds == cls)
            target_cls = (targets == cls)
            intersection = (pred_cls & target_cls).sum().item()
            union = (pred_cls | target_cls).sum().item()
            self.total_inter[cls] += intersection
            self.total_union[cls] += union

    def compute_per_class_iou_over_single_epoch(self):
        """
        计算单个 epoch 中各类的 IoU
        :return: list，每个元素是一个类别的 IoU
        """
        ious = []
        for inter, union in zip(self.total_inter, self.total_union):
            if union > 0:
                ious.append(inter / union)
            else:
                ious.append(float("nan"))

        self.per_class_iou_over_epochs.append(ious)
        self.mean_iou_over_epochs.append(np.nanmean(ious))
        self.per_class_iou_over_single_epoch = ious

    def plot_per_class_iou_over_single_epoch(self, title="Per-Class IoU over Single Epoch", save_path=None):
        """画当前 epoch 的 per-class IoU 柱状图"""
        ious = self.per_class_iou_over_single_epoch
        if not ious:
            print("There is no data available for plotting in this epoch.")
            return

        # 生成一个 0 到 self.num_classes - 1 的整数数组
        x = np.arange(self.num_classes)
        # 创建一个新的窗口（figure），设置画布大小
        plt.figure(figsize=(10, 5))
        # 绘制柱状图
        bars = plt.bar(x, ious, color="skyblue")
        # 设置横坐标的标签
        plt.xticks(x, self.class_names, rotation=45)
        # 设置纵轴的标签
        plt.ylabel("IoU")
        # 设置纵轴的显示范围
        plt.ylim(0, 1.0)
        # 设置柱状图的标题
        plt.title(title)
        # 给柱状图添加数值标注
        for bar, iou in zip(bars, ious):
            if not np.isnan(iou):
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{iou:.2f}", ha='center',
                         va='bottom', fontsize=9)

        # 自动调整图表的边距和布局，防止标签或标题被裁剪掉
        plt.tight_layout()
        # 展示柱状图
        plt.show()

        # 保存柱状图
        if save_path:
            plt.savefig(save_path)

    def plot_per_class_iou_over_epochs(self, title="Per-Class IoU over Epochs", save_path=None):
        """画每类在多个 epoch 上的 IoU 折线图"""
        if not self.per_class_iou_over_epochs:
            print("There is no data available for plotting in any epoch.")
            return

        print("self.per_class_iou_over_epochs:", self.per_class_iou_over_epochs)

        epochs = np.arange(1, len(self.per_class_iou_over_epochs) + 1)
        # 转置
        per_class_ious = list(zip(*self.per_class_iou_over_epochs))

        plt.figure(figsize=(10, 6))
        for i, single_class_ious in enumerate(per_class_ious):
            plt.plot(epochs, single_class_ious, label=self.class_names[i], marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("IoU")
        plt.title(title)
        plt.ylim(0, 1.0)
        plt.legend()  # 显示图例——每条线代表什么类别
        plt.grid(True)  # 显示网格线
        plt.tight_layout()
        plt.show()

        if save_path:
            plt.savefig(save_path)

    def plot_mean_iou_over_epochs(self, title="Mean IoU over Epochs", save_path=None):
        """画多个 epoch 的 mIoU折线图"""
        if not self.mean_iou_over_epochs:
            print("There is no data available for plotting in any epoch.")
            return

        print("self.mean_iou_over_epochs:", self.mean_iou_over_epochs)

        epochs = np.arange(1, len(self.mean_iou_over_epochs) + 1)
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, self.mean_iou_over_epochs, marker="o", color="orange")
        plt.xlabel("Epoch")
        plt.ylabel("mIoU")
        plt.ylim(0, 1.0)
        plt.title(title)
        plt.grid()
        plt.tight_layout()
        plt.show()

        if save_path:
            plt.savefig(save_path)
