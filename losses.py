"""
将CE与Dice两个损失函数结合起来，必须注意：
DiceLoss 与 CELoss 两个函数接收的pred和target传进来是一样的，但在函数中处理不同
* 两个函数接收到的target都是类别编码图[B, H, W]（long），pred都是类别得分图[B, C, H, W]（float）
    * CELoss 函数直接使用接收到的内容
    * DiceLoss 函数需要对传进来的参数进行处理
        * target 类别编码图 ---> one-hot编码图（float）
        * pred   类别得分图 ---> 类别概率图（float）

@类别编码图：只有一个通道，每个通道中的每个像素值为该点的类别 [B, H, W]
@one-hot编码图：有（类别数）个通道，该点的类别对应的通道的该点值为1，其余通道的该点值为0 [B, C, H, W]
@类别得分图：模型预测出来还没被处理 [B, C, H, W]
@类别概率图：类别得分图经过softmax或其他函数处理 [B, C, H, W]
"""

import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLossWithWeight(nn.Module):
    """加权交叉熵损失函数"""
    """单独再写成一个类是为了便于拓展"""

    def __init__(self, weight=None):
        # 加入了weight之后，交叉熵函数就变成了加权交叉熵函数
        # 一般类别数量少的权重大
        super(CrossEntropyLossWithWeight, self).__init__()
        self.loss = nn.CrossEntropyLoss(weight=weight)

    def forward(self, pred, target):
        return self.loss(pred, target)


class DiceLoss(nn.Module):
    """骰子损失函数"""

    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # target 类别编码图（long） ---> one-hot编码图（float）
        num_classes = pred.shape[1]
        target_one_hot = F.one_hot(target, num_classes=num_classes)  # [B, H, W, C]
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()  # -> [B, C, H, W]

        # pred 类别概率图（float） ---> 类别概率图（float）
        pred = F.softmax(pred, dim=1)  # softmax 获取每类的概率值

        # 依据对应公式计算 Dice系数（重叠系数）：2 * |A ∩ B| / (|A| + |B|)
        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        loss = 1 - dice.mean()
        return loss


class DiceCELoss(nn.Module):
    """将两个损失函数结合起来，一般是损失值相加即可"""

    def __init__(self, weight=None, smooth=1e-6):
        super(DiceCELoss, self).__init__()
        self.ce_loss_func = CrossEntropyLossWithWeight(weight)
        self.dice_loss_func = DiceLoss(smooth)

    def forward(self, pred, target):
        ce_loss = self.ce_loss_func(pred, target)
        dice_loss = self.dice_loss_func(pred, target)
        return ce_loss + dice_loss
