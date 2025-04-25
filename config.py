import torch


class Config:
    # 数据配置
    train_dir = "data/train/"
    valid_dir = "data/valid/"

    # 训练配置
    epochs = 50
    classes = 8
    batch_size = 32
    lr = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模型保存
    best_model_path = "checkpoints/best_model_.pt"
    final_model_path = "checkpoints/final_model_.pt"

    # 评估标准图保存
    save_metrics_dir = "evaluation_results/"
