# 模型配置
MODEL_CONFIG = {
    'in_channels': 1,
    'model_channels': 64,
    'out_channels': 1,
    'num_res_blocks': 2,
    'attention_resolutions': [16],
    'channel_mult': [1, 2, 4],
    'num_classes': 10,
    'dropout': 0.1
}

# 训练配置
TRAIN_CONFIG = {
    'epochs': 100,
    'batch_size': 128,
    'learning_rate': 2e-4,
    'weight_decay': 0.01,
    'save_every': 10,
    'num_workers': 4,
    'grad_clip_norm': 1.0
}

# 数据配置
DATA_CONFIG = {
    'dataset': 'mnist',
    'image_size': 28,
    'normalize_range': [-1, 1]  # 归一化到 [-1, 1]
}