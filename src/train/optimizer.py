import torch.optim as optim

def get_optimizer(model, config):
    """
    configファイルに基づいて最適化関数を選択する関数。
    """
    optimizer_name = config.feature_extraction.optimizer.type
    lr = float(config.feature_extraction.optimizer.lr)
    momentum = float(config.feature_extraction.optimizer.momentum)
    weight_decay = float(config.feature_extraction.optimizer.weight_decay)
    
    if optimizer_name == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    elif optimizer_name == "adam":
        return optim.Adam(model.parameters(), lr=lr)
    
    elif optimizer_name == "sgd":
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    
    elif optimizer_name == "rmsprop":
        return optim.RMSprop(model.parameters(), lr=lr, momentum=momentum)
    
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
