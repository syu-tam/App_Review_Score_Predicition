import random
import numpy as np
import torch
from transformers import set_seed as hf_set_seed

def set_seed(seed):
    """
    シード値を固定して、再現性を保つ。
    
    Parameters:
    - seed (int): 再現性確保のためのシード値
    """
    random.seed(seed)               # Python標準の乱数シードを設定
    np.random.seed(seed)            # Numpyの乱数シードを設定
    torch.manual_seed(seed)         # PyTorchのCPUシードを設定
    hf_set_seed(seed)               # transformersライブラリのシードを設定
    if torch.cuda.is_available():   # GPUが使用可能ならCUDAのシードも設定
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # 計算結果を再現可能に
    torch.backends.cudnn.benchmark = False     # 学習速度の最適化を無効化し再現性を重視

def print_step(step_title, detail):
    """
    処理のステップ情報を出力する。

    Parameters:
    - step_title (str): 処理ステップのタイトル
    - detail (str): 処理ステップの詳細説明
    """
    print(f"\n====== {step_title} ======")
    print(detail)
