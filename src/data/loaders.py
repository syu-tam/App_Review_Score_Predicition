from .dataset import ReviewReplyDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import torch

def create_data_loader(dataframe, tokenizer, review_column_name, config):
    """
    データセットをロードし、設定に応じたDataLoaderを作成する関数。
    """
    batch_size = config.feature_extraction.batch_size
    max_len = config.feature_extraction.max_length
    shuffle = config.feature_extraction.shuffle
    num_workers = config.base.num_workers

    # 訓練データ全体を使用するか確認
    if config.feature_extraction.all_train:
        dataset = ReviewReplyDataset(dataframe, tokenizer, max_len, review_column_name)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )
    
    else:
        # データをトレーニングと検証に分割
        train_dataframe, val_dataframe = train_test_split(
            dataframe, test_size=config.feature_extraction.test_size, random_state=config.base.seed)

        # インデックスをリセットしてDataLoader用に整形
        train_dataframe = train_dataframe.reset_index(drop=True)
        val_dataframe = val_dataframe.reset_index(drop=True)

        # トレーニングおよび検証用のデータセットを作成
        train_dataset = ReviewReplyDataset(train_dataframe, tokenizer, max_len, review_column_name)
        val_dataset = ReviewReplyDataset(val_dataframe, tokenizer, max_len, review_column_name) 

        # トレーニングおよび検証用のデータローダーを返す
        return (
            DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                worker_init_fn=worker_init_fn
            ),
            DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                worker_init_fn=worker_init_fn
            )
        )

def worker_init_fn(worker_id):
    """
    ワーカーごとに異なるシードを設定する関数。
    """
    np.random.seed(torch.initial_seed() % 2**32)
