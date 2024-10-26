import pandas as pd
from torch.utils.data import Dataset
import torch

def load_csv(config):
    """
    訓練データとテストデータのCSVファイルを読み込み、1つのデータフレームに結合する関数。
    :param config: 設定ファイルから取得したパラメータ
    :return: データフレーム (df) と訓練データの長さ (len_train)
    """
    train_data = pd.read_csv(config.data.train_file, index_col=0)
    test_data = pd.read_csv(config.data.test_file, index_col=0)
    len_train = len(train_data)
    df = pd.concat([train_data, test_data], axis=0)
    
    return df, len_train

class ReviewReplyDataset(Dataset):
    """
    レビューと返信データをトークナイズし、モデルに入力可能な形式に変換するデータセットクラス。
    """
    def __init__(self, dataframe, tokenizer, max_len, review_column_name):
        self.reviews = dataframe[review_column_name].fillna('')
        self.replies = dataframe['replyContent'].fillna('')
        self.labels = dataframe['score']
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        """
        レビューと返信のトークン化とテンソル化を行い、ラベルと共に辞書形式で返す。
        """
        review_encoding = self.tokenizer.encode_plus(
            str(self.reviews[item]),
            add_special_tokens=True,
            max_length=self.max_len // 2,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        reply_encoding = self.tokenizer.encode_plus(
            str(self.replies[item]),
            add_special_tokens=True,
            max_length=self.max_len // 2,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'review_input_ids': review_encoding['input_ids'].flatten(),
            'review_attention_mask': review_encoding['attention_mask'].flatten(),
            'reply_input_ids': reply_encoding['input_ids'].flatten(),
            'reply_attention_mask': reply_encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[item], dtype=torch.long)
        }
