from transformers import BertTokenizer
from models.bert_sentiment import CustomSentimentModel
import torch
import os

def build_model_and_tokenizer(config):
    """
    モデル名に基づいてトークナイザーとモデルを生成する関数。
    """
    model_type = config.feature_extraction.type
    
    if model_type == "bert_sentiment":
        # BERTトークナイザーと感情分析モデルのロード
        tokenizer = BertTokenizer.from_pretrained(config.feature_extraction.pretrained_model_name)
        model = CustomSentimentModel(config)
        return tokenizer, model
    
    # 他のモデルタイプを追加するためのプレースホルダー
    # elif model_type == "another_model":
    #     tokenizer = ...
    #     model = ...
    #     return tokenizer, model
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def load_trained_model(config):
    """
    保存済みモデルとトークナイザーを指定パスから読み込む関数。
    """
    # モデルとトークナイザーパスの取得
    model_path = os.path.join(config.feature_extraction.trained_model_path, 'model.pth')
    tokenizer_path = os.path.join(config.feature_extraction.trained_model_path, 'tokenizer')

    # 新しいモデルとトークナイザーのビルド
    _, model = build_model_and_tokenizer(config)

    # 保存されたモデルのパラメータをロード
    model.load_state_dict(torch.load(model_path))
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

    return model, tokenizer
