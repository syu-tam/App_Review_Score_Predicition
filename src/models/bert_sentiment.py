import torch
import torch.nn as nn
from transformers import BertModel

class CustomSentimentModel(nn.Module):
    """
    レビューと返信の入力を基に感情分類を行うBERTベースのモデル。
    """
    def __init__(self, config):
        super(CustomSentimentModel, self).__init__()
        
        # 事前学習済みBERTモデルのロード
        self.bert = BertModel.from_pretrained(config.feature_extraction.pretrained_model_name)
        
        # ドロップアウト層と全結合層の設定
        self.dropout = nn.Dropout(p=config.feature_extraction.dropout_ratio)
        self.fc = nn.Linear(self.bert.config.hidden_size * 2, config.feature_extraction.num_classes)  # 2つのBERT出力を結合するための設定

    def forward(self, review_input_ids, review_attention_mask, reply_input_ids, reply_attention_mask):
        """
        フォワードパス: レビューと返信のBERT出力を結合し、分類スコアを返す。
        """
        # BERTの出力取得（レビューと返信）
        review_outputs = self.bert(input_ids=review_input_ids, attention_mask=review_attention_mask)
        reply_outputs = self.bert(input_ids=reply_input_ids, attention_mask=reply_attention_mask)

        # CLSトークンの出力抽出
        review_cls_output = review_outputs.last_hidden_state[:, 0, :]  # レビューのCLSトークン
        reply_cls_output = reply_outputs.last_hidden_state[:, 0, :]    # 返信のCLSトークン

        # CLSトークンの結合と分類スコア計算
        combined_output = torch.cat((review_cls_output, reply_cls_output), dim=1)  # 結合
        combined_output = self.dropout(combined_output)  # ドロップアウト適用
        logits = self.fc(combined_output)  # 最終分類スコア

        return logits
