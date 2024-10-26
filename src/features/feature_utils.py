import emoji
import torch
from torch.nn import functional as F
import pandas as pd
import numpy as np
import re
from sklearn.decomposition import PCA

def preprocess(df, review_column):
    """
    特定の単語を置き換え、NaNを空文字列に変換し、前処理を行う関数。
    """
    df[review_column] = df[review_column].replace(
        {'push': 'good', 'Keep': 'great', r'bagus\w*': 'good', 'challenging': 'troublesome'}, regex=True)
    df[review_column] = df[review_column].fillna('')
    df['replyContent'] = df['replyContent'].fillna('')
    return df

def replace_emoji_with_great(row, review_column):
    """
    テキスト全体が絵文字の場合、テキストを "great" に置き換える関数。
    """
    if all(emoji.is_emoji(char) for char in row['review']):
        return 'great'
    return row[review_column]

def analyze_sentiment(row, model, tokenizer, device, max_len, review_column):
    """
    BERTモデルを用いて、レビューと返信の感情スコアを分析する関数。
    """
    model.eval()
    model.to(device)

    review = row[review_column]
    reply = row['replyContent']

    review_encoding = tokenizer.encode_plus(
        review, add_special_tokens=True, max_length=max_len // 2,
        padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt'
    )
    reply_encoding = tokenizer.encode_plus(
        reply, add_special_tokens=True, max_length=max_len // 2,
        padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt'
    )

    review_input_ids = review_encoding['input_ids'].to(device)
    review_attention_mask = review_encoding['attention_mask'].to(device)
    reply_input_ids = reply_encoding['input_ids'].to(device)
    reply_attention_mask = reply_encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(
            review_input_ids=review_input_ids,
            review_attention_mask=review_attention_mask,
            reply_input_ids=reply_input_ids,
            reply_attention_mask=reply_attention_mask
        )
        scores = F.softmax(outputs, dim=1).cpu().numpy()

    return scores

def extract_sentiment_scores(sentiment_scores):
    """
    各感情スコアを列としてDataFrameに追加するための関数。
    """
    return pd.Series({f"{i}_prob": score for i, score in enumerate(sentiment_scores[0])})

def get_one_hot_sentiment_vector(sentiment_scores, num_classes):
    """
    感情スコアの最大値を持つクラスをワンホットベクトルとして返す関数。
    """
    label = np.argmax(sentiment_scores[0])
    one_hot = [0] * num_classes
    one_hot[label] = 1
    return one_hot

def extract_days(time_to_reply):
    """
    'timeToReply' 列から日数を抽出する関数。
    """
    try:
        return int(time_to_reply.split()[0])
    except Exception:
        return None
    
def add_text_features(df, review_column):
    """
    特定の単語やパターンの有無を特徴量として追加する関数。
    """
    text_features = {
        '5-star': ('replyContent', '5-star|5 stars|five-star|five stars'),
        'issues': ('replyContent', 'issues|issue|problem'),
        'sorry': ('replyContent', 'sorry|apologize'),
        'transactions': ('replyContent', 'transactions'),
        'thanks': ('replyContent', 'thanks'),
        '@BANK.co.id.': ('replyContent', '@BANK.co.id.'),
        '08121214017': ('replyContent', '08121214017'),
        'contains_question_mark': (review_column, r'\?')
    }

    for feature, (column, pattern) in text_features.items():
        df[feature] = df[column].str.contains(pattern, flags=re.IGNORECASE, regex=True).astype(int)

    return df

def apply_pca_to_sentiment(df, sentiment_columns, n_components):
    """
    感情スコアに対してPCAを適用し、指定次元数に圧縮した特徴量をデータフレームに追加する関数。
    """
    pca = PCA(n_components=n_components)
    pca_features = pca.fit_transform(df[sentiment_columns])
    
    for i in range(n_components):
        df[f'pca_{i+1}'] = pca_features[:, i]
    
    return df
