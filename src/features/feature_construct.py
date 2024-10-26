from .feature_utils import (
    preprocess, replace_emoji_with_great, analyze_sentiment,
    extract_sentiment_scores, get_one_hot_sentiment_vector, 
    extract_days, add_text_features, apply_pca_to_sentiment
)
from tqdm import tqdm
import numpy as np

def feature_engineering(model, tokenizer, df, review_column, config):
    """
    指定されたデータフレームに対して特徴量エンジニアリングを行い、結果を返す関数。
    """
    device = config.base.device
    num_classes = config.feature_extraction.num_classes

    # テキストの前処理
    df = preprocess(df, review_column)
    df[review_column] = df.apply(lambda row: replace_emoji_with_great(row, review_column), axis=1)

    # 感情分析を実行し、結果を格納
    tqdm.pandas(desc="Analyzing sentiment")
    df['sentiment_analysis'] = df.progress_apply(
        lambda row: analyze_sentiment(row, model, tokenizer, device, config.feature_extraction.max_length, review_column),
        axis=1
    )

    # 感情分析結果から各クラスのスコアを抽出
    df[[f"{i}_prob" for i in range(num_classes)]] = df['sentiment_analysis'].apply(extract_sentiment_scores)

    # ワンホットベクトルを生成し、各クラスに対応する列を作成
    one_hot_vectors = df['sentiment_analysis'].apply(lambda row: get_one_hot_sentiment_vector(row, num_classes))
    for i in range(num_classes):
        df[f'sentiment_{i}'] = one_hot_vectors.apply(lambda x: x[i])

    # 日数を抽出し、返信までの日数を示す 'days_to_reply' 列を追加
    df['days_to_reply'] = df['timeToReply'].apply(extract_days)

    # 特定の単語が含まれているかを特徴量として追加
    df = add_text_features(df, review_column)

    # レビューの長さとレビューの重要度を特徴量として追加
    df['review_length'] = df[review_column].apply(len)
    df['review_importance'] = np.log(df['review_length'] + 1) * np.log(df['thumbsUpCount'] + 1)

    # PCAを使用して感情スコアの次元を削減し、最終データに追加
    sentiment_columns = [f'{i}_prob' for i in range(num_classes)]
    df = apply_pca_to_sentiment(df, sentiment_columns, n_components=config.feature_extraction.pca_component)
    
    # 加工済みデータをCSVとして保存
    df.to_csv('data/main_data.csv', index=False)

    return df
