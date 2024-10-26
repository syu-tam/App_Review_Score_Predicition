from utils.config_loader import load_all_configs
from utils.utils import set_seed
from data.dataset import load_csv
from translate.translate import translate_reviews
from train.feature_extraction_train import train_feature_extraction_model
from models.model_builder import load_trained_model
from features.feature_construct import feature_engineering
from train.boosting_train import train_and_evaluate_with_ensemble
import pandas as pd
import os
import sys
from utils.utils import print_step

def main():
    # 設定ファイルの読み込み
    print_step("Configuration Loading", "Loading configuration files...")
    config = load_all_configs()
    print("Configuration loading complete.")

    # シードの設定
    print_step("Seed Setting", f"Setting random seed to {config.base.seed}...")
    set_seed(config.base.seed)
    print("Seed setting complete.")

    # データの読み込み
    print_step("Dataset Loading", "Loading dataset...")
    df, len_train = load_csv(config)
    print("Dataset loading complete.")

    # 翻訳処理の確認
    if 'translated_review' in df.columns and config.data.translate_reviews:
        print_step("Translation Check", "Translated data already exists. Skipping translation.")
        review_column = 'translated_review'
    elif config.data.translate_reviews:
        print_step("Review Translation", "Translating reviews...")
        df = translate_reviews(df, config)
        review_column = 'translated_review'
        print("Review translation complete.")
    else:
        print_step("Translation Skipped", "Using the original reviews.")
        review_column = 'review'

    # 特徴抽出の実行またはモデルの読み込み
    if not config.feature_extraction.skip_if_files_exist:
        print_step("Feature Extraction", f"Extracting features using {config.feature_extraction.type} model...")
        model, tokenizer = train_feature_extraction_model(review_column, df[:len_train], config)
        print("Feature extraction complete.")
    elif os.path.isdir(config.feature_extraction.trained_model_path):
        print_step("Model Loading", "Loading trained feature extraction model...")
        model, tokenizer = load_trained_model(config)
        print("Model loading complete.")
    else:
        print("Error: Trained model not found. Feature extraction is required.")
        sys.exit(1)

    # 特徴量エンジニアリングの実行またはデータの読み込み
    if not os.path.isfile(config.feature_extraction.extracted_data_file):
        print_step("Feature Engineering", "Performing feature engineering...")
        extracted_features = feature_engineering(model, tokenizer, df, review_column, config)
        print("Feature engineering complete.")
    else:
        print_step("Feature Loading", "Loading existing extracted features...")
        extracted_features = pd.read_csv(config.feature_extraction.extracted_data_file, index_col=0)
        print("Feature loading complete.")

    # アンサンブルモデルの訓練と評価
    print_step("Ensemble Training and Evaluation", "Training and evaluating ensemble model...")
    train_and_evaluate_with_ensemble(extracted_features, config, len_train)
    print("Ensemble training and evaluation complete.")

if __name__ == "__main__":
    main()
