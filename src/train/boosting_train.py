import os
import numpy as np
from sklearn.model_selection import KFold
from scipy.stats import mode
from catboost import CatBoostRegressor
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from .qwk import qwk_metric_lgb, qwk_metric_xgb, optimize_thresholds
from utils.save_result import evaluate_and_save_results

# アンサンブル学習のメイン関数
def train_and_evaluate_with_ensemble(df, config, len_train):
    """
    KFoldでのクロスバリデーション、アンサンブル、評価、予測処理。
    """
    train_df, test_df, y, all_predictions, all_test_predictions, model_weights, kf, seeds, output_dir = preprocess_and_init(df, config, len_train)

    # 各シードでの試行
    for i, seed in enumerate(seeds):
        print(f"Running trial {i + 1}/{len(seeds)} with seed {seed}")
        np.random.seed(seed)

        # クロスバリデーションでの学習と予測
        fold_predictions, fold_thresholds = kfold_train_predict(kf, train_df, y, model_weights, config, seed)
        all_predictions[i, :] = fold_predictions

        # テストデータに対するモデル予測
        models = {model_type: train_and_predict(train_df, y, None, None, model_type, config, seed)[0]
                  for model_type in ['lightgbm', 'catboost', 'xgboost', 'random_forest']}
        all_test_predictions[i, :] = ensemble_test_predictions(test_df, model_weights, models, fold_thresholds)

    # シードごとの結果から最頻値を計算して評価
    final_predictions_seeds = mode(all_predictions, axis=0)[0].astype(int).flatten()
    final_test_predictions_seeds = mode(all_test_predictions, axis=0)[0].astype(int).flatten()
    evaluate_and_save_results(y, final_predictions_seeds, final_test_predictions_seeds, config, output_dir)

# 初期化処理
def preprocess_and_init(df, config, len_train):
    sentiment_columns = [f'{i}_prob' for i in range(config.feature_extraction.num_classes)]
    columns_to_drop = ['sentiment_analysis', 'review', 'timeToReply', 'thumbsUpCount', 
                       'review_length', 'replyContent', 'score', 'language', 'translated_review'] + sentiment_columns

    extract_df = df.drop(columns=columns_to_drop, errors='ignore')
    extract_df.info()  # データ確認
    
    # データ分割
    train_df, test_df = extract_df.iloc[:len_train].copy(), extract_df.iloc[len_train:].copy()
    y = df['score'].iloc[:len_train].copy()

    # メトリクス用の準備
    all_predictions = np.zeros((config.boosting.num_trials, len(train_df)))
    all_test_predictions = np.zeros((config.boosting.num_trials, len(test_df)))
    model_weights = {model: getattr(config.boosting.weights, model) for model in ['lightgbm', 'catboost', 'xgboost', 'random_forest']}
    seeds = [config.base.seed * (i + 1) for i in range(config.boosting.num_trials)]

    # 結果保存ディレクトリの作成
    output_dir = os.path.join(config.base.base_output_dir, 'boosting')
    os.makedirs(output_dir, exist_ok=True)

    return train_df, test_df, y, all_predictions, all_test_predictions, model_weights, KFold(n_splits=config.boosting.kfold_split, shuffle=True, random_state=config.base.seed), seeds, output_dir

# KFoldの学習と予測処理
def kfold_train_predict(kf, train_df, y, model_weights, config, seed):
    fold_predictions = np.zeros(len(train_df))
    thresholds_list = []

    for train_index, val_index in kf.split(train_df):
        X_train, X_val = train_df.iloc[train_index], train_df.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        # 各モデルでの予測
        predictions = {model_type: train_and_predict(X_train, y_train, X_val, y_val, model_type, config, seed)[1]
                       for model_type in ['lightgbm', 'catboost', 'xgboost', 'random_forest']}
        
        # アンサンブルと閾値最適化
        ensemble_pred = ensemble_predictions(predictions, model_weights)
        optimal_thresholds_fold = optimize_thresholds(ensemble_pred, y_val)
        thresholds_list.append(optimal_thresholds_fold)
        fold_predictions[val_index] = np.digitize(ensemble_pred, optimal_thresholds_fold)

    # 各foldの閾値を平均して最終的な閾値を決定
    fold_thresholds = np.mean(thresholds_list, axis=0)
    return fold_predictions, fold_thresholds

# モデルの学習と予測
def train_and_predict(X_train, y_train, X_val, y_val, model_type, config, seed):
    model, predictions = None, None
    model_config = getattr(config.boosting, model_type).__dict__.copy()
    if model_type == 'random_forest':
        model_config['random_state'] = seed
    elif model_type == 'catboost':
        model_config['random_seed'] = seed
    else:
        model_config['seed'] = seed

    if model_type == 'lightgbm':
        lgb_train = lgb.Dataset(X_train, label=y_train)
        valid_sets = [lgb.Dataset(X_val, label=y_val)] if X_val is not None else None
        num_boost_round = model_config.pop('num_boost_round')
        early_stopping_rounds = model_config.pop('early_stopping')

        if valid_sets:
            model = lgb.train(
                model_config, lgb_train, valid_sets=valid_sets, num_boost_round=num_boost_round,
                feval=qwk_metric_lgb, callbacks=[lgb.early_stopping(early_stopping_rounds)]
            )
            predictions = model.predict(X_val)
        else:
            model = lgb.train(model_config, lgb_train, num_boost_round=num_boost_round)
            predictions = None

    elif model_type == 'catboost':
        model = CatBoostRegressor(**{k: v for k, v in model_config.items() if k != 'early_stopping'})
        model.fit(X_train, y_train, eval_set=(X_val, y_val) if X_val is not None else None,
                  early_stopping_rounds=model_config['early_stopping'], verbose=False)
        predictions = model.predict(X_val) if X_val is not None else None

    elif model_type == 'xgboost':
        dtrain = xgb.DMatrix(X_train, label=y_train)
        num_boost_round = model_config.pop('num_boost_round')
        early_stopping_rounds = model_config.pop('early_stopping')

        if X_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            model = xgb.train(
                model_config, dtrain, num_boost_round=num_boost_round,
                evals=[(dval, 'validation')], custom_metric=qwk_metric_xgb,
                early_stopping_rounds=early_stopping_rounds
            )
            predictions = model.predict(dval)
        else:
            model = xgb.train(model_config, dtrain, num_boost_round=num_boost_round)
            predictions = None

    elif model_type == 'random_forest':
        model = RandomForestRegressor(**model_config)
        model.fit(X_train, y_train)
        predictions = model.predict(X_val) if X_val is not None else None

    return model, predictions

# アンサンブルの予測処理
def ensemble_predictions(predictions, weights):
    return sum(weights[model] * predictions[model] for model in predictions)

# テストデータのアンサンブル予測
def ensemble_test_predictions(test_df, model_weights, models, final_thresholds):
    predictions = {model_type: predict_test(test_df, models[model_type], model_type)
                   for model_type in models}
    ensemble_pred = ensemble_predictions(predictions, model_weights)
    return np.digitize(ensemble_pred, final_thresholds)

# 各モデルのテストデータに対する予測
def predict_test(test_df, model, model_type):
    if model_type == 'lightgbm':
        return model.predict(test_df, num_iteration=model.best_iteration)
    elif model_type == 'catboost':
        return model.predict(test_df)
    elif model_type == 'xgboost':
        return model.predict(xgb.DMatrix(test_df), iteration_range=(0, model.best_iteration)) if hasattr(model, 'best_iteration') else model.predict(xgb.DMatrix(test_df))
    elif model_type == 'random_forest':
        return model.predict(test_df)
