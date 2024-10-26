import numpy as np
from sklearn.metrics import cohen_kappa_score
from scipy.optimize import minimize

def qwk_metric_lgb(preds, train_data):
    """
    LightGBMモデル用のQWKスコアを計算。
    """
    labels = train_data.get_label()
    preds = np.round(preds).astype(int)
    score = cohen_kappa_score(labels, preds, weights='quadratic')
    return 'qwk', score, True

def qwk_metric_xgb(preds, dtrain):
    """
    XGBoostモデル用のQWKスコアを計算。
    """
    labels = dtrain.get_label()
    preds = np.round(preds).astype(int)
    score = cohen_kappa_score(labels, preds, weights='quadratic')
    return 'qwk', score

def optimize_thresholds(predictions, y_val):
    """
    QWKスコアを最大化する閾値を最適化。
    """

    def score_function(thresholds):
        thresholds = sorted(thresholds)
        preds = np.digitize(predictions, thresholds)
        return -cohen_kappa_score(y_val, preds, weights='quadratic')

    initial_thresholds = [0.5, 1.5, 2.5, 3.5]
    result = minimize(score_function, initial_thresholds, method='nelder-mead')
    return sorted(result.x)
