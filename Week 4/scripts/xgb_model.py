import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
import configuration as cfg


def train_xgb(X_train, y_train):
    """使用 GridSearchCV 训练 XGBoost 回归模型"""
    xgb = XGBRegressor(
        objective='reg:squarederror',
        random_state=cfg.RANDOM_STATE,
        n_jobs=-1
    )

    grid = GridSearchCV(
        estimator=xgb,
        param_grid=cfg.XGB_PARAM_GRID,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X_train, y_train)

    print("XGBoost 最优参数组合:", grid.best_params_)
    print("XGBoost 最优交叉验证 MSE:", -grid.best_score_)
    return grid.best_estimator_