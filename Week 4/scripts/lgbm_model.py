import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV
import configuration as cfg

def train_lgbm(X_train, y_train):
    """使用 GridSearchCV 训练 LightGBM 回归模型"""
    lgbm = LGBMRegressor(
        objective='regression',
        random_state=cfg.RANDOM_STATE,
        n_jobs=-1
    )

    grid = GridSearchCV(
        estimator=lgbm,
        param_grid=cfg.LGBM_PARAM_GRID,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X_train, y_train)

    print("LGBM 最优参数组合:", grid.best_params_)
    print("LGBM 最优交叉验证 MSE:", -grid.best_score_)
    return grid.best_estimator_