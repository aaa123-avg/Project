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


def evaluate_lgbm(model, X_train, y_train, X_test, y_test):
    """评估 LightGBM 模型"""
    from sklearn.metrics import mean_squared_error, r2_score

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_mse = mean_squared_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)

    test_mse = mean_squared_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print("\nLGBM 训练集表现：")
    print(f"均方误差: {train_mse}")
    print(f"R^2 分数: {train_r2}")

    print("\nLGBM 测试集表现：")
    print(f"均方误差: {test_mse}")
    print(f"R^2 分数: {test_r2}")

    return {
        'train_mse': train_mse,
        'train_r2': train_r2,
        'test_mse': test_mse,
        'test_r2': test_r2
    }