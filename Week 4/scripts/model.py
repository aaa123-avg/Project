import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
import configuration as cfg

def train_model(X_train, y_train):
    gbdt = GradientBoostingRegressor(random_state=cfg.RANDOM_STATE)
    grid = GridSearchCV(
        estimator=gbdt,
        param_grid=cfg.GBDT_PARAM_GRID,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X_train, y_train)

    print("最优参数组合:", grid.best_params_)
    print("最优交叉验证 MSE:", -grid.best_score_)

    return grid.best_estimator_