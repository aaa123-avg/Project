# 配置参数
MISSING_THRESHOLD = 0.75
TEST_SIZE = 0.2
RANDOM_STATE = 42


GBDT_PARAM_GRID = {
    'learning_rate': [0.1],
    'n_estimators': [100],
    'max_depth': [5]
}

LGBM_PARAM_GRID = {
    'learning_rate': GBDT_PARAM_GRID['learning_rate'],
    'n_estimators': GBDT_PARAM_GRID['n_estimators'],
    'max_depth': GBDT_PARAM_GRID['max_depth'],
    'num_leaves': [31],
    'subsample': [1.0],
    'colsample_bytree': [1.0]
}

XGB_PARAM_GRID = {
    'learning_rate': GBDT_PARAM_GRID['learning_rate'],
    'n_estimators': GBDT_PARAM_GRID['n_estimators'],
    'max_depth': GBDT_PARAM_GRID['max_depth'],
    'subsample': [1.0],
    'colsample_bytree': [1.0]
}
