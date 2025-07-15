# feature_processing.py
import pandas as pd
import configuration as cfg
from sklearn.model_selection import train_test_split
import re
import warnings
warnings.filterwarnings('ignore')

def preprocess_features(df):
    df['Date'] = pd.to_datetime(df['Date'])

    # 提取数值型和分类型列
    numeric_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(exclude=['number']).columns

    # 删除价格列
    df = df.drop(columns=['Low Price', 'Mostly High', 'Mostly Low'])

    # 独热编码
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # 去重
    df = df.drop_duplicates()

    # 删除缺失值
    df = df.dropna()

    return df


def split_data(df):
    X = df.drop(columns=['High Price'])
    y = df['High Price']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg.TEST_SIZE,
        random_state=cfg.RANDOM_STATE
    )

    # 清洗列名，避免 LightGBM 报错
    clean = lambda c: re.sub(r'[^A-Za-z0-9_]+', '_', str(c))
    X_train = X_train.rename(columns=clean)
    X_test  = X_test.rename(columns=clean)

    return X_train, X_test, y_train, y_test