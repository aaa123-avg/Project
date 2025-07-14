# data_analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import configuration as cfg

def load_data(file_path=r'C:\DeskTop\小学期-机器学习\music_data_analysis_project\Week 4\data\US-pumpkins.csv'):
    df = pd.read_csv(file_path)
    return df

def explore_data(df):
    df.info()
    print(df.head())

def drop_high_missing_cols(df, threshold=cfg.MISSING_THRESHOLD):
    missing_ratio = df.isnull().mean()
    cols_to_keep = missing_ratio[missing_ratio < threshold].index
    cols_to_drop = missing_ratio[missing_ratio >= threshold].index
    print("Dropped columns:", cols_to_drop.tolist())
    return df[cols_to_keep]

def analyze_correlation(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if 'High Price' in numeric_cols:
        corr = df[numeric_cols].corr()['High Price'].sort_values(ascending=False)
        print("Correlation with 'High Price':")
        print(corr)

def plot_categorical_vs_target(df):
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.sans-serif'] = ['SimHei']

    for col in non_numeric_cols:
        if 'High Price' in df.columns:
            grouped = df.groupby(col)['High Price'].mean().sort_values(ascending=False)
            plt.figure(figsize=(8, 5))
            plt.plot(grouped.index, grouped.values)
            plt.xlabel(col)
            plt.ylabel('High Price平均值')
            plt.title(f'High Price平均值与{col}的关系')
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.show()
            print(f"\nAverage 'High Price' by '{col}':")
            print(grouped.head(5))