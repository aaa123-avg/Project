# show_and_save.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
CSV_PATH = 'output_key.csv'  # 关键样本


def show_and_export(model, X_test, y_test, model_name: str):
    y_pred = model.predict(X_test)

    # 排序
    sort_idx = np.argsort(y_test.values)
    y_true_sorted = y_test.values[sort_idx]
    y_pred_sorted = y_pred[sort_idx]
    residual_sorted = y_true_sorted - y_pred_sorted
    x_axis = np.arange(len(y_true_sorted))

    # 1) 真实值-预测值折线
    plt.figure(figsize=(5, 3))
    plt.plot(x_axis, y_true_sorted, label='真实值', lw=1.8)
    plt.plot(x_axis, y_pred_sorted, label='预测值', lw=1.8)
    plt.fill_between(x_axis, y_true_sorted, y_pred_sorted,
                     color='gray', alpha=0.15, label='误差带')
    plt.title(f'{model_name}：真实值 vs 预测值')
    plt.xlabel('样本序号（按真实值升序）')
    plt.ylabel('High Price')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 2) 残差折线
    plt.figure(figsize=(5, 3))
    plt.plot(x_axis, residual_sorted, color='tomato', lw=1.5)
    plt.axhline(0, ls='--', c='k')
    plt.title(f'{model_name}：残差折线')
    plt.xlabel('样本序号（按真实值升序）')
    plt.ylabel('残差')
    plt.tight_layout()
    plt.show()

    # 3) 关键样本
    results = pd.DataFrame({
        'model': model_name,
        'y_true': y_test.values,
        'y_pred': y_pred,
        'abs_error': np.abs(y_test - y_pred)
    }, index=y_test.index)

    worst5 = results.nlargest(5, 'abs_error').assign(tag='worst')
    best5 = results.nsmallest(5, 'abs_error').assign(tag='best')
    median_val = results['abs_error'].quantile(0.5)
    median5 = results.iloc[(results['abs_error'] - median_val).abs().argsort()[:5]]\
                     .assign(tag='median')

    key_df = pd.concat([worst5, best5, median5]).sort_values('model')
    # 追加写 CSV
    header = not os.path.isfile(CSV_PATH)
    key_df.to_csv(CSV_PATH, mode='a', header=header, float_format='%.4f')
    print(f'【{model_name}】关键样本已追加写入 {CSV_PATH}')


def run_show_and_save(models_dict, X_test, y_test):
    for name, mdl in models_dict.items():
        show_and_export(mdl, X_test, y_test, name)