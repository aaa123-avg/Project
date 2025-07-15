import data_analysis as da
import feature_processing as fp
import model as md
import lgbm_model as lgbm_mod
import xgb_model as xgb_mod
import evaluate as ev
import json, warnings, sys
warnings.filterwarnings("ignore") 


def main():
    df = da.load_data()
    da.explore_data(df)
    df = da.drop_high_missing_cols(df)
    da.analyze_correlation(df)
    da.plot_categorical_vs_target(df)

    df = fp.preprocess_features(df)
    X_train, X_test, y_train, y_test = fp.split_data(df)
    print(f"训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")

    # LightGBM
    lgbm = lgbm_mod.train_lgbm(X_train, y_train)
    ev.evaluate_model(lgbm, X_train, y_train, X_test, y_test, "LightGBM")

    # GBDT
    gbdt = md.train_model(X_train, y_train)
    ev.evaluate_model(gbdt, X_train, y_train, X_test, y_test, "GBDT")

    # XGBoost
    xgb = xgb_mod.train_xgb(X_train, y_train)
    ev.evaluate_model(xgb, X_train, y_train, X_test, y_test, "XGBoost")

    metrics = {}
    for name, mdl in [ ("LightGBM", lgbm), ("GBDT", gbdt),("XGBoost", xgb)]:
        res = ev.evaluate_model(mdl, X_train, y_train, X_test, y_test, name)
        metrics[name] = {
            "train_mse": round(res["train_mse"], 2),
            "train_r2":  round(res["train_r2"],  2),
            "test_mse":  round(res["test_mse"], 2),
            "test_r2":   round(res["test_r2"],  2)
        }

    # ---------- 写入 JSON ----------
    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)
    print("训练集与测试集评估结果已保存至 output.json")

if __name__ == "__main__":
    main()