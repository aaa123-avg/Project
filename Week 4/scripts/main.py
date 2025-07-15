import data_analysis as da
import feature_processing as fp
import model as md
import lgbm_model as lgbm_mod
import xgb_model as xgb_mod
import evaluate as ev

def main():
    df = da.load_data()
    da.explore_data(df)
    df = da.drop_high_missing_cols(df)
    da.analyze_correlation(df)
    da.plot_categorical_vs_target(df)

    df = fp.preprocess_features(df)
    X_train, X_test, y_train, y_test = fp.split_data(df)
    print(f"训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")

    # GBDT
    gbdt = md.train_model(X_train, y_train)
    ev.evaluate_model(gbdt, X_train, y_train, X_test, y_test, "GBDT")

    # LightGBM
    lgbm = lgbm_mod.train_lgbm(X_train, y_train)
    ev.evaluate_model(lgbm, X_train, y_train, X_test, y_test, "LightGBM")

    # XGBoost
    xgb = xgb_mod.train_xgb(X_train, y_train)
    ev.evaluate_model(xgb, X_train, y_train, X_test, y_test, "XGBoost")

if __name__ == "__main__":
    main()