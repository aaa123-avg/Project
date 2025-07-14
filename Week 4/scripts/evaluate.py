# evaluate.py
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(model, X_train, y_train, X_test, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_mse = mean_squared_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)

    test_mse = mean_squared_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print("\n训练集表现：")
    print(f"均方误差: {train_mse}")
    print(f"R^2 分数: {train_r2}")

    print("\n测试集表现：")
    print(f"均方误差: {test_mse}")
    print(f"R^2 分数: {test_r2}")