from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import *
import xgboost as xgb
import lightgbm as lgbm


def model_sgd():
    params = {
    }

    model = LogisticRegression()
    name = "svm_model.pkl"

    d_model = {
        "model": model,
        "name_model": name,
        "params": params
    }

    return d_model


def model_random_forest():
    params = {
        "criterion": "gini",
        "n_estimators": (30, 300),
    }

    model = RandomForestClassifier()
    name = "random_forest_model.pkl"
    d_model = {
        "model": model,
        "name_model": name,
        "params": params
    }

    return d_model


def model_xgboost():
    params = {
        'max_depth': (20, 50),
        'n_estimators': (100, 500),
        'colsample_bytree': (0.8, 1),
        'subsample': (0.8, 0.1),
        'scale_pos_weight': (2, 15)
    }
    model = xgb.XGBClassifier()
    name = "xgboost_model.pkl"
    d_model = {
        "model": model,
        "name_model": name,
        "params": params
    }

    return d_model


def model_lgbm():
    params = {
        'num_leaves': [10, 30, 50, 60],
        'n_estimators': [80, 100, 120, 150, 200],
        'subsample': [0.6, 0.7, 0.8, 0.9, 0.95],
        'colsample_bytree': [0.7, 0.8, 0.9, 0.95],
        'scale_pos_weight': [2, 5, 7, 10, 15, 20],
        'drop_rate': [0.05, 0.11, 0.2, 0.3]
    }
    model = lgbm.LGBMClassifier()
    name = "lgbm_model.pkl"
    d_model = {
        "model": model,
        "name_model": name,
        "params": params
    }

    return d_model


def model_gradient_boosting():
    params = {
        "criterion": ["friedman_mse",  "mae"],
        'n_estimators': [80],
        'max_depth': [50],
        'learning_rate': [0.05],
    }

    model = GradientBoostingClassifier()
    name = "graident_boosting_model.pkl"
    d_model = {
        "model": model,
        "name_model": name,
        "params": params
    }

    return d_model


def model_linear():
    model = LinearRegression()
    name = "linear_regression_model.pkl"
    d_model = {
        "model": model,
        "name_model": name,
        "params": {}
    }

    return d_model


def model_ridge():
    model = LogisticRegression()
    name = "ridge_model.pkl"
    params = {
        'class_weight': [
            {
                0: 1,
                1: 1
            },
            {
                0: 1,
                1: 3
            },
            {
                0: 1,
                1: 5
            },
            {
                0: 1,
                1: 10
            },
        ]
    }
    d_model = {
        "model": model,
        "name_model": name,
        "params": params
    }

    return d_model


if __name__ == '__main__':
    model = lgbm.LGBMClassifier(use_missing=True)

    # model = model_xgboost['model']
    from module_dataset.preprocess_dataset.handle_dataloader import *
    x, y, a = load_data_to_numpy("/home/trangtv/Documents/project/creditScoring/module_dataset/dataset/processed_dataset/case_use_missing_none_scaler/processed_train_use_missing_none_scaler_3.csv", is_train=True)
    # model.fit(x, y)
    # print(model.predict_proba(x)[0:20])