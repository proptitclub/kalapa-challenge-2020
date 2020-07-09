from module_dataset.preprocess_dataset.handle_dataloader import load_data_to_numpy
from module_train.model_ml import *
from module_evaluate.evaluate import draw_roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.metrics import roc_auc_score

import numpy as np
import os
import random
import torch
from statistics import mean
from bayes_opt import BayesianOptimization

l_int_parameter = ['max_depth', 'n_estimators', 'num_leaves']


def convert_dict_to_string_format(d_params):
    line_write = ""
    for key, value in d_params.items():
        if isinstance(value, str):
            pass
        else:
            if key in l_int_parameter:
                value = int(round(value))
            else:
                value = round(value, 1)
        line_write += "{}_{}_".format(key, value)
    return line_write


def split_dict_tuple_key(d_params):
    n_dict_tuple = {}
    n_dict_map_1_1 = {}
    for e_key, e_value in d_params.items():
        if isinstance(e_value, tuple):
            n_dict_tuple[e_key] = e_value
        else:
            n_dict_map_1_1[e_key] = e_value
    return n_dict_tuple, n_dict_map_1_1


def norm_dict_discrete(d_params):
    n_dict_norm = {}
    for e_key, e_value in d_params.items():
        if e_key in l_int_parameter:
            n_dict_norm[e_key] = int(round(e_value))
        else:
            n_dict_norm[e_key] = e_value
    return n_dict_norm


def plot_feature_important(ft_importances, list_columns):
    # sorted(zip(clf.feature_importances_, X.columns), reverse=True)
    feature_imp = pd.DataFrame(sorted(zip(ft_importances, list_columns)), columns=['Value', 'Feature'])

    plt.figure(figsize=(30, 20))
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.show()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)


def train_model_with_k_fold(path_data,
                            d_cf_model,
                            n_folds=5, seed=42):
    set_seed(seed)
    x, y, columns_name = load_data_to_numpy(path_data, is_train=True)
    str_k_fold = StratifiedKFold(n_splits=n_folds)

    folder_save_model = d_cf_model['folder_save_model']
    name_model = d_cf_model['name_model']

    path_log_model = os.path.join(folder_save_model, "log_train_{}.txt".format(name_model))
    w_log = open(path_log_model, "w")

    l_best_score_model = []
    for idx, (train_index, val_index) in enumerate(str_k_fold.split(x, y)):
        w_log.write("Fold {}: \n".format(idx))

        best_score_each_fold = 0

        x_train = x[train_index]
        y_train = y[train_index]

        x_val = x[val_index]
        y_val = y[val_index]

        param_grids = ParameterGrid(d_cf_model['params'])
        model = d_cf_model['model']
        for e_param in param_grids:
            model.set_params(**e_param)
            model.fit(x_train, y_train)

            y_train_predict = model.predict_proba(x_train)
            y_train_predict = y_train_predict[:, 1]
            # draw_roc_curve(y_train, y_train_predict)

            auc_score_train = roc_auc_score(y_train, y_train_predict)
            gini_train = 2 * auc_score_train - 1

            y_val_predict = model.predict_proba(x_val)
            y_val_predict = y_val_predict[:, 1]
            auc_score_test = roc_auc_score(y_val, y_val_predict)
            gini_val = 2 * auc_score_test - 1
            # draw_roc_curve(y_val, y_val_predict)

            if best_score_each_fold < gini_val:
                best_score_each_fold = gini_val

            string_params = convert_dict_to_string_format(e_param)
            name_save_model = "Fold_{}_{}_g_train_{}_g_test_{}_{}.pkl".format(
                idx,
                d_cf_model['name_model'],
                round(gini_train, 4),
                round(gini_val, 4),
                string_params
            )
            print("{}".format(name_save_model))
            w_log.write("{}\n".format(name_save_model))

            path_save_model = os.path.join(folder_save_model, name_save_model)
            # joblib.dump(model, path_save_model)

        print("End Fold {} with best gini score: {}\n".format(idx, best_score_each_fold))
        w_log.write("\nEnd Fold {} with best gini score: {}\n".format(idx, best_score_each_fold))
        l_best_score_model.append(best_score_each_fold)
    print(mean(l_best_score_model))
    w_log.close()
    return l_best_score_model


def train_with_bayes_opt(path_data, d_cf_model, n_folds=5, init_points=10, n_iters=50, seed=42):
    set_seed(seed)
    x, y, columns_name = load_data_to_numpy(path_data, is_train=True)
    # print(x[0:10])
    str_k_fold = StratifiedKFold(n_splits=n_folds)

    folder_save_model = d_cf_model['folder_save_model']
    name_model = d_cf_model['name_model']

    path_log_model = os.path.join(folder_save_model, "log_train_{}.txt".format(name_model))
    w_log = open(path_log_model, "w")

    l_best_score_model = []
    for idx, (train_index, val_index) in enumerate(str_k_fold.split(x, y)):
        w_log.write("Fold {}: \n".format(idx))

        x_train = x[train_index]
        y_train = y[train_index]

        x_val = x[val_index]
        y_val = y[val_index]

        d_result_train = {}
        d_result_test = {}

        params = d_cf_model['params']
        dict_tuple, dict_1_1 = split_dict_tuple_key(params)

        def train_e_ml(**parameters):
            model = d_cf_model['model']
            parameters.update(dict_1_1)

            parameters = norm_dict_discrete(parameters)
            model.set_params(**parameters)
            model.fit(x_train, y_train)

            # component get feature impotant
            # plot_feature_important(model.feature_importances_, columns_name)
            string_params = convert_dict_to_string_format(parameters)

            y_train_predict = model.predict_proba(x_train)
            y_train_predict = y_train_predict[:, 1]

            auc_score_train = roc_auc_score(y_train, y_train_predict)
            gini_train = 2 * auc_score_train - 1
            d_result_train[string_params] = gini_train

            y_val_predict = model.predict_proba(x_val)
            y_val_predict = y_val_predict[:, 1]

            auc_score_test = roc_auc_score(y_val, y_val_predict)
            gini_val = 2 * auc_score_test - 1
            d_result_test[string_params] = gini_val

            name_save_model = "fold_{}_{}_g_train_{}_g_test_{}_{}.pkl".format(
                idx,
                d_cf_model['name_model'],
                round(gini_train, 4),
                round(gini_val, 4),
                string_params
            )
            # print("{}\n".format(name_save_model))
            w_log.write("{}\n".format(name_save_model))
            # path_save_model = os.path.join(folder_save_model, name_save_model)
            # joblib.dump(model, path_save_model)
            return gini_val

        optimizer = BayesianOptimization(f=train_e_ml,
                                         pbounds=dict_tuple,
                                         random_state=seed)

        optimizer.maximize(init_points=init_points, n_iter=n_iters)
        best_score_each_fold = optimizer.max['target']
        str_best_params = convert_dict_to_string_format(optimizer.max['params'])

        print("\nEnd Fold {} with best gini score: {} with params: {}\n".format(idx,
                                                                                best_score_each_fold,
                                                                                str_best_params))
        l_best_score_model.append(best_score_each_fold)

    w_log.close()
    return l_best_score_model


def train_all_ml_model(path_data, l_dict_models, n_folds=5, seed=42):

    for e_d_model in l_dict_models:
        result_best_score_each_fold = train_model_with_k_fold(path_data,
                                                              e_d_model,
                                                              n_folds,
                                                              seed)

        print("Model {} has score fold: {}".format(
            e_d_model['name_model'],
            "_".join([str(round(e_score, 4)) for e_score in result_best_score_each_fold])
        ))


if __name__ == "__main__":
    path_data = "../module_dataset/dataset/processed_dataset/" \
                "case_use_missing_none_scaler/processed_train_use_missing_not_nan_none_scaler.csv"
    # xgb_model = model_xgboost()
    # # xgb_model['params'].update({"missing": np.NaN})
    # xgb_model['params'].update({"learning_rate": [0.05, 0.1, 0.15, 0.2]})
    #
    # xgb_model.update({'folder_save_model': "save_model/case_use_missing_none_scaler/xgboost"})
    # # train_with_bayes_opt(path_data,
    # #                      xgb_model,
    # #                      n_folds=5,
    # #                      init_points=10,
    # #                      n_iters=50,
    # #                      seed=42)
    # train_model_with_k_fold(path_data,
    #                         xgb_model)

    lgbm_model = model_lgbm()
    # xgb_model['params'].update({"missing": -1})
    lgbm_model['params'].update({"learning_rate": [0.05, 0.1, 0.15, 0.2, 0.3]})
    # lgbm_model['params'].update({"application": 'cross_entropy'})
    # lgbm_model['params'].update({"nthread": 4})
    lgbm_model.update({'folder_save_model': "save_model/case_use_missing_none_scaler/light_gbm"})
    # train_with_bayes_opt(path_data,
    #                      lgbm_model,
    #                      n_folds=5,
    #                      init_points=100,
    #                      n_iters=500,
    #                      seed=42)

    train_model_with_k_fold(path_data,
                            lgbm_model)

    # path_data = "../module_dataset/dataset/processed_dataset/" \
    #             "case_nan_as_mean_std_scaler/processed_train_use_nan_as_mean_std_scaler.csv"
    #
    # # svm_model = model_sgd()
    # # svm_model['params'].update({"kernel": ["linear"]})
    # # svm_model['params'].update({"loss": ["hinge", "log"]})
    # # # svm_model['params'].update({"C": [0.5, 10, 100, 1000, 500]})
    # # svm_model['params'].update({"max_iter": [1000, 2000, 5000, 10000]})
    # # svm_model.update({'folder_save_model': "save_model/case_nan_as_mean_std_scaler/svm_model"})
    # # train_model_with_k_fold(path_data,
    # #                         svm_model,
    # #                         n_folds=5,
    # #                         seed=42)
    #
    # linear_model = model_ridge()
    #
    # linear_model.update({'folder_save_model': "save_model/case_nan_as_mean_std_scaler/linear_model"})
    # train_model_with_k_fold(path_data,
    #                         d_cf_model=linear_model,
    #                         n_folds=5,
    #                         seed=42)
    #
    # # rf_model = model_random_forest()
    # # rf_model.update()

