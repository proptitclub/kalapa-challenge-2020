
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import re
import warnings
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from itertools import combinations
from datetime import datetime


param = {
    'bagging_freq': 5,
    'bagging_fraction': 0.4,
    'boost_from_average':'false',
    'boost': 'gbdt',
    'feature_fraction': 0.15,
    'learning_rate': 0.01,
    'metric':'auc',
    'num_leaves': 18,
    'num_threads': 8,
    'max_depth':12,
    'tree_learner': 'serial',
    'objective': 'binary', 
    'verbosity': 1
}

def gini(y_true, y_score):
    return roc_auc_score(y_true, y_score)*2 - 1

def lgb_gini(y_pred, dataset_true):
    y_true = dataset_true.get_label()
    return 'gini', gini(y_true, y_pred), True

folds = StratifiedKFold(n_splits=10, shuffle=False, random_state=44000)

oof = np.zeros(len(train_fe)) # lưu lại các lần predict trên tập split 1-10

predictions = np.zeros(len(test_fe)) # predict avg mỗi một model 

feature_importance_df = pd.DataFrame() # lưu lại dữ liệu để đánh giá avg feature quan trọng 

seed_train_gini = 0.

seed_val_gini = 0.

total = 0

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_fe.values, target.values)):

    print("Fold {}".format(fold_))
    # load_data format lgb
    trn_data = lgb.Dataset(train_fe.iloc[trn_idx][features], label=target.iloc[trn_idx])
    val_data = lgb.Dataset(train_fe.iloc[val_idx][features], label=target.iloc[val_idx])
    
    # số roudn train gradient tối đa, verbose_eval=200 cứ 200 it in ra màn hình kq,early_stopping_rounds=400: trong 400 rounds mà validation k tăng thì dừng
    num_round = 5000
    
    clf = lgb.train(param, trn_data, num_round,feval=lgb_gini, valid_sets = [trn_data, val_data], verbose_eval=200,early_stopping_rounds=400)
    
    oof[val_idx] = clf.predict(train_fe.iloc[val_idx][features], num_iteration=clf.best_iteration)
    
    seed_train_gini += clf.best_score["training"]["gini"] / 10.
    seed_val_gini += clf.best_score["valid_1"]["gini"] / 10.
    
    fold_importance_df = pd.DataFrame()
    
    fold_importance_df["Feature"] = features
    
    fold_importance_df["importance"] = clf.feature_importance()
    
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    predictions += clf.predict(test_fe[features], num_iteration=clf.best_iteration) / folds.n_splits
    
    print("Fold {}: {}/{}".format(fold_, clf.best_score["training"]["gini"], clf.best_score["valid_1"]["gini"]))
    total+=1
print("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))
