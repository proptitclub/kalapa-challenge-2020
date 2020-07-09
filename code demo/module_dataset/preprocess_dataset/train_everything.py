import pandas as pd
from sklearn import preprocessing


# le = preprocessing.LabelEncoder()
#
# le.fit(["True", "False", "True", "first", "second"])
# print(le.__dict__)
# print(le.classes_)
# print(le.transform(["True", "False", "True", "first", "second"]))
# df_train = pd.read_csv("/home/trangtv/Documents/project/creditScoring/module_dataset/dataset/raw_dataset/train.csv")
# # df_train['FIELD_2'].fillna("missing")
# df_train['maCv'].fillna("None", inplace=True)
# from nltk.util import ngrams
# a = "tarng la ta day"
# l_a = a.split(" ")
# print(l_a)
# print(l_a[:-1])
# output = list(ngrams(l_a, 2))
# print(output)

#
# from sklearn.model_selection import ParameterGrid
#
# params = {
#     'kernel': ['linear'],
#     'C': [0.5, 1, 10]
# }
# param_grid = ParameterGrid(params)
# for e_param in param_grid:
#     print(e_param)
#
# a = {'C': 0.5, 'kernel': 'linear'}
# line_write = ""
# for key, value in a.items():
#     line_write += "{}_{}_".format(key, value)
# print(line_write)
import math
a = {"trang": 2, "la": 3}
print(**a)


