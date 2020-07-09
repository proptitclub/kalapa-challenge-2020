import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import re
import numpy as np
import xgboost as xgb
import pickle

xgb_c = xgb.XGBClassifier()

patterns = {
    '[àáảãạăắằẵặẳâầấậẫẩ]': 'a',
    '[đ]': 'd',
    '[èéẻẽẹêềếểễệ]': 'e',
    '[ìíỉĩị]': 'i',
    '[òóỏõọôồốổỗộơờớởỡợ]': 'o',
    '[ùúủũụưừứửữự]': 'u',
    '[ỳýỷỹỵ]': 'y'
}

list_cat_field_7 = ['GD', 'TE', 'DN', 'HN', 'CH', 'TQ', 'SV', 'HS', 'GB', 'HT', 'TN', 'DK', 'XD', 'XK',
                    'BT', 'CN', 'DT', 'HC', 'KC', 'TA', 'CB', 'TK', 'TC', 'CC', 'TS', 'CK', 'NN', 'HD', 'MS',
                    'XV', 'HX', 'NO', 'PV', 'LS', 'XN', 'TB', 'QN']
l_most_freq_cv = ["điện tử", "lắp ráp", "nhân viên", "bán hàng", "kỹ thuật", "bảo vệ", "lái xe", "kiểm tra",
                 "vận hành", "kinh doanh", "linh kiện",  "máy may", "giáo viên", "công nghiệp", "sản xuất",
                 "công nhân", "thành hình", "kế toán", "cán bộ"]

missing_value = "missing"
missing_value_dis = -1
missing_value_common = -1

name_file_save_setting = "dict_setting.pkl"
name_file_save_fill_nan_discrete = "dict_map_fill_nan_discrete.pkl"
name_file_save_convert_category = "dict_map_convert_cat.pkl"
name_file_save_scaler = "model_scaler.pkl"


def remove_accent(text):
    output = text.lower()
    for regex, replace in patterns.items():
        output = re.sub(regex, replace, output)
        # deal with upper case
        output = re.sub(regex.upper(), replace.upper(), output)
        # output = unidecode.unidecode(output)
    return output


def norm_age_from_2_source(age_1, age_2, value_fill=-1):

    if age_1 != missing_value_dis and age_2 != missing_value_dis:
        if age_1 == age_2:
            return age_1
        else:
            return float((age_1 + age_2) / 2)
    else:
        if age_1 != missing_value_dis:
            return age_1
        elif age_2 != missing_value_dis:
            return age_2
        else:
            return value_fill


def norm_job_type(job_des, job_specify):
    job_des = job_des.lower()
    if job_des == "none" or job_des == "undefined":
        return 0
    else:
        job_des.replace("cn", "công nhân ")
        job_des.replace("nv", "nhân viên ")
        if job_specify in job_des:
            return 1
        return 0


def check_nan_job(job_des):
    job_des = job_des.lower()
    if job_des == "none" or job_des == "undefined":
        return 1
    return 0


def norm_field_7(value_field_7, value_specify):
    value_specify = value_specify.replace("FIELD_7_", "")
    e_field_7 = str(value_field_7)
    e_field_7 = e_field_7.replace("[", "").replace("]", "").replace(", ", ",").replace("'", "")
    l_e_field_7 = e_field_7.split(",")
    count_value_specify = l_e_field_7.count(value_specify)
    return count_value_specify


def support_convert_list_to_dict(l_type, keep_missing=True):
    if keep_missing:
        if missing_value not in l_type:
            d_label = {k: v for v, k in enumerate(l_type)}
        else:
            l_type.remove(missing_value)
            d_label = {k: v for v, k in enumerate(l_type)}
            d_label.update({missing_value: missing_value_common})
    else:
        d_label = {k: v for v, k in enumerate(l_type)}

    return d_label


def handle_missing_value_l_cat_train(x, dict_convert_cat, type_convert_cat=0):
    l_key = list(dict_convert_cat.keys())
    if x not in l_key:
        if type_convert_cat == 0:
            x = missing_value
        else:
            x = np.NaN
    return x


def preprocess_train_data(path_train_data,
                          path_save_process_source,
                          path_processed_data,
                          fill_nan_method_discrete=0, fill_nan_cat=0,
                          convert_cat=0, type_scaler=0):
    """
    :param path_data: path_train_data
    :param path_save_process_source: folder to save resource after handle
    train data for process test data
    :param path_processed_data: name file to save csv file after processed
    :param fill_nan_method_discrete:
    0: mean just fill value with missing value
    1: fill value = mean
    :param fill_nan_cat
    fill_nan_cat: 0 fill missing value with "missing" its be good for xgboost.
    fill_nan_cat: 1 fill missing value with "ffill" method in pandas
    fill_nan_cat: 2 fill missing value with missing as value not missing
    :param convert_cat
    convert_category: 0 category just convert with label encoder
    convert_category: 1 category convert with mean encoding (reflect from label)
    convert_category: 2 category convert to WoE encoding
    :param type_scaler
    type_scaler: 0 do nothing, not scaler using for decision tree based
    type_scaler: 1 min - max scaler
    type_scaler: 2 standard scaler

    :return: data for modeling
    """

    # save setting for test data
    dict_setting = {
        "fill_nan_method_discrete": fill_nan_method_discrete,
        "fill_nan_cat": fill_nan_cat,
        "convert_cat": convert_cat,
        "type_scaler": type_scaler
    }

    dict_save_fill_missing_discrete = {}
    dict_save_convert_category = {}
    dict_scaler = {}

    df_train = pd.read_csv(path_train_data)

    # process provinces just fill nan value with string "missing_location" and norm text
    df_train['province'].fillna(missing_value, inplace=True)
    df_train['province'] = df_train['province'].apply(lambda x: remove_accent(x))

    # drop district for nothing useful here
    df_train.drop('district', 1, inplace=True)

    # handle age field
    df_train['age_source1'].fillna(missing_value_dis, inplace=True)
    df_train['age_source2'].fillna(missing_value_dis, inplace=True)

    if fill_nan_method_discrete == 0:
        df_train['age_combine'] = df_train.apply(lambda x: norm_age_from_2_source(x['age_source1'], x['age_source2'],
                                                                                  value_fill=missing_value_dis), axis=1)
    else:
        df_train['age_combine'] = df_train.apply(lambda x:
                                                 norm_age_from_2_source(x['age_source1'],
                                                                        x['age_source2'],
                                                                        value_fill=df_train['age_source1'].mean()),
                                                 axis=1)
        dict_save_fill_missing_discrete['age_combine'] = df_train['age_source1'].mean()

    df_train.drop('age_source1', axis=1, inplace=True)
    df_train.drop('age_source2', axis=1, inplace=True)
    # print(df_train['age_combine'].head())

    # handle work field
    df_train['maCv'].fillna("None", inplace=True)
    # print(df_train['maCv'].value_counts().keys())
    l_cat_cv_extend = []
    for e_most_cv in l_most_freq_cv:
        l_cat_cv_extend.append('maCv_{}'.format(e_most_cv))

    for e_most_freq_cv in l_most_freq_cv:
        df_train['maCv_{}'.format(e_most_freq_cv)] = df_train['maCv'].apply(lambda x: norm_job_type(x, e_most_freq_cv))

    df_train['maCv_check_nan'] = df_train['maCv'].apply(lambda x: check_nan_job(x))

    df_train.drop('maCv', axis=1, inplace=True)

    # handle convert -1 => nan value
    df_train['FIELD_3'].replace(-1, np.NaN, inplace=True)

    # convert field 7 ['TE', 'HS', 'DN', 'DN', 'DN', 'DN']
    l_cat_field_7_extend = []
    for e_action in list_cat_field_7:
        l_cat_field_7_extend.append('FIELD_7_{}'.format(e_action))

    for e_action_extend in l_cat_field_7_extend:
        df_train[e_action_extend] = df_train['FIELD_7'].apply(lambda x: norm_field_7(x, e_action_extend))

    df_train.drop('FIELD_7', axis=1, inplace=True)

    l_category_common = ["FIELD_1", "FIELD_2", "FIELD_8", 'FIELD_9', "FIELD_10", "FIELD_12", "FIELD_13", "FIELD_14",
                         'FIELD_15', "FIELD_16", "FIELD_17", "FIELD_18", "FIELD_19", "FIELD_20", "FIELD_21",
                         "FIELD_23", "FIELD_24", "FIELD_25", "FIELD_26", "FIELD_27", "FIELD_28", "FIELD_29",
                         "FIELD_30", "FIELD_31", "FIELD_32", "FIELD_33", "FIELD_34", "FIELD_35", "FIELD_36",
                         "FIELD_37", "FIELD_38", "FIELD_39", "FIELD_40", "FIELD_41", "FIELD_42", "FIELD_43",
                         "FIELD_44", "FIELD_45", "FIELD_46", "FIELD_47", "FIELD_48", "FIELD_49"]

    l_discrete = ["FIELD_3", "FIELD_4", "FIELD_5", "FIELD_6", "FIELD_11", "FIELD_22", "FIELD_50",
                  "FIELD_51", "FIELD_52", "FIELD_53", "FIELD_54", "FIELD_55", 'FIELD_56', "FIELD_57"]

    # handle some keyword nan
    # Note some field has nan, null or none in value field 9: category nan
    # Note: field 23 just has value True
    df_train['FIELD_9'].replace("na", np.NaN, inplace=True)
    df_train['FIELD_11'].replace("None", np.NaN, inplace=True)
    df_train['FIELD_11'] = df_train['FIELD_11'].astype(float)

    # field 12 None value has different value with missing value, so we change None => None_value
    for e_field_none in l_category_common:
        df_train[e_field_none].replace("None", np.NaN, inplace=True)

    # just keep show do nothing.
    l_field_has_nan_dis = ['FIELD_54', 'FIELD_55', 'FIELD_56', 'FIELD_57']

    l_category_common.extend(["province"])
    l_category_common.extend(l_cat_cv_extend)
    l_category_common.extend(["maCv_check_nan"])

    # fill missing category
    for e_category in l_category_common:
        if fill_nan_cat == 0 or fill_nan_cat == 3:
            df_train[e_category].fillna(missing_value, inplace=True)
        elif fill_nan_cat == 1:
            df_train[e_category].fillna(method="ffill", inplace=True)
            df_train[e_category].fillna(method="bfill", inplace=True)

    # fill missing discrete
    for e_discrete in l_discrete:
        if fill_nan_method_discrete == 0:
            df_train[e_discrete].fillna(missing_value_dis, inplace=True)
        elif fill_nan_method_discrete == 1:
            df_train[e_discrete].fillna(df_train[e_discrete].mean(), inplace=True)
            # save mapping discrete for test data
            dict_save_fill_missing_discrete[e_discrete] = df_train[e_discrete].mean()

    # handle convert category data
    for e_category in l_category_common:
        if convert_cat == 0 and fill_nan_cat == 0:
            l_type_in_cat = list(df_train[e_category].value_counts().keys())
            d_label = support_convert_list_to_dict(l_type_in_cat, keep_missing=True)
            # save convert category for test data
            dict_save_convert_category[e_category] = d_label

            df_train[e_category].replace(d_label, inplace=True)

        elif convert_cat == 0 and fill_nan_cat != 0:
            l_type_in_cat = list(df_train[e_category].value_counts().keys())
            d_label = support_convert_list_to_dict(l_type_in_cat, keep_missing=False)
            # save convert category for test data
            dict_save_convert_category[e_category] = d_label

            df_train[e_category].replace(d_label, inplace=True)

        # convert cat as mean encoding
        elif convert_cat == 1:
            mean_encode = df_train.groupby(e_category)['label'].mean()
            # save encode for convert test data
            dict_save_convert_category[e_category] = mean_encode

            df_train.loc[:, '{}_m_encode'.format(e_category)] = df_train[e_category].map(mean_encode)
            df_train.drop(e_category, axis=1, inplace=True)

        # convert Weight of Evidence Encoding
        elif convert_cat == 2:
            woe_df = df_train.groupby(e_category)['label'].mean()
            woe_df = pd.DataFrame(woe_df)
            woe_df.rename(columns={'label': 'good'}, inplace=True)
            woe_df['bad'] = 1 - woe_df.good
            woe_df['bad'] = np.where(woe_df['bad'] == 0, 0.000001, woe_df['bad'])
            woe_df['woe'] = np.log(woe_df.good / woe_df.bad)
            # save woe for convert test data
            dict_save_convert_category[e_category] = woe_df

            df_train.loc[:, '{}_m_encode'.format(e_category)] = df_train[e_category].map(woe_df['woe'])
            df_train.drop(e_category, axis=1, inplace=True)

    # handle scaler data for
    l_discrete.extend(l_cat_field_7_extend)
    l_discrete.extend(['age_combine'])
    for e_discrete in l_discrete:
        # don't scaler
        if type_scaler == 0:
            pass
        # minmax scaler
        if type_scaler == 1:
            min_max_scaler = MinMaxScaler()
            df_train["{}_scaler".format(e_discrete)] = min_max_scaler.fit_transform(df_train[[e_discrete]].to_numpy())
            # save model scaler for test data
            # path_save_scaler = os.path.join(path_save_process_source, name_file_save_scaler)
            # joblib.dump(min_max_scaler, path_save_scaler)
            dict_scaler[e_discrete] = min_max_scaler

            df_train.drop(e_discrete, axis=1, inplace=True)
        # standard scaler
        elif type_scaler == 2:
            stand_scaler = StandardScaler()
            df_train["{}_scaler".format(e_discrete)] = stand_scaler.fit_transform(df_train[[e_discrete]].to_numpy())
            # save model scaler for test data
            dict_scaler[e_discrete] = stand_scaler
            # path_save_scaler = os.path.join(path_save_process_source, name_file_save_scaler)
            # joblib.dump(stand_scaler, path_save_scaler)

            df_train.drop(e_discrete, axis=1, inplace=True)

    # save to folder
    path_save_setting = os.path.join(path_save_process_source, name_file_save_setting)
    with open(path_save_setting, 'wb') as w_save_dict_setting:
        pickle.dump(dict_setting, w_save_dict_setting, pickle.HIGHEST_PROTOCOL)

    path_save_dict_fill_nan_discrete = os.path.join(path_save_process_source, name_file_save_fill_nan_discrete)
    with open(path_save_dict_fill_nan_discrete, "wb") as w_save_fill_nan_discrete:
        pickle.dump(dict_save_fill_missing_discrete, w_save_fill_nan_discrete, pickle.HIGHEST_PROTOCOL)

    path_save_dict_convert_category = os.path.join(path_save_process_source, name_file_save_convert_category)
    with open(path_save_dict_convert_category, "wb") as w_save_convert_cat:
        pickle.dump(dict_save_convert_category, w_save_convert_cat, pickle.HIGHEST_PROTOCOL)

    path_save_scaler = os.path.join(path_save_process_source, name_file_save_scaler)
    with open(path_save_scaler, "wb") as w_save_scaler:
        pickle.dump(dict_scaler, w_save_scaler, pickle.HIGHEST_PROTOCOL)

    # print(dict_save_fill_missing_discrete)
    # print(dict_save_convert_category)
    df_train.to_csv(path_processed_data, header=True, index=False)


def preprocess_test_data(path_test_data, path_save_process_source, path_processed_test_data):
    path_save_setting = os.path.join(path_save_process_source, name_file_save_setting)
    path_save_dict_fill_nan_discrete = os.path.join(path_save_process_source, name_file_save_fill_nan_discrete)
    path_save_dict_convert_category = os.path.join(path_save_process_source, name_file_save_convert_category)
    path_save_scaler = os.path.join(path_save_process_resource, name_file_save_scaler)

    with open(path_save_setting, 'rb') as r_dict_setting:
        dict_setting = pickle.load(r_dict_setting)

    with open(path_save_dict_fill_nan_discrete, "rb") as r_fill_nan_discrete:
        dict_fill_nan_discrete = pickle.load(r_fill_nan_discrete)

    with open(path_save_dict_convert_category, "rb") as r_convert_cat:
        dict_convert_cat = pickle.load(r_convert_cat)

    with open(path_save_scaler, "rb") as r_save_scaler:
        dict_scaler = pickle.load(r_save_scaler)

    df_test = pd.read_csv(path_test_data)

    # process provinces just fill nan value with string "missing_location" and norm text
    df_test['province'].fillna(missing_value, inplace=True)
    df_test['province'] = df_test['province'].apply(lambda x: remove_accent(x))

    # drop district for nothing useful here
    df_test.drop('district', 1, inplace=True)

    # handle age field
    df_test['age_source1'].fillna(missing_value_dis, inplace=True)
    df_test['age_source2'].fillna(missing_value_dis, inplace=True)

    if dict_setting['fill_nan_method_discrete'] == 0:
        df_test['age_combine'] = df_test.apply(lambda x: norm_age_from_2_source(x['age_source1'], x['age_source2'],
                                                                                  value_fill=missing_value_dis), axis=1)
    else:
        df_test['age_combine'] = df_test.apply(lambda x:
                                                 norm_age_from_2_source(x['age_source1'],
                                                                        x['age_source2'],
                                                                        value_fill=dict_fill_nan_discrete['age_combine']),
                                                 axis=1)
    df_test.drop('age_source1', axis=1, inplace=True)
    df_test.drop('age_source2', axis=1, inplace=True)

    # handle work field
    df_test['maCv'].fillna("None", inplace=True)
    # print(df_train['maCv'].value_counts().keys())
    l_cat_cv_extend = []
    for e_most_cv in l_most_freq_cv:
        l_cat_cv_extend.append('maCv_{}'.format(e_most_cv))

    for e_most_freq_cv in l_most_freq_cv:
        df_test['maCv_{}'.format(e_most_freq_cv)] = df_test['maCv'].apply(lambda x: norm_job_type(x, e_most_freq_cv))

    df_test['maCv_check_nan'] = df_test['maCv'].apply(lambda x: check_nan_job(x))

    df_test.drop('maCv', axis=1, inplace=True)

    # handle convert -1 => nan value
    df_test['FIELD_3'].replace(-1, np.NaN, inplace=True)
    # convert field 7 ['TE', 'HS', 'DN', 'DN', 'DN', 'DN']
    l_cat_field_7_extend = []
    for e_action in list_cat_field_7:
        l_cat_field_7_extend.append('FIELD_7_{}'.format(e_action))

    for e_action_extend in l_cat_field_7_extend:
        df_test[e_action_extend] = df_test['FIELD_7'].apply(lambda x: norm_field_7(x, e_action_extend))

    df_test.drop(['FIELD_7'], axis=1, inplace=True)

    l_category_common = ["FIELD_1", "FIELD_2", "FIELD_8", 'FIELD_9', "FIELD_10", "FIELD_12", "FIELD_13", "FIELD_14",
                         'FIELD_15', "FIELD_16", "FIELD_17", "FIELD_18", "FIELD_19", "FIELD_20", "FIELD_21",
                         "FIELD_23", "FIELD_24", "FIELD_25", "FIELD_26", "FIELD_27", "FIELD_28", "FIELD_29",
                         "FIELD_30", "FIELD_31", "FIELD_32", "FIELD_33", "FIELD_34", "FIELD_35", "FIELD_36",
                         "FIELD_37", "FIELD_38", "FIELD_39", "FIELD_40", "FIELD_41", "FIELD_42", "FIELD_43",
                         "FIELD_44", "FIELD_45", "FIELD_46", "FIELD_47", "FIELD_48", "FIELD_49"]

    l_discrete = ["FIELD_3", "FIELD_4", "FIELD_5", "FIELD_6", "FIELD_11", "FIELD_22", "FIELD_50",
                  "FIELD_51", "FIELD_52", "FIELD_53", "FIELD_54", "FIELD_55", 'FIELD_56', "FIELD_57"]

    # handle some keyword nan
    # Note some field has nan, null or none in value field 9: category nan
    # Note: field 23 just has value True
    df_test['FIELD_9'].replace("na", np.NaN, inplace=True)
    df_test['FIELD_11'].replace("None", np.NaN, inplace=True)
    # field 12 None value has different value with missing value, so we change None => None_value
    for e_field_none in l_category_common:
        df_test[e_field_none].replace("None", np.NaN, inplace=True)

    l_category_common.extend(["province"])
    l_category_common.extend(l_cat_cv_extend)
    l_category_common.extend(["maCv_check_nan"])

    fill_nan_cat = dict_setting['fill_nan_cat']
    for e_category in l_category_common:
        if fill_nan_cat == 0 or fill_nan_cat == 3:
            # handle convert the category not in list category save before in train
            df_test[e_category] = df_test[e_category].apply(lambda x:
                                                            handle_missing_value_l_cat_train(x,
                                                                                             dict_convert_cat[
                                                                                                 e_category],
                                                                                             type_convert_cat=0))

            df_test[e_category].fillna(missing_value, inplace=True)
        elif fill_nan_cat == 1:
            df_test[e_category] = df_test[e_category].apply(lambda x:
                                                            handle_missing_value_l_cat_train(x,
                                                                                             dict_convert_cat[
                                                                                                 e_category],
                                                                                             type_convert_cat=1))
            df_test[e_category].fillna(method="ffill", inplace=True)
            df_test[e_category].fillna(method="bfill", inplace=True)

    fill_nan_method_discrete = dict_setting['fill_nan_method_discrete']
    for e_discrete in l_discrete:
        if fill_nan_method_discrete == 0:
            df_test[e_discrete].fillna(missing_value_dis, inplace=True)
        elif fill_nan_method_discrete == 1:
            df_test[e_discrete].fillna(dict_fill_nan_discrete[e_discrete], inplace=True)

    # handle convert category data
    convert_cat = dict_setting['convert_cat']
    for e_category in l_category_common:
        if convert_cat == 0 and fill_nan_cat == 0:
            df_test[e_category].replace(dict_convert_cat[e_category], inplace=True)

        elif convert_cat == 0 and fill_nan_cat != 0:
            df_test[e_category].replace(dict_convert_cat[e_category], inplace=True)

        # convert cat as mean encoding
        elif convert_cat == 1:
            df_test.loc[:, '{}_m_encode'.format(e_category)] = df_test[e_category].map(dict_convert_cat[e_category])
            df_test.drop(e_category, axis=1, inplace=True)

        # convert Weight of Evidence Encoding
        elif convert_cat == 2:
            df_test.loc[:, '{}_m_encode'.format(e_category)] = df_test[e_category].map(dict_convert_cat[e_category])
            df_test.drop(e_category, axis=1, inplace=True)

    # handle scaler data for
    type_scaler = dict_setting['type_scaler']
    l_discrete.extend(l_cat_field_7_extend)
    l_discrete.extend(['age_combine'])
    for e_discrete in l_discrete:
        # don't scaler
        if type_scaler == 0:
            pass
        # minmax or standard scaler
        if type_scaler == 1 or type_scaler == 2:
            model_scaler = dict_scaler[e_discrete]
            df_test["{}_scaler".format(e_discrete)] = model_scaler.transform(df_test[[e_discrete]].to_numpy())
            df_test.drop(e_discrete, axis=1, inplace=True)

    df_test.to_csv(path_processed_test_data, header=True, index=False)


if __name__ == '__main__':
    path_train_data = "../dataset/raw_dataset/train.csv"
    path_save_process_resource = "../dataset/processed_dataset/case_use_missing_none_scaler/"
    path_processed_train = "../dataset/processed_dataset/case_use_missing_none_scaler/" \
                           "processed_train_use_missing_not_nan_none_scaler.csv"

    path_test_data = "../dataset/raw_dataset/test.csv"
    path_processed_test_data = "../dataset/processed_dataset/case_use_missing_none_scaler/" \
                               "processed_test_use_missing_not_nan_none_scaler.csv"

    preprocess_train_data(path_train_data,
                          path_save_process_resource,
                          path_processed_train,
                          fill_nan_method_discrete=0, fill_nan_cat=0,
                          convert_cat=0, type_scaler=0)
    preprocess_test_data(path_test_data, path_save_process_resource, path_processed_test_data)
