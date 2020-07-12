import os
import pandas as pd 
import re 
import numpy as np 
import config as c

def drop_fields(df_train):
    for name in c.list_drop_field:
        df_train.drop(name, axis=1, inplace=True)
    return df_train

def norm_job(job_des):
    if job_des == "none" or job_des == "undefined" or job_des == c.missing_value:
        return c.missing_value
    else:
        job_des = job_des.lower() 
        job_des.replace("cn", "công nhân ")
        job_des.replace("nv", "nhân viên ")
        for regex, replace in c.list_jobs.items():
            if job_des.find(regex) != -1:
                return replace
        return "exception"

def remove_accent(text):
    output = text.lower()
    for regex, replace in c.patterns.items():
        output = re.sub(regex, replace, output)
        output = re.sub(regex.upper(), replace.upper(), output)
    return output

def get_province(address):
    address = address.lower()
    list_province = c.get_all_province()
    for province in list_province:
        if address.find(province) != -1:
            return province

    return c.missing_value

def get_city(address):
    address = address.lower()
    list_province = c.get_all_province()
    for province in list_province:
        if address.find(province) != -1:
            for city in c.get_cities_by_province(province):
                if address.find(city) != -1:
                    return city
    return c.missing_value         

def norm_dateTime(date):
    date = str(date)
    if date != c.missing_value:
        if date.find(":") != -1:
            date = date[:10]
        
        elif date.find("/") != -1:
            date = date.split("/")[-1]+"-"+date.split("/")[0]+"-"+date.split("/")[1]
            
        elif len(date) == 8:
            date = date[:4]+"-"+date[4:6]+"-"+date[6:]
        
        return date
    return c.missing_value

def get_age(time):
    if time != c.missing_value:
        age = 2020-int(str(time)[:4])
        if age <= 14:
            return "TN"
        elif age <= 24:
            return "HSSV"
        elif age <= 50:
            return "LD"
        else:
            return "HT"
    return c.missing_value

def norm_field_47(x):
    if x == "Zero":
        return 0
    elif x == "One":
        return 1
    elif x == "Two":
        return 2
    elif x == "Three":
        return 3
    elif x == "Four":
        return 4
    
    return c.missing_value

def preprocess_train_data(data_train_path, output_path):
    df_train = pd.read_csv(data_train_path)

    # fill nan value with "missing"
    for col in df_train.columns:
        df_train[col].fillna(c.missing_value, inplace=True)

    # split address to 2 fields (province and city)
    df_train["province"] = df_train["diaChi"].apply(lambda x: get_province(x)) 
    df_train["city"] = df_train["diaChi"].apply(lambda x: get_city(x))
    # df_train['province'] = df_train['province'].apply(lambda x: remove_accent(x))

    # norm jobs
    df_train['maCv'] = df_train['maCv'].apply(lambda x: norm_job(x))

    # norm date time
    for field_name in c.list_date_field:
        df_train[field_name] = df_train[field_name].apply(lambda x: norm_dateTime(x))

    # get age
    df_train["ngaySinh"] = df_train["ngaySinh"].apply(lambda x: get_age(x))

    # norm field 47
    df_train["Field_47"] = df_train["Field_47"].apply(lambda x: norm_field_47(x))    

    # drop fields
    for field in c.list_drop_field_train:
        df_train.drop(field, axis=1, inplace=True)

    # save csv
    df_train.to_csv(output_path, header=True, index=False)


def preprocess_test_data(data_test_path, output_path):
    df_test = pd.read_csv(data_test_path)

    # fill nan value with "missing"
    for col in df_test.columns:
        df_test[col].fillna(c.missing_value, inplace=True)

    # split address to 2 fields (province and city)
    df_test["province"] = df_test["diaChi"].apply(lambda x: get_province(x)) 
    df_test["city"] = df_test["diaChi"].apply(lambda x: get_city(x))
    # df_test['province'] = df_test['province'].apply(lambda x: remove_accent(x))

    # norm jobs
    df_test['maCv'] = df_test['maCv'].apply(lambda x: norm_job(x))

    # norm date time
    for field_name in c.list_date_field:
        df_test[field_name] = df_test[field_name].apply(lambda x: norm_dateTime(x))
        
    # get age
    df_test["ngaySinh"] = df_test["ngaySinh"].apply(lambda x: get_age(x))

    # norm field 47
    df_test["Field_47"] = df_test["Field_47"].apply(lambda x: norm_field_47(x))  

    # drop fields
    for field in c.list_drop_field_test:
        df_test.drop(field, axis=1, inplace=True)

    # save csv
    df_test.to_csv(output_path, header=True, index=False)


if __name__ == "__main__":
    preprocess_train_data('data/raw/train.csv','data/normed/train.csv' )
    preprocess_test_data('data/raw/test.csv','data/normed/test.csv')