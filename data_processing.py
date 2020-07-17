import os
import pandas as pd 
import re 
import numpy as np 
import config as c
from multiprocessing import Pool


def drop_fields(df_train):
    for name in c.list_drop_field:
        df_train.drop(name, axis=1, inplace=True)
    return df_train

def norm_job(job_des):
    if job_des == "none" or job_des == "undefined" or job_des == c.missing_value:
        return c.missing_value
    else:
        job_des = job_des.lower() 
        job_des = job_des.replace("cn", "cong nhan")
        job_des = job_des.replace("nv", "nhan vien")
        for regex, replace in c.list_jobs.items():
            if job_des.find(regex) != -1:
                return replace
        # f = open("listCv.txt", "a")
        # f.write(job_des+"\n")
        # f.close()
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
        elif date.find(".") != -1:
            date = date.split(".")[0]
            date = date[:4]+"-"+date[4:6]+"-"+date[6:]
        elif len(date) == 8:
            date = date[:4]+"-"+date[4:6]+"-"+date[6:]
        elif date.find("-") != -1 and len(date) <= 10:
            date = date
        elif date.find("TS") != -1:
            date = date.split(" ")[0]
            date = date[:4]+"-"+date[4:6]+"-"+date[6:] 
        else:
            print(date)
            return c.missing_value
        
        if int(date.split("-")[1]) > 12 or int(date.split("-")[-1]) > 31:
            date = c.missing_value 
        elif int(date.split("-")[-1]) == 31:
            date = date.replace("31","30")
        if date[5:] == "02-29" or date[5:] == "02-30" or date[5:] == "02-31" :
            date = date.replace("02-29","02-28")
            date = date.replace("02-30","02-28")
            date = date.replace("02-31","02-28")


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

def norm_field_46(x):
    x = x.lower()
    if x.find("cti") != -1 or x.find("cong ti") != -1:
        return "CTY"
    elif x.find("ho gia dinh") != -1 or x.find("hgd") != -1 or x.find("ho gd") != -1:
        return "HGD"
    elif x.find("truong") != -1 or x.find("giao duc") != -1 :
        return "TH"
    elif x.find("dai li") != -1 or x.find("dl") != -1:
        return "DL"
    elif x.find("benh vien") != -1 or x.find("bv") != -1 or x.find("i te") != -1:
        return "BV" 
    elif x.find("trung tam dich vu") != -1 or x.find("viec lam") != -1 or x.find("that nghiep") != -1:
        return "TT" 
    elif x.find("dac biet kho khan") != -1 or x.find("ho ngheo") != -1 or x.find("ktkk") != -1 or x.find("nguoi ngheo") != -1 or x.find("can ngheo") != -1:
        return "HN" 
    elif x.find("nhan dan") != -1 or x.find("ubnd") != -1 or x.find("canbo") != -1 or x.find("sinh song") != -1 or x.find("quanli") != -1:
        return "ND"
    elif x.find("ngan hang") != -1 or x.find("tai chinh") != -1:
        return "TC"  
    elif x != c.missing_value:
        f = open("list_46.txt", "a")
        f.write(x+"\n")
        f.close()

def get_sub_date(df_train, startDateField, endDateField, outputFieldName):
    for i in range(1,len(df_train[startDateField])):
        startDate = df_train[startDateField][i]
        endDate = df_train[endDateField][i]
        if startDate == c.missing_value or endDate == c.missing_value:
            df_train[outputFieldName][i] = c.missing_value
        else:
            sub = pd.to_datetime(endDate) - pd.to_datetime(startDate)
            df_train[outputFieldName][i] = str(sub).split("days")[0]

def get_now_sub_date(df_train, startDateField, outputFieldName):
    for i in range(1,len(df_train[startDateField])):
        startDate = df_train[startDateField][i]
        endDate = "2020-07-14"
        if startDate == c.missing_value:
            df_train[outputFieldName][i] = c.missing_value
        else:
            sub = pd.to_datetime(endDate) - pd.to_datetime(startDate)
            df_train[outputFieldName][i] = str(sub).split("days")[0]

def preprocess_train_data(data_train_path, output_path):
    df_train = pd.read_csv(data_train_path)

    # fill nan value with "missing"
    for col in df_train.columns:
        df_train[col].fillna(c.missing_value, inplace=True)
    
    df_train['Field_46'] = df_train['Field_46'].apply(lambda x: remove_accent(x))
    df_train["Field_46"] = df_train["Field_46"].apply(lambda x: norm_field_46(x))

    # # split address to 2 fields (province and city)
    # df_train["province"] = df_train["diaChi"].apply(lambda x: get_province(x)) 
    # df_train["city"] = df_train["diaChi"].apply(lambda x: get_city(x))
    # df_train['city'] = df_train['city'].apply(lambda x: remove_accent(x))
    # df_train['province'] = df_train['province'].apply(lambda x: remove_accent(x))

    # df_train["province2"] = df_train["Field_49"].apply(lambda x: get_province(x)) 
    # df_train["city2"] = df_train["Field_49"].apply(lambda x: get_city(x))
    # df_train['city2'] = df_train['city2'].apply(lambda x: remove_accent(x))
    # df_train['province2'] = df_train['province2'].apply(lambda x: remove_accent(x))

    # # norm jobs
    # df_train['maCv'] = df_train['maCv'].apply(lambda x: remove_accent(x))
    # df_train['maCv'] = df_train['maCv'].apply(lambda x: norm_job(x))

    # # norm date time
    # for field_name in c.list_date_field:
    #     df_train[field_name] = df_train[field_name].apply(lambda x: norm_dateTime(x))
    # # get age
    # df_train["ngaySinh"] = df_train["ngaySinh"].apply(lambda x: get_age(x))

    # # norm field 47
    # df_train["Field_47"] = df_train["Field_47"].apply(lambda x: norm_field_47(x))    
    
    # # sub date field
    # for i in range(0,len(c.list_date_field_alone)):
    #     output_field_name = "now-"+str(c.list_date_field_alone[i])
    #     df_train[output_field_name] = ""
    #     get_now_sub_date(df_train, c.list_date_field_alone[i], output_field_name)
    #     c.list_drop_field_train.append(c.list_date_field_alone[i])

    # for i in range(0,len(c.list_date_field_start)):
    #     output_field_name = str(c.list_date_field_end[i])+"-"+str(c.list_date_field_start[i])
    #     df_train[output_field_name] = ""
    #     get_sub_date(df_train, c.list_date_field_start[i], c.list_date_field_end[i], output_field_name)

    #     c.list_drop_field_train.append(c.list_date_field_end[i])
    #     c.list_drop_field_train.append(c.list_date_field_start[i])

    # # drop fields
    # for field in c.list_drop_field_train:
    #     df_train.drop(field, axis=1, inplace=True)

    # # save csv
    # df_train.to_csv(output_path, header=True, index=False)


def preprocess_test_data(data_test_path, output_path):
    df_test = pd.read_csv(data_test_path)

    # fill nan value with "missing"
    for col in df_test.columns:
        df_test[col].fillna(c.missing_value, inplace=True)

    # split address to 2 fields (province and city)
    df_test["province"] = df_test["diaChi"].apply(lambda x: get_province(x)) 
    df_test["city"] = df_test["diaChi"].apply(lambda x: get_city(x))
    df_test['province'] = df_test['province'].apply(lambda x: remove_accent(x))

    df_test["province2"] = df_test["Field_49"].apply(lambda x: get_province(x)) 
    df_test["city2"] = df_test["Field_49"].apply(lambda x: get_city(x))
    df_test['city2'] = df_test['city2'].apply(lambda x: remove_accent(x))
    df_test['province2'] = df_test['province2'].apply(lambda x: remove_accent(x))

    # norm jobs
    df_test['maCv'] = df_test['maCv'].apply(lambda x: remove_accent(x))
    df_test['maCv'] = df_test['maCv'].apply(lambda x: norm_job(x))

    # norm date time
    for field_name in c.list_date_field:
        df_test[field_name] = df_test[field_name].apply(lambda x: norm_dateTime(x))
        
    # get age
    df_test["ngaySinh"] = df_test["ngaySinh"].apply(lambda x: get_age(x))

    # norm field 47
    df_test["Field_47"] = df_test["Field_47"].apply(lambda x: norm_field_47(x))  

    # sub date field
    for i in range(0,len(c.list_date_field_alone)):
        output_field_name = "now-"+str(c.list_date_field_alone[i])
        df_test[output_field_name] = ""
        get_now_sub_date(df_test, c.list_date_field_alone[i], output_field_name)
        c.list_drop_field_test.append(c.list_date_field_alone[i])

    for i in range(0,len(c.list_date_field_start)):
        output_field_name = str(c.list_date_field_end[i])+"-"+str(c.list_date_field_start[i])
        df_test[output_field_name] = ""
        get_sub_date(df_test, c.list_date_field_start[i], c.list_date_field_end[i], output_field_name)

        c.list_drop_field_test.append(c.list_date_field_end[i])
        c.list_drop_field_test.append(c.list_date_field_start[i])

    # drop fields
    for field in c.list_drop_field_test:
        df_test.drop(field, axis=1, inplace=True)

    # save csv
    df_test.to_csv(output_path, header=True, index=False)


if __name__ == "__main__":
    preprocess_train_data('data/raw/train.csv','data/normed/train.csv' )
    # preprocess_test_data('data/raw/test.csv','data/normed/test.csv')
