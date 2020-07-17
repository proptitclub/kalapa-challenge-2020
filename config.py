patterns = {
    '[àáảãạăắằẵặẳâầấậẫẩ]': 'a',
    '[đ]': 'd',
    '[èéẻẽẹêềếểễệ]': 'e',
    '[ìíỉĩị]': 'i',
    '[òóỏõọôồốổỗộơờớởỡợ]': 'o',
    '[ùúủũụưừứửữự]': 'u',
    '[ỳýỷỹỵy]': 'i'
}

list_jobs = {
    "nhan vien": 'NV',
    "ban hang": 'BH', 
    "ki thuat": 'KT1', 
    "bao ve": 'BV', 
    "lai xe": 'LX', 
    "kiem tra": 'KT2',
    "van hanh": 'VH', 
    "kinh doanh": 'KD',
    "cong nhan": 'CN',
    "giao vien": 'GV',
    "chuien vien": 'CV',
    "ki su": 'KS',
    "i si": 'BS',
    "ke toan": 'KT3', 
    "can bo": 'CB',
    "ki thuat vien": 'KTV',
    "truong phong": 'TP',
    "tho": 'T',
    "giam doc": 'GD',
    "giam sat": 'GS',
    "dieu duong": 'DD',
    "lao dong": "LD",
    "quan li": "QL",
    "hieu truong": "HT",
    "thu kho": "TK",
    "tai xe": "TX",
    "duoc si":"BS",
    "bao mau": 'LD',
    "van phong": 'CB',
    "phuc vu": 'LD',
    "bi thu":'GD',
    "tro li": 'CB',
    "sua chua": 'LD'
    }

missing_value = "missing"

list_drop_field_train = ["Field_11", "Field_3", "Field_4", "Field_13", "Field_14", "Field_24", "Field_25", "Field_26", "Field_30", "Field_31","Field_42", 
 "Field_35", "Field_51", "Field_52", "Field_55","Field_56", "Field_57", "Field_59", "Field_61", "diaChi", "namSinh", "Field_16", "Field_17"]

list_drop_field_test = ["Field_11", "Field_3", "Field_4", "Field_13", "Field_14", "Field_24", "Field_25", "Field_26", "Field_30", "Field_31","Field_42", 
 "Field_35", "Field_51", "Field_52", "Field_55","Field_56", "Field_57", "Field_59", "Field_61", "diaChi", "namSinh", "Field_16", "Field_17",
 "Field_65", "Field_48"]

list_date_field = ["Field_1", "Field_2", "Field_5","Field_6", "Field_7", "Field_8","Field_9", "Field_15", "Field_32", "Field_33","Field_40", "Field_43", "Field_44",
                   "Field_34", "F_startDate","F_endDate", "E_startDate","E_endDate", 
                   "C_startDate","C_endDate","G_startDate","G_endDate","A_startDate","A_endDate"]

list_date_field_start = ["F_startDate", "E_startDate", "C_startDate", "G_startDate", "A_startDate"]
list_date_field_end = ["F_endDate", "E_endDate", "C_endDate", "G_endDate", "A_endDate"]
list_date_field_alone = ["Field_1", "Field_2", "Field_5","Field_6", "Field_7", "Field_8","Field_9", "Field_15", "Field_32", "Field_33","Field_40", "Field_43", "Field_44",
                   "Field_34"]

def get_all_province():
    import json

    result = []
    with open("district.json", 'r') as j:
        contents = json.loads(j.read())
        for province_code in contents:
            result.append(contents[province_code]['name'].lower())
    
    return result 

def get_cities_by_province(province):
    import json

    result = []
    with open("district.json", 'r') as j:
        contents = json.loads(j.read())
        for province_code in contents:
            if province == contents[province_code]['name'].lower():
                for city in contents[province_code]["cities"]:
                    result.append(contents[province_code]["cities"][city].lower())

    return result 
