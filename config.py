patterns = {
    '[àáảãạăắằẵặẳâầấậẫẩ]': 'a',
    '[đ]': 'd',
    '[èéẻẽẹêềếểễệ]': 'e',
    '[ìíỉĩị]': 'i',
    '[òóỏõọôồốổỗộơờớởỡợ]': 'o',
    '[ùúủũụưừứửữự]': 'u',
    '[ỳýỷỹỵ]': 'y'
}

list_jobs = {
    "điện tử": 'DT', 
    "lắp ráp": 'LR', 
    "nhân viên": 'NV',
    "bán hàng": 'BH', 
    "kỹ thuật": 'KT1', 
    "bảo vệ": 'BV', 
    "lái xe": 'LX', 
    "kiểm tra": 'KT2',
    "vận hành": 'VH', 
    "kinh doanh": 'KD',
    "công nhân": 'CN',
    "giáo viên": 'GV',
    "chuyên viên": 'CV',
    "kỹ sư": 'KS',
    "y sỹ": 'BS',
    "kế toán": 'KT3', 
    "cán bộ": 'CB',
    "kỹ thuật viên": 'KTV',
    "trưởng phòng": 'TP'}

missing_value = "missing"

list_drop_field_train = ["Field_3", "Field_4", "Field_13", "Field_14", "Field_24", "Field_25", "Field_26", "Field_30", "Field_31","Field_42", 
 "Field_35", "Field_51", "Field_52", "Field_55","Field_56", "Field_57", "Field_59", "Field_61", "diaChi", "namSinh", "Field_16", "Field_17"]

list_drop_field_test = ["Field_3", "Field_4", "Field_13", "Field_14", "Field_24", "Field_25", "Field_26", "Field_30", "Field_31","Field_42", 
 "Field_35", "Field_51", "Field_52", "Field_55","Field_56", "Field_57", "Field_59", "Field_61", "diaChi", "namSinh", "Field_16", "Field_17"]

list_date_field = ["Field_1", "Field_2", "Field_5","Field_6", "Field_7", "Field_8","Field_9", "Field_15", "Field_25", "Field_32", "Field_33","Field_40", "Field_43", "Field_44",
                   "Field_34", "F_startDate","F_endDate", "E_startDate","E_endDate", 
                   "C_startDate","C_endDate","G_startDate","G_endDate","A_startDate","A_endDate"]


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
