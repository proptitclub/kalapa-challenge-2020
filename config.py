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

list_drop_field_train = ["Field_3", "Field_4", "Field_13", "Field_14", "Field_35", "Field_57"]

list_drop_field_test = []

list_date_field = ["Field_1", "Field_2", "Field_5"]


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
