import pandas as pd


def load_data_to_numpy(path_file_data, is_train=True):
    df_data = pd.read_csv(path_file_data)
    columns_name = df_data.columns
    # print(columns_name)
    # f = open("cols.txt","a")
    # for col in columns_name:
    #     f.write(str(col)+"\n")
    # f.close()
    if is_train:
        x = df_data[df_data.columns[2:]].to_numpy()
        y = df_data['label'].to_numpy()
    else:
        x = df_data[df_data.columns[1:]].to_numpy()
        y = df_data['label'].to_numpy()
    return x, y, columns_name


if __name__ == "__main__":
    load_data_to_numpy("/home/local/WorkSpace/AI/kalapa-challenge-2020/data/train.csv")