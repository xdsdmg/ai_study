import pandas as pd

if __name__ == "__main__":
    data_file_path = "/home/zhangchi/workarea/code/ai_study/code/data/boston.csv"
    df = pd.read_csv(data_file_path)
    print(df.head())
