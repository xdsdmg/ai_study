import pandas as pd


def hypotheses(theta: list[float], x: list[float]) -> float:
    if len(theta) != len(x):
        raise Exception("theta and x do not have the same dimension")


if __name__ == "__main__":
    data_file_path = "/home/zhangchi/workarea/code/ai_study/code/data/boston.csv"
    df = pd.read_csv(data_file_path)
    print(df.head())
