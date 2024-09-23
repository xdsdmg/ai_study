import logistic_regression_machine as lrm
import pandas as pd
import numpy as np

DATA_FILE_PATH = "/home/zhangchi/workarea/code/ai_study/code/data/IRIS-2.csv"
TEST_FILE_PATH = "/home/zhangchi/workarea/code/ai_study/code/data/IRIS-2-test.csv"
ALPHA = 0.001
ITER_TOTAL = 100
FEATRUE_NUM = 4

if __name__ == "__main__":
    df = pd.read_csv(DATA_FILE_PATH)
    data = df.values

    m = lrm.LogisticRegressionMachine(
        data=data,
        alpha=ALPHA,
        feature_num=FEATRUE_NUM,
        iter_total=ITER_TOTAL,
    )

    res = m.batch_gradient_descent()

    df_t = pd.read_csv(TEST_FILE_PATH)
    data_t = df_t.values
    examples = np.hstack((np.ones((data_t.shape[0], 1)), data_t))
    dim = examples.shape[1] - 1

    correct = 0

    for i in range(examples.shape[0]):
        x_i = examples[i, :][: FEATRUE_NUM + 1]
        y_i = examples[i, :][dim]

        p = m.p(y_i, x_i)
        if p > 0.5:
            correct += 1
        print(p)

    print("Correct Rate: {}%".format(100 * correct / examples.shape[0]))
