import multi_classification_machine as mcm
import pandas as pd
import numpy as np

DATA_FILE_PATH = "/home/zhangchi/workarea/code/ai_study/code/data/IRIS-3.csv"
TEST_FILE_PATH = "/home/zhangchi/workarea/code/ai_study/code/data/IRIS-3-test.csv"
ALPHA = 0.001
ITER_TOTAL = 1000
FEATRUE_NUM = 4
CLASS_NUM = 3

if __name__ == "__main__":
    df = pd.read_csv(DATA_FILE_PATH)
    data = df.values

    m = mcm.MultiClassificationMachine(
        data=data,
        alpha=ALPHA,
        class_num=CLASS_NUM,
        feature_num=FEATRUE_NUM,
        iter_total=ITER_TOTAL,
    )

    res = m.batch_gradient_descent()

    df_t = pd.read_csv(TEST_FILE_PATH)
    data_t = df_t.values
    examples = np.hstack((np.ones((data_t.shape[0], 1)), data_t))
    dim = examples.shape[1] - 1

    errors = 0

    for i in range(examples.shape[0]):
        x_i = examples[i, :][: FEATRUE_NUM + 1]
        y_i = examples[i, :][dim]

        p = m.p(y_i, x_i)

        for c in range(CLASS_NUM):
            if int(y_i) != c and m.p(c, x_i) > p:
                errors += 1
                break

    print(
        "Correct Rate: {}%".format(
            100 * (examples.shape[0] - errors) / examples.shape[0]
        )
    )
