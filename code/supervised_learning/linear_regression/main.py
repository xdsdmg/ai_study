import linear_regression_machine as lrm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

DATA_FILE_PATH = "/home/zhangchi/workarea/code/ai_study/code/data/boston.csv"
ALPHA = 0.001
ITER_TOTAL = 10
THRESHOLD = 200


def draw_loss_func(m: lrm.LinearRegressionMachine):
    begin = -100
    end = 100
    total = 101

    X, Y = np.meshgrid(np.linspace(begin, end, total), np.linspace(begin, end, total))

    Z = np.empty(X.shape)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            m.set_theta(np.array([X[i, j], Y[i, j]]))
            Z[i, j] = m.loss_func()

    fig = plt.figure()

    ax = fig.add_subplot(111, projection="3d")
    # ax.scatter(X, Y, Z, s=10)
    # ax.plot_surface(X, Y, Z)

    m.set_theta(np.array([100.0, 100.0]))
    res = m.batch_gradient_descent()
    ax.plot(res[:, 0], res[:, 1], res[:, 2], c="red")
    ax.scatter(res[:, 0], res[:, 1], res[:, 2], s=40, c="red")

    ax.set_xlabel(r"$\theta_0$")
    ax.set_ylabel(r"$\theta_1$")
    ax.set_zlabel("Loss Function")

    # plt.contour(X, Y, Z)
    # plt.colorbar()

    # plt.savefig("loss_func.png", dpi=600, format="png")

    plt.show()


if __name__ == "__main__":
    df = pd.read_csv(DATA_FILE_PATH)
    data = df.values
    data = data[:20, :]
    data[:, 0] = data[:, 5].copy()

    m = lrm.LinearRegressionMachine(
        data=data,
        alpha=ALPHA,
        feature_num=1,
        iter_total=ITER_TOTAL,
        threshold=THRESHOLD,
    )

    # point_size = 2
    # x = m.__examples[:, 1]
    # y = m.__examples[:, m.__dim]

    # plt.scatter(x, y, point_size)
    # plt.xlabel("RM")
    # plt.ylabel("MDEV")
    # plt.grid(True)

    # m.run()
    # print(m.__theta, m.loss_func())
    # x = np.linspace(5, 8, 10)
    # y = np.array([m.__theta[0] + m.__theta[1] * e for e in x])
    # plt.scatter(x, y, point_size)
    # plt.plot(x, y)

    # plt.savefig("test.png", dpi=600, format="png")

    draw_loss_func(m)
