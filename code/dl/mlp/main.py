import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.utils import data
import time

BATCH_SIZE = 256

NUM_INPUTS = 784
NUM_OUTPUTS = 10
NUM_HIDDENS = 256

NUM_EPOCHS = 10

LR = 0.01


class Timer:
    def __init__(self):
        self.start = time.time()

    def stop(self) -> float:
        return time.time() - self.start


def get_dataloader_workers():
    return 4


def load_data_fashion_mnist(batch_size, resize=None):
    trans = []
    trans.append(transforms.ToTensor())

    if resize:
        trans.insert(0, transforms.Resize(resize))

    trans = transforms.Compose(trans)

    mnist_train = datasets.FashionMNIST(
        root="../../data", train=True, transform=trans, download=True
    )
    mnist_test = datasets.FashionMNIST(
        root="../../data", train=False, transform=trans, download=True
    )

    return (
        data.DataLoader(
            dataset=mnist_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=get_dataloader_workers(),
        ),
        data.DataLoader(
            dataset=mnist_test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=get_dataloader_workers(),
        ),
    )


def get_fashion_minst_labels(labels):
    text_labels = [
        "t-shirt",
        "trouser",
        "pullover",
        "dress",
        "coat",
        "sandal",
        "shirt",
        "sneaker",
        "bag",
        "ankle boot",
    ]

    return [text_labels[int(i)] for i in labels]


def test_load_time():
    timer = Timer()
    batch_size = 256

    train_iter, _ = load_data_fashion_mnist(batch_size=batch_size)

    for X, y in train_iter:
        continue

    print(f"{timer.stop():.2f} sec")


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    # plt.imshow(mnist_train[0][0][0], cmap="gray")
    # plt.show()
    pass


def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)


def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


class MLP:
    def __init__(self):
        self.W1 = torch.nn.Parameter(
            torch.randn(NUM_INPUTS, NUM_HIDDENS, requires_grad=True) * 0.01
        )
        self.b1 = torch.nn.Parameter(torch.zeros(NUM_HIDDENS, requires_grad=True))
        self.W2 = torch.nn.Parameter(
            torch.randn(NUM_HIDDENS, NUM_OUTPUTS, requires_grad=True) * 0.01
        )
        self.b2 = torch.nn.Parameter(torch.zeros(NUM_OUTPUTS, requires_grad=True))

        self.params = [self.W1, self.b1, self.W2, self.b2]

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = X.reshape((-1, NUM_INPUTS))
        H = relu(X @ self.W1 + self.b1)
        return H @ self.W2 + self.b2


def train(model, train_iter, test_iter, loss):
    for epoch in range(NUM_EPOCHS):
        for X, y in train_iter:
            l = loss(model.forward(X), y)
            l.sum().backward()
            sgd(model.params, LR, BATCH_SIZE)
        with torch.no_grad():
            train_l = loss(model.forward(train_iter), test_iter)
            print(f"epoch {epoch + 1}, loss, {float(train_l.mean()):f}")


if __name__ == "__main__":
    loss = torch.nn.CrossEntropyLoss(reduction="none")
    m = MLP()
    train_iter, test_iter = load_data_fashion_mnist(BATCH_SIZE)
    train(m, train_iter, test_iter, loss)
