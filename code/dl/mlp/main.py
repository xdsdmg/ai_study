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

LR = 0.1


class Timer:
    def __init__(self):
        self.start = time.time()

    def stop(self) -> float:
        return time.time() - self.start


def get_data_loader_workers():
    return 4


def load_fashion_mnist(batch_size, resize=None):
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
            num_workers=get_data_loader_workers(),
        ),
        data.DataLoader(
            dataset=mnist_test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=get_data_loader_workers(),
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


def test_load_time(batch_size):
    timer = Timer()

    train_iter, _ = load_fashion_mnist(batch_size=batch_size)

    for X, y in train_iter:
        continue

    print(f"{timer.stop():.2f} sec")


def relu(X):
    return torch.max(X, torch.zeros_like(X))


def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


class MLP:
    def __init__(self, num_inputs, num_hiddens, num_outputs):
        """
        Hyper parameters
        """
        self.num_inputs = num_inputs
        self.num_hiddens = num_hiddens
        self.num_outputs = num_outputs

        """
        Parameters
        """
        self.W1 = torch.nn.Parameter(
            torch.randn(num_inputs, num_hiddens, requires_grad=True) * 0.01
        )
        self.b1 = torch.nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
        self.W2 = torch.nn.Parameter(
            torch.randn(num_hiddens, num_outputs, requires_grad=True) * 0.01
        )
        self.b2 = torch.nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

        self.params = [self.W1, self.b1, self.W2, self.b2]

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = X.reshape((-1, self.num_inputs))
        H = relu(X @ self.W1 + self.b1)
        return H @ self.W2 + self.b2


def evaluate(model, iter, loss_func):
    iter_total = 0
    loss = 0

    correct = 0
    total = 0

    for X, y in iter:
        y_hat = model.forward(X)
        loss_ = loss_func(y_hat, y)
        loss += float(loss_.mean())
        iter_total += 1

        y_hat = y_hat.argmax(axis=1)
        cmp = y_hat == y
        correct += cmp.sum()
        total += X.shape[0]

    loss = loss / iter_total
    accuracy = correct / total

    return loss, accuracy


def train(model, train_iter, test_iter, num_epochs, lr, loss_func):
    loss_arr = []
    train_acc_arr = []
    test_acc_arr = []

    """
    Begin training
    """
    for epoch in range(num_epochs):
        for X, y in train_iter:
            l = loss_func(model.forward(X), y)
            l.sum().backward()
            sgd(model.params, lr, X.shape[0])

        """
        Evaluate training resut 
        """
        with torch.no_grad():
            # Training loss
            loss, train_acc = evaluate(model, train_iter, loss_func)

            # Testing loss
            _, test_acc = evaluate(model, test_iter, loss_func)

            print(
                f"epoch: {epoch + 1}, loss: {loss:f}, train acc: {train_acc:f}, test acc: {test_acc:f}"
            )

            loss_arr.append(loss)
            train_acc_arr.append(train_acc)
            test_acc_arr.append(test_acc)

    epochs = range(1, num_epochs + 1)
    plt.plot(epochs, loss_arr, linestyle="-", label="loss")
    plt.plot(epochs, train_acc_arr, linestyle="--", label="train acc")
    plt.plot(epochs, test_acc_arr, linestyle="-.", label="test acc")
    plt.xlabel("epoch")
    plt.legend()
    plt.grid(True)
    plt.xlim(1, num_epochs)
    plt.show()


if __name__ == "__main__":
    loss = torch.nn.CrossEntropyLoss(reduction="none")

    m = MLP(NUM_INPUTS, NUM_HIDDENS, NUM_OUTPUTS)

    train_iter, test_iter = load_fashion_mnist(BATCH_SIZE)

    train(m, train_iter, test_iter, NUM_EPOCHS, LR, loss)
