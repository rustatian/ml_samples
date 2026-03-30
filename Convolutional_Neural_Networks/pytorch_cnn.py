import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from datetime import datetime
from scipy.io import loadmat
from sklearn.utils import shuffle


def y2indicator(y):
    N = len(y)
    ind = np.zeros((N, 10))
    for i in range(N):
        ind[i, int(y[i])] = 1
    return ind


def error_rate(p, t):
    return np.mean(p != t)


def rearrange(X):
    # input is (32, 32, 3, N) from MATLAB .mat format
    # output is (N, 3, 32, 32) for PyTorch NCHW
    N = X.shape[-1]
    out = np.zeros((N, 3, 32, 32), dtype=np.float32)
    for i in range(N):
        for j in range(3):
            out[i, j, :, :] = X[:, :, j, i]
    return out / 255


class SVHN_CNN(nn.Module):
    """CNN for SVHN street view house number recognition.

    Architecture:
        Conv2d(3->20, 5x5) -> Tanh -> MaxPool(2x2) ->
        Conv2d(20->50, 5x5) -> Tanh -> MaxPool(2x2) ->
        Flatten -> Dense(1250->500) -> ReLU ->
        Dense(500->10)
    """

    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 20, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.MaxPool2d(2),
            nn.Conv2d(20, 50, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.MaxPool2d(2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(50 * 8 * 8, 500),
            nn.ReLU(),
            nn.Linear(500, 10),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device: {}".format(device))

    train = loadmat('train_32x32.mat')
    test = loadmat('test_32x32.mat')

    Xtrain = rearrange(train['X'])
    Ytrain = train['y'].flatten() - 1
    del train
    Xtrain, Ytrain = shuffle(Xtrain, Ytrain)

    Xtest = rearrange(test['X'])
    Ytest = test['y'].flatten() - 1
    del test

    # Training parameters
    max_iter = 20
    print_period = 10
    batch_sz = 500
    N = Xtrain.shape[0]
    n_batches = N // batch_sz

    # Trim to exact batch multiples
    Xtrain = Xtrain[:n_batches * batch_sz]
    Ytrain = Ytrain[:n_batches * batch_sz]

    model = SVHN_CNN().to(device)
    optimizer = optim.RMSprop(model.parameters(), lr=0.0001, alpha=0.99, momentum=0.9)
    criterion = nn.CrossEntropyLoss(reduction='sum')

    # Convert test data to tensors
    Xtest_t = torch.from_numpy(Xtest).to(device)
    Ytest_t = torch.from_numpy(Ytest).long().to(device)

    t0 = datetime.now()
    losses = []

    for i in range(max_iter):
        for j in range(n_batches):
            Xbatch = torch.from_numpy(
                Xtrain[j * batch_sz:(j + 1) * batch_sz]
            ).to(device)
            Ybatch = torch.from_numpy(
                Ytrain[j * batch_sz:(j + 1) * batch_sz]
            ).long().to(device)

            optimizer.zero_grad()
            output = model(Xbatch)
            loss = criterion(output, Ybatch)
            loss.backward()
            optimizer.step()

            if j % print_period == 0:
                model.eval()
                with torch.no_grad():
                    test_output = model(Xtest_t)
                    test_loss = criterion(test_output, Ytest_t).item()
                    prediction = test_output.argmax(dim=1).cpu().numpy()
                model.train()

                err = error_rate(prediction, Ytest)
                print("Cost / err at iteration i={}, j={}: {:.3f} / {:.3f}".format(i, j, test_loss, err))
                losses.append(test_loss)

    print("Elapsed time: {}".format(datetime.now() - t0))
    plt.plot(losses)
    plt.show()


if __name__ == '__main__':
    main()
