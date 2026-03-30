import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from util import get_normalized_data


def error_rate(p, t):
    return np.mean(p != t)


def main():
    X, Y = get_normalized_data()
    max_iter = 20
    print_period = 10
    lr = 0.00004
    reg = 0.01

    Xtrain = X[:-1000]
    Ytrain = Y[:-1000]
    Xtest = X[-1000:]
    Ytest = Y[-1000:]

    N, D = Xtrain.shape
    batch_sz = 500
    n_batches = N // batch_sz

    M = 300
    K = 10

    # Convert to tensors
    Xtrain_t = torch.from_numpy(Xtrain).float()
    Ytrain_t = torch.from_numpy(Ytrain).long()
    Xtest_t = torch.from_numpy(Xtest).float()
    Ytest_t = torch.from_numpy(Ytest).long()

    model = nn.Sequential(
        nn.Linear(D, M),
        nn.ReLU(),
        nn.Linear(M, K),
    )

    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=reg)
    criterion = nn.CrossEntropyLoss(reduction='sum')

    LL = []
    for i in range(max_iter):
        for j in range(n_batches):
            Xbatch = Xtrain_t[j * batch_sz:(j + 1) * batch_sz]
            Ybatch = Ytrain_t[j * batch_sz:(j + 1) * batch_sz]

            optimizer.zero_grad()
            output = model(Xbatch)
            loss = criterion(output, Ybatch)
            loss.backward()
            optimizer.step()

            if j % print_period == 0:
                model.eval()
                with torch.no_grad():
                    test_output = model(Xtest_t)
                    cost_val = criterion(test_output, Ytest_t).item()
                    prediction_val = test_output.argmax(dim=1).numpy()
                model.train()

                err = error_rate(prediction_val, Ytest)
                print("Iteration: {}, Cost: {:.2f}, Error: {:.4f}".format(i, cost_val, err))
                LL.append(cost_val)

    plt.plot(LL)
    plt.show()


if __name__ == '__main__':
    main()
