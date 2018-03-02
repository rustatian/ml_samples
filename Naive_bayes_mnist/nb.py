
import numpy as np
import util
from datetime import datetime
from scipy.stats import norm
import better_exceptions
from scipy.stats import multivariate_normal as mvn


class NaiveBayers(object):
    def __init__(self):
        # Gaussian deviation
        self.gaussians = dict()
        # Class priors
        self.priors = dict()

    def fit(self, X, Y, smoothing=10e-3):
        N, D = X.shape
        # 1,2,3,4,5,6,7,8,9,0 - is labels
        labels = set(Y)

        for c in labels:
            # get the current slice [0:number] where X in our class
            current_x = X[Y == c]
            # Compute mean and variance. Store in the dictionary by class key
            self.gaussians[c] = {
                'mean': current_x.mean(axis=0),
                'var': np.var(current_x.T) + smoothing,
            }
            # Simple calculate prior probability. Divide current class by all classes
            self.priors[c] = float(len(Y[Y == c])) / len(Y)

    def score(self, X, Y):
        # Get the predictions
        P = self.predict(X)
        # Return mean of array
        return np.mean(P == Y)

    def predict(self, X):
        # N - samples, D - features (classes)
        N, D = X.shape

        # Hyperparameter (10)
        K = len(self.gaussians)

        # Fill by Zeros
        P = np.zeros((N, K))

        # for each class and mean/covariance
        for c, g in self.gaussians.items():
            mean, var = g['mean'], g['var']
            log = np.log(self.priors[c])

            # Calculate Log of the probability density function, all at once
            P[:, c] = mvn.logpdf(X, mean=mean, cov=var) + log
        return np.argmax(P, axis=1)

if __name__ == '__main__':
    # Get train data
    X, Y = util.get_data(40000)
    Ntrain = len(Y) // 2
    Xtest, Ytest = util.get_test_data(40000)

    Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
    # Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

    model = NaiveBayers()
    t0 = datetime.now()
    model.fit(Xtrain, Ytrain)

    print("Training time: ", (datetime.now() - t0))

    t0 = datetime.now()
    print("Training accuracy: ", model.score(Xtrain, Ytrain))
    print("Time to compute train accuracy: ", (datetime.now() - t0), "Train size: ", len(Ytrain))

    t0 = datetime.now()
    print("Test accuracy: ", model.score(Xtest, Ytest))
    print("Time to compute test accuracy: ", (datetime.now() - t0), "Test size: ", len(Ytest))
