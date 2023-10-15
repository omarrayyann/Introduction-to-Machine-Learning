import numpy as np


def univariate_linear_regression(x, y):
    B1 = np.cov(x, y)[0, 1]/np.var(x)
    B0 = np.mean(y) - B1*np.mean(x)
    return [B0, B1]


def multivariate_linear_regression(x, y):
    B = np.matmul(np.linalg.inv(np.matmul(np.transpose(x), x)),
                  np.matmul(np.transpose(x), y))
    return B
