import numpy as np

def univariate_linear_regression(x, y):
    B1 = np.cov(x, y)[0, 1]/np.var(x)
    B0 = np.mean(y) - B1*np.mean(x)
    return [B0, B1]

x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

B = univariate_linear_regression(x, y)

