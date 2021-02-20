import os
import numpy as np
np.random.seed(101)


def load_synthetic(m, n, alpha):
    # Reference: https://github.com/andylamp/federated_pca/blob/master/synthetic_data_gen.m
    k = min(m, n)
    U, _ = np.linalg.qr(np.random.randn(m, m))
    Sigma = np.array(list(range(1, k+1))).astype(np.float32) ** -alpha
    V = np.random.randn(k, n)
    Y = (U @ np.diag(Sigma) @ V) / np.sqrt(n-1)
    yn = np.max(np.sqrt(np.sum(Y ** 2, axis=1)))
    Y /= yn
    return Y, None

