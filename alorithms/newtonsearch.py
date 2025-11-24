import numpy as np
import sys
from numpy.linalg import norm
np.seterr(all='warn')


def fH(X):
    x = X[0]
    y = X[1]
    v = (x**2 + y - 11)**2 + (x + y**2 - 7)**2
    return v


def dfH(X):
    x = X[0]
    y = X[1]
    v = np.copy(X)
    v[0] = 2 * (x**2 + y - 11) * (2 * x) + 2 * (x + y**2 - 7)
    v[1] = 2 * (x**2 + y - 11) + 2 * (x + y**2 - 7) * (2 * y)

    return v


def fR(X):
    x = X[0]
    y = X[1]
    v = (1 - x)**2 + 100*(y - x**2)**2
    return v


def dfR(X):
    x = X[0]
    y = X[1]
    v = np.copy(X)
    v[0] = -2 * (1 - x) + 200 * (y - x**2)*(- 2 * x)
    v[1] = 200 * (y - x**2)
    return v




def H(x0, tol, df):
    x0 = np.asarray(x0).flatten()
    n = len(x0)
    ddf = np.zeros((n, n))
    delta = 0.1 * tol

    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0

        x_plus = x0 + delta * e_j
        x_minus = x0 - delta * e_j

        df_plus = np.asarray(df(x_plus)).flatten()
        df_minus = np.asarray(df(x_minus)).flatten()

        ddf[:, j] = (df_plus - df_minus) / (2 * delta)

    return ddf


def nsearch(f, df, x0, tol):
    kmax = 1000
    k = 0
    neval = 0

    x = np.copy(x0)
    coords = []
    coords.append(np.copy(x))

    deltaX = np.inf

    while (norm(deltaX) >= tol) and (k < kmax):
        g0 = df(x)
        neval += 1

        H0 = H(x, tol, df)

        dx = np.linalg.solve(H0, -g0)

        deltaX = dx
        x = x + dx

        coords.append(np.copy(x))

        k += 1

    xmin = x
    fmin = f(xmin)

    answer_ = [xmin, fmin, neval, coords]
    return answer_
