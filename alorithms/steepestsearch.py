import numpy as np
import sys
from numpy.linalg import norm


def goldensectionsearch(f, interval, tol):

    phi = (np.sqrt(5) - 1) / 2

    a, b = interval[0], interval[1]
    neval = 0

    x1 = b - phi * (b - a)
    x2 = a + phi * (b - a)

    f1 = f(x1)
    f2 = f(x2)
    neval += 2

    while abs(b - a) > tol:
        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = b - phi * (b - a)
            f1 = f(x1)
            neval += 1
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + phi * (b - a)
            f2 = f(x2)
            neval += 1

    xmin = (a + b) / 2
    fmin = f(xmin)
    neval += 1

    answer_ = [xmin, fmin, neval]
    return answer_


def fH(X):
    x = X[0]
    y = X[1]
    v = (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2
    return v


def dfH(X):
    x = X[0]
    y = X[1]
    v = np.copy(X)
    v[0] = 2 * (x ** 2 + y - 11) * (2 * x) + 2 * (x + y ** 2 - 7)
    v[1] = 2 * (x ** 2 + y - 11) + 2 * (x + y ** 2 - 7) * (2 * y)
    return v


def fR(X):
    x = X[0]
    y = X[1]
    v = (1 - x) ** 2 + 100 * (y - x ** 2) ** 2
    return v


def dfR(X):
    x = X[0]
    y = X[1]
    v = np.copy(X)
    v[0] = -2 * (1 - x) + 200 * (y - x ** 2) * (- 2 * x)
    v[1] = 200 * (y - x ** 2)
    return v


def sdsearch(f, df, x0, tol):
    kmax = 1000
    k = 0
    neval = 0

    x = np.copy(x0)
    coords = []
    coords.append(np.copy(x))

    interval = [0, 1]

    while k < kmax:
        grad = df(x)
        neval += 1

        direction = -grad

        if norm(grad) < tol:
            break


        f1dim = lambda alpha: f(x + alpha * direction)

        result = goldensectionsearch(f1dim, interval, tol)
        alpha = result[0]

        x_new = x + alpha * direction

        deltaX = x_new - x

        x = x_new
        coords.append(np.copy(x))

        k += 1

        if norm(deltaX) < tol:
            break

    xmin = x
    fmin = f(xmin)

    answer_ = [xmin, fmin, neval, coords]
    return answer_