import numpy as np


def f(x): return x ** 2 - 10 * np.cos(0.3 * np.pi * x) - 20


def df(x): return 2 * x + 3 * np.pi * np.sin(0.3 * np.pi * x)


def ssearch(interval, tol):
    # SSEARCH searches for minimum using secant method
    # answer_ = ssearch(interval,tol)
    # INPUT ARGUMENTS
    # interval = [a, b] - search interval
    # tol - set for bot range and function value
    # OUTPUT ARGUMENTS
    # answer_ = [xmin, fmin, neval, coords]
    # xmin is a function minimizer
    # fmin = f(xmin)
    # neval - number of function evaluations
    # coords - array of x values found during optimization

    a, b = interval[0], interval[1]
    coords = []
    neval = 0

    x_old = a
    x = b

    df_old = df(x_old)
    df_x = df(x)
    neval += 2

    coords.append([x, a, b])

    while (np.abs(df_x) > tol) and (np.abs(b - a) > tol):

        x_new = x - df_x * (x - x_old) / (df_x - df_old)

        if x_new < x:
            b = x
            a = min(a, x_new)
        else:
            a = x
            b = max(b, x_new)

        x_old = x
        df_old = df_x
        x = x_new
        df_x = df(x)
        neval += 1

        coords.append([x, a, b])

    xmin = x
    fmin = f(xmin)

    answer_ = [xmin, fmin, neval, coords]
    return answer_
