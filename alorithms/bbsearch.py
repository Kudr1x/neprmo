import numpy as np
import sys
from numpy.linalg import norm


def goldensectionsearch(f, interval, tol):
    phi = (np.sqrt(5) - 1) / 2

    a, b = interval[0], interval[1]
    neval = 0
    coord = []

    x1 = a + (1 - phi) * (b - a)
    x2 = a + phi * (b - a)

    f1 = f(x1)
    f2 = f(x2)
    neval += 2

    L = b - a

    while np.abs(L) > tol:
        coord.append([x1, x2, a, b])

        if f1 < f2:
            b = x2

            x2 = x1
            f2 = f1

            x1 = a + (1 - phi) * (b - a)
            f1 = f(x1)
        else:
            a = x1

            x1 = x2
            f1 = f2

            x2 = a + phi * (b - a)
            f2 = f(x2)

        neval += 1
        L = b - a

    xmin = (a + b) / 2
    fmin = f(xmin)

    return xmin


# F_ROSENBROCK is a Rosenbrock function
# 	v = F_ROSENBROCK(X)
#	INPUT ARGUMENTS:
#	X - is 2x1 vector of input variables
#	OUTPUT ARGUMENTS:
#	v is a function value

def fR(X):
    x = X[0]
    y = X[1]
    v = (1 - x) ** 2 + 100 * (y - x ** 2) ** 2
    return v


# DF_ROSENBROCK is a Rosenbrock function derivative
# 	v = DF_ROSENBROCK(X)
#	INPUT ARGUMENTS:
#	X - is 2x1 vector of input variables
#	OUTPUT ARGUMENTS:
#	v is a derivative function value

def dfR(X):
    x = X[0]
    y = X[1]
    v = np.copy(X)
    v[0] = -2 * (1 - x) + 200 * (y - x ** 2) * (- 2 * x)
    v[1] = 200 * (y - x ** 2)
    return v


def bbsearch(f, df, x0, tol):
    # BBSEARCH searches for minimum using stabilized BB1 method
    # 	answer_ = bbsearch(f, df, x0, tol)
    #   INPUT ARGUMENTS
    #   f  - objective function
    #   df - gradient
    # 	x0 - start point
    # 	tol - set for bot range and function value
    #   OUTPUT ARGUMENTS
    #   answer_ = [xmin, fmin, neval, coords]
    # 	xmin is a function minimizer
    # 	fmin = f(xmin)
    # 	neval - number of function evaluations
    #   coords - array of statistics

    k = 0
    xmin = x0
    f1 = lambda alpha: f(x0 - alpha * df(x0))
    alpha = goldensectionsearch(f1, [0, 1], tol)
    coords = []
    D = 0.1

    while True:
        xmin = x0 - alpha * df(x0)
        deltaX = xmin - x0
        dg = df(xmin) - df(x0)
        alpha = np.dot(np.transpose(deltaX), dg) / np.dot(np.transpose(dg), dg)

        ak_stab = D / norm(df(xmin))
        alpha = min(alpha, ak_stab)
        x0 = xmin
        coords.append(xmin)
        k = k + 1

        if not ((norm(deltaX) >= tol) and (k < 1000)):
            break

    answer_ = [xmin, f(xmin), k, coords]
    return answer_


import numpy as np
import sys
from numpy.linalg import norm


def goldensectionsearch(f, interval, tol):
    phi = (np.sqrt(5) - 1) / 2

    a, b = interval[0], interval[1]
    neval = 0
    coord = []

    x1 = a + (1 - phi) * (b - a)
    x2 = a + phi * (b - a)

    f1 = f(x1)
    f2 = f(x2)
    neval += 2

    L = b - a

    while np.abs(L) > tol:
        coord.append([x1, x2, a, b])

        if f1 < f2:
            b = x2

            x2 = x1
            f2 = f1

            x1 = a + (1 - phi) * (b - a)
            f1 = f(x1)
        else:
            a = x1

            x1 = x2
            f1 = f2

            x2 = a + phi * (b - a)
            f2 = f(x2)

        neval += 1
        L = b - a

    xmin = (a + b) / 2
    fmin = f(xmin)

    return xmin


# F_ROSENBROCK is a Rosenbrock function
# 	v = F_ROSENBROCK(X)
#	INPUT ARGUMENTS:
#	X - is 2x1 vector of input variables
#	OUTPUT ARGUMENTS:
#	v is a function value

def fR(X):
    x = X[0]
    y = X[1]
    v = (1 - x) ** 2 + 100 * (y - x ** 2) ** 2
    return v


# DF_ROSENBROCK is a Rosenbrock function derivative
# 	v = DF_ROSENBROCK(X)
#	INPUT ARGUMENTS:
#	X - is 2x1 vector of input variables
#	OUTPUT ARGUMENTS:
#	v is a derivative function value

def dfR(X):
    x = X[0]
    y = X[1]
    v = np.copy(X)
    v[0] = -2 * (1 - x) + 200 * (y - x ** 2) * (- 2 * x)
    v[1] = 200 * (y - x ** 2)
    return v


def bbsearch(f, df, x0, tol):
    # BBSEARCH searches for minimum using stabilized BB1 method
    # 	answer_ = bbsearch(f, df, x0, tol)
    #   INPUT ARGUMENTS
    #   f  - objective function
    #   df - gradient
    # 	x0 - start point
    # 	tol - set for bot range and function value
    #   OUTPUT ARGUMENTS
    #   answer_ = [xmin, fmin, neval, coords]
    # 	xmin is a function minimizer
    # 	fmin = f(xmin)
    # 	neval - number of function evaluations
    #   coords - array of statistics

    k = 0
    xmin = x0
    f1 = lambda alpha: f(x0 - alpha * df(x0))
    alpha = goldensectionsearch(f1, [0, 1], tol)
    coords = []
    D = 0.1

    while True:
        xmin = x0 - alpha * df(x0)
        deltaX = xmin - x0
        dg = df(xmin) - df(x0)
        alpha = np.dot(np.transpose(deltaX), dg) / np.dot(np.transpose(dg), dg)

        ak_stab = D / norm(df(xmin))
        alpha = min(alpha, ak_stab)
        x0 = xmin
        coords.append(xmin)
        k = k + 1

        if not ((norm(deltaX) >= tol) and (k < 1000)):
            break

    answer_ = [xmin, f(xmin), k, coords]
    return answer_
