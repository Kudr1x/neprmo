import numpy as np
import sys
from numpy.linalg import norm


# F_HIMMELBLAU is a Himmelblau function
# 	v = F_HIMMELBLAU(X)
#	INPUT ARGUMENTS:
#	X - is 2x1 vector of input variables
#	OUTPUT ARGUMENTS:
#	v is a function value
def f(X):
    x = X[0]
    y = X[1]
    # Версия питона в codeboard не поддерживает метод библиотеки numpy float_power
    v = (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2
    return v


# DF_HIMMELBLAU is a Himmelblau function derivative
# 	v = DF_HIMMELBLAU(X)
#	INPUT ARGUMENTS:
#	X - is 2x1 vector of input variables
#	OUTPUT ARGUMENTS:
#	v is a derivative function value
def df(X):
    x = X[0]
    y = X[1]
    v = np.copy(X)
    v[0] = 2 * (x ** 2 + y - 11) * (2 * x) + 2 * (x + y ** 2 - 7)
    v[1] = 2 * (x ** 2 + y - 11) + 2 * (x + y ** 2 - 7) * (2 * y)
    return v


def grsearch(x0, tol):
    # GRSEARCH searches for minimum using gradient descent method
    # 	answer_ = grsearch(x0,tol)
    #   INPUT ARGUMENTS
    #	x0 - starting point
    # 	tol - set for bot range and function value
    #   OUTPUT ARGUMENTS
    #   answer_ = [xmin, fmin, neval, coords]
    # 	xmin is a function minimizer
    # 	fmin = f(xmin)
    # 	neval - number of function evaluations
    #   coords - array of x values found during optimization

    al = 0.001
    kmax = 1000

    x = np.copy(x0)
    coords = []
    coords.append(np.copy(x))

    neval = 0
    k = 0

    while k < kmax:
        grad = df(x)
        neval += 1

        x_new = x - al * grad

        deltaX = x_new - x

        coords.append(np.copy(x_new))

        if norm(deltaX) < tol:
            x = x_new
            break

        x = x_new
        k += 1

    xmin = x
    fmin = f(xmin)

    answer_ = [xmin, fmin, neval, coords]
    return answer_
# [array([2.9914098 , 2.02029624]), np.float64(0.006301617005614693), 17, [array([0, 0]), array([0.14, 0.22]), array([0.33649024, 0.49515008]), array([0.60472417, 0.83010415]), array([0.9560018 , 1.21565795]), array([1.38653012, 1.61510223]), array([1.86050388, 1.95848065]), array([2.30184867, 2.17222411]), array([2.62639423, 2.24103641]), array([2.80893203, 2.22011185]), array([2.89414973, 2.17238515]), array([2.93415135, 2.12811314]), array([2.95571635, 2.09395191]), array([2.96898325, 2.06884055]), array([2.97781302, 2.05055386]), array([2.98393759, 2.03721943]), array([2.98828338, 2.02746153]), array([2.9914098 , 2.02029624])]]
#