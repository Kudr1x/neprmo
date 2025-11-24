import numpy as np
import sys
from numpy.linalg import norm
from numpy.linalg import inv



# F_HIMMELBLAU is a Himmelblau function
# 	v = F_HIMMELBLAU(X)
#	INPUT ARGUMENTS:
#	X - is 2x1 vector of input variables
#	OUTPUT ARGUMENTS:
#	v is a function value
def fH(X):
    x = X[0]
    y = X[1]
    v = (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2
    return v


# DF_HIMMELBLAU is a Himmelblau function derivative
# 	v = DF_HIMMELBLAU(X)
#	INPUT ARGUMENTS:
#	X - is 2x1 vector of input variables
#	OUTPUT ARGUMENTS:
#	v is a derivative function value

def dfH(X):
    x = X[0]
    y = X[1]
    v = np.copy(X)
    v[0] = 2 * (x ** 2 + y - 11) * (2 * x) + 2 * (x + y ** 2 - 7)
    v[1] = 2 * (x ** 2 + y - 11) + 2 * (x + y ** 2 - 7) * (2 * y)

    return v


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


def goldensectionsearch(f, interval, tol):
    a = interval[0]
    b = interval[1]
    Phi = (1 + np.sqrt(5)) / 2
    L = b - a
    x1 = b - L / Phi
    x2 = a + L / Phi
    y1 = f(x1)
    y2 = f(x2)
    neval = 2
    xmin = x1
    fmin = y1

    # main loop
    while np.abs(L) > tol:
        if y1 > y2:
            a = x1
            xmin = x2
            fmin = y2
            x1 = x2
            y1 = y2
            L = b - a
            x2 = a + L / Phi
            y2 = f(x2)
            neval += 1
        else:
            b = x2
            xmin = x1
            fmin = y1
            x2 = x1
            y2 = y1
            L = b - a
            x1 = b - L / Phi
            y1 = f(x1)
            neval += 1

    answer_ = [xmin, fmin, neval]
    return answer_


def pparam(pU, pB, tau):
    if (tau <= 1):
        p = np.dot(tau, pU)
    else:
        p = pU + (tau - 1) * (pB - pU)
    return p


def doglegsearch(mod, g0, B0, Delta, tol):
    # dogleg local search
    xcv = np.dot(-g0.transpose(), g0) / np.dot(np.dot(g0.transpose(), B0), g0)
    pU = xcv *g0
    xcvb = inv(- B0)
    pB = np.dot(inv(- B0), g0)

    func = lambda x: mod(np.dot(x, pB))
    al = goldensectionsearch(func, [-Delta / norm(pB), Delta / norm(pB)], tol)[0]
    pB = al * pB
    func_pau = lambda x: mod(pparam(pU, pB, x))
    tau = goldensectionsearch(func_pau, [0, 2], tol)[0]
    pmin = pparam(pU, pB, tau)
    if norm(pmin) > Delta:
        pmin_dop = (Delta / norm(pmin))
        pmin = np.dot(pmin_dop, pmin)
    return pmin



def trustreg(f, df, x0, tol):
    x_curr = x0.ravel()
    n = len(x_curr)

    B = np.eye(n)

    Delta = 1.0
    Delta_max = 100.0
    eta = 0.1
    kmax = 1000

    g_curr = df(x_curr)
    neval = 1

    k = 0
    coords = []
    radii = []

    dx = np.ones(n) * (tol + 1.0)

    while (norm(dx) >= tol) and (k < kmax):

        coords.append(np.copy(x_curr))
        radii.append(Delta)

        model = lambda p: np.dot(g_curr, p) + 0.5 * np.dot(p, np.dot(B, p))

        p = doglegsearch(model, g_curr, B, Delta, tol)

        f_curr = f(x_curr)
        f_next_candidate = f(x_curr + p)
        neval += 1

        actual_reduction = f_curr - f_next_candidate
        predicted_reduction = -model(p)

        rho = -1.0
        if predicted_reduction > 1e-9:
            rho = actual_reduction / predicted_reduction
        if rho < 0.25:
            Delta = 0.25 * Delta
        elif rho > 0.75 and abs(norm(p) - Delta) < 1e-6:
                Delta = min(2 * Delta, Delta_max)

        if rho > eta:
            x_next = x_curr + p
            dx = p

            g_next = df(x_next)
            neval += 1

            y_vec = g_next - g_curr
            s_vec = p

            ys = np.dot(y_vec, s_vec)

            if ys > 1e-9:
                Bs = np.dot(B, s_vec)
                term1 = np.outer(Bs, Bs) / np.dot(s_vec, Bs)
                term2 = np.outer(y_vec, y_vec) / ys
                B = B - term1 + term2

            x_curr = x_next
            g_curr = g_next

        k += 1

    xmin = x_curr
    fmin = f(xmin)

    answer_ = [xmin, fmin, neval, coords, radii]
    return answer_