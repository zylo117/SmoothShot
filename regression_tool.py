import math
import numpy
from numpy import log
from numpy import exp
from scipy.optimize import curve_fit


# 线性拟合
def linefit(x, y):
    N = float(len(x))
    sx, sy, sxx, syy, sxy = 0, 0, 0, 0, 0
    for i in range(0, int(N)):
        sx += x[i]
        sy += y[i]
        sxx += x[i] * x[i]
        syy += y[i] * y[i]
        sxy += x[i] * y[i]
    a = (sy * sx / N - sxy) / (sx * sx / N - sxx)
    b = (sy - a * sx) / N
    r = abs(sy * sx / N - sxy) / math.sqrt((sxx - sx * sx / N) * (syy - sy * sy / N))
    return a, b, r


# N项式拟合
def polyfit(x, y, degree):
    return numpy.polyfit(x, y, degree)


# 对数/指数/幂数拟合

def logfunc(x, a, b):
    y = a * log(x) + b
    return y


def logfit(x, y):
    popt, pcov = curve_fit(logfunc, x, y)
    return popt, pcov


def expfunc(x, a, b, c):
    return a * exp(-b * x) + c


def expfit(x, y):
    popt, pcov = curve_fit(expfunc, x, y)
    return popt, pcov


def powerfunc(x, a, b):
    return x ** a + b


def powerfit(x, y):
    popt, pcov = curve_fit(powerfunc, x, y)
    return popt, pcov
