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
    numpy.polyfit(x, y, degree)


# 对数/指数拟合

def logfunc(x, a, b):
    y = a * log(x) + b
    return y


def logfit(x, y, degree):
    results = {}
    # coeffs = numpy.polyfit(x, y, degree)
    popt, pcov = curve_fit(logfunc, x, y)
    results['polynomial'] = popt

    # r-squared
    yhat = logfunc(x, popt[0], popt[1])  # or [p(z) for z in x]
    ybar = numpy.sum(y) / len(y)  # or sum(y)/len(y)
    ssreg = numpy.sum((yhat - ybar) ** 2)  # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = numpy.sum((y - ybar) ** 2)  # or sum([ (yi - ybar)**2 for yi in y])
    results['determination'] = ssreg / sstot

    return results


def expfunc(x, a, b):
    y = a * exp(x) + b
    return y


def expfit(x, y, degree):
    results = {}
    # coeffs = numpy.polyfit(x, y, degree)
    popt, pcov = curve_fit(expfunc, x, y)
    results['polynomial'] = popt

    # r-squared
    yhat = expfunc(x, popt[0], popt[1])  # or [p(z) for z in x]
    ybar = numpy.sum(y) / len(y)  # or sum(y)/len(y)
    ssreg = numpy.sum((yhat - ybar) ** 2)  # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = numpy.sum((y - ybar) ** 2)  # or sum([ (yi - ybar)**2 for yi in y])
    results['determination'] = ssreg / sstot

    return results
