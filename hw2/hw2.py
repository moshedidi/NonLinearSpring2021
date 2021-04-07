#!/usr/bin/python
# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt

"""
Author:  moshed & eladb
Created on 06/04/2021

"""


def f(x):
    return np.square(x) + (np.square(x) - 3 * x + 10) / (2 + x)


def df(x):
    return 2 * x + (np.square(x) + 4 * x - 16) / np.square(2 + x)


def ddf(x):
    return 2 + 2 / (2 + x) - 2 * (2 * x - 3) / np.square(2 + x) + 2 * (np.square(x) - 3 * x + 10) / np.power(2 + x, 3)

def generic_hybrid(f,df,ddf,l,u,eps1,eps2,k):
    pass


def generic_newton(f, df, ddf, x0, eps, k):
    fv = []
    x = x0
    counter = 0
    while counter <= k:
        fv.append(f(x))
        counter += 1
        x = x - df(x) / ddf(x)
    fv.append(f(x))
    return x, fv


def generic_bisect(f, df, l, u, eps, k):
    fv = []
    x = (u + l) / 2
    counter = 0
    while np.abs(u - l) >= eps and counter < k:
        fv.append(f(x))
        if df(u) * df(x) > 0:
            u = x
        else:
            l = x
        counter += 1
        x = (u + l) / 2
    fv.append(f(x))
    return x, fv

    print('good job')


def main():
    l = -1
    u = 5
    x0 = (u + l) / 2
    k = 50
    eps = 1 / np.power(10, 6)
    x_bisect, fv_bisect = generic_bisect(f, df, l, u, eps, k)
    plt.semilogy(np.arange(len(fv_bisect)), np.array(fv_bisect) - 3.5825439993037,
                 label="logarithmic difference with respect to iterations")
    x_newton, fv_newton = generic_newton(f, df, ddf, x0, eps, k)
    plt.semilogy(np.arange(len(fv_newton)), np.array(fv_newton) - 3.5825439993037,
                 label="logarithmic difference with respect to iterations")
    plt.show()
    print(x_bisect)


if __name__ == '__main__':
    main()
