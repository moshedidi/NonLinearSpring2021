#!/usr/bin/python
# coding=utf-8
"""
Author:  moshed & eladb
Created on 22/04/2021

"""
import time

import numpy as np

# TODO: check input
A = np.array([[100, 2, 3, 4, 5],
              [6, 100, 8, 9, 10],
              [11, 12, 100, 14, 15],
              [16, 17, 18, 100, 20],
              [21, 22, 23, 24, 100]])


def f(x):
    return ((np.dot(x.T, A.T)).dot(A)).dot(x)


def gf(x):
    return 2 * ((np.dot(A.T, A)).dot(x))


''' Q1 b'''


def const_step(s):
    return lambda f, xk, gk: s


def exact_quad(A):
    np.linalg.cholesky(A)  # TODO: check
    return lambda f, xk, gk: np.square(np.linalg.norm(gk)) / (np.dot(gk.T, A).dot(gk))


def back(alpha, beta, s):
    def lsearch(f, xk, gk):
        t = s
        while f(xk - t * gk) >= f(xk) - alpha * t * np.square(np.linalg.norm(gk)):
            t *= beta
        return t

    return lsearch


def generic_grad(f, gf, lsearch, x0, eps):
    xk = x0
    xk_1 = xk - lsearch(f, xk, gf(xk)) * gf(xk)
    fs, gs, ts = [f(xk)], [np.linalg.norm(gf(xk))], [time.time()]
    while np.abs(f(xk) - f(xk_1)) > eps:
        xk = xk_1
        xk_1 = xk - lsearch(f, xk, gf(xk)) * gf(xk)
        fs.append(f(xk))
        gs.append(np.linalg.norm(gf(xk)))
        ts.append(time.time())
    return xk, fs, gs, ts


def q1():
    print(generic_grad(f,
                       gf,
                       back(alpha, beta, s),
                       np.array([1, 1, 1, 1, 1]),
                       1 / np.power(10, 5)))


def main():
    q1()


if __name__ == '__main__':
    main()
