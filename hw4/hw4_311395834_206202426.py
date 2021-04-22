#!/usr/bin/python
# coding=utf-8
"""
Author:  moshed & eladb
Created on 22/04/2021

"""
import time

import numpy as np

A = np.array([[100, 2, 3, 4, 5],
              [6, 100, 8, 9, 10],
              [11, 12, 100, 14, 15],
              [16, 17, 18, 100, 20],
              [21, 22, 23, 24, 100]])


def f(x):
    return ((np.dot(x.T, A.T)).dot(A)).dot(x)


def gf(x):
    return 2 * ((np.dot(A.T, A)).dot(x))


def lsearch():
    return 1 / (2 * np.max(np.linalg.eigvals(np.dot(A.T, A))))


def generic_grad(f, gf, lsearch, x0, eps):
    x_k = x0
    x_k_1 = x_k - lsearch() * gf(x_k)
    fs, gs, ts = [f(x_k)], [np.linalg.norm(gf(x_k))], [time.time()]
    while np.abs(f(x_k) - f(x_k_1)) > eps:
        x_k = x_k_1
        x_k_1 = x_k - lsearch() * gf(x_k)
        fs.append(f(x_k))
        gs.append(np.linalg.norm(gf(x_k)))
        ts.append(time.time())
    return x_k, fs, gs, ts


def q1():
    print(generic_grad(f,
                       gf,
                       lsearch,
                       np.array([1, 1, 1, 1, 1]),
                       1 / np.power(10, 5)))


def main():
    q1()


if __name__ == '__main__':
    main()
