#!/usr/bin/python
# coding=utf-8
"""
Author:  moshed
Created on 03/05/2021

"""
import time

import matplotlib.pyplot as plt
import numpy as np

from blur import blur

A, b, x = blur(128, 5, 1)


def f(x):
    return np.square(np.linalg.norm(np.dot(A, x) - b))


def gf(x):
    temp = np.dot(A, x) - b
    return 2 * np.dot(A.T, temp)


def exact_quad(A):
    return lambda _f, xk, gk: 0.5 * np.square(np.linalg.norm(gk)) / np.square(np.linalg.norm(np.dot(A, gk)))


def generic_grad(f, gf, lsearch, x0, num_of_iterations):
    xk = x0
    xk_1 = xk - lsearch(f, xk, gf(xk)) * gf(xk)
    fs, gs, ts = [f(xk)], [np.linalg.norm(gf(xk))], [time.time()]
    for i in range(num_of_iterations):
        xk = xk_1
        xk_1 = xk - lsearch(f, xk, gf(xk)) * gf(xk)
        fs.append(f(xk))
        gs.append(np.linalg.norm(gf(xk)))
        ts.append(time.time())
    return xk, fs, gs, ts


def q3():
    plt.figure(figsize=(6, 6))
    plt.imshow(x.reshape(128, 128), cmap='gray')
    plt.show()
    plt.figure(figsize=(6, 6))
    plt.imshow(b.reshape(128, 128), cmap='gray')
    plt.show()
    x0 = np.zeros(A.shape[0])
    xk, fs, gs, ts = generic_grad(f, gf, exact_quad(A), x0, 10)
    plt.figure(figsize=(6, 6))
    plt.imshow(xk.reshape(128, 128), cmap='gray')
    plt.show()
    pass


def main():
    q3()


if __name__ == '__main__':
    main()
