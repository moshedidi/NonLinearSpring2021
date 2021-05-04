#!/usr/bin/python
# coding=utf-8
"""
Author:  moshed
Created on 03/05/2021

"""
import time

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import eigvals

from blur import blur

dimen = 128
A, b, original_x = blur(dimen, 5, 1)


def f(x):
    return np.square(np.linalg.norm(A.dot(x) - b))


def gf(x):
    return 2 * A.T.dot(A.dot(x) - b)


def const_step(s):
    return lambda _f, xk, gk: s


def exact_quad(A):
    return lambda _f, xk, gk: 0.5 * np.square(np.linalg.norm(gk)) / np.square(np.linalg.norm(A.dot(gk)))


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


def q3_a():
    plt.figure(figsize=(6, 6))
    plt.imshow(original_x.reshape(dimen, dimen), cmap='gray')
    plt.show()
    plt.figure(figsize=(6, 6))
    plt.imshow(b.reshape(dimen, dimen), cmap='gray')
    plt.show()
    x0 = np.zeros(A.shape[0]).reshape(dimen * dimen, 1)

    plt.figure(figsize=(10, 10))  # specifying the overall grid size

    for i, j in enumerate([1, 10, 100, 1000]):
        ax = plt.subplot(2, 2, i + 1)  # the number of images in the grid is 5*5 (25)
        xk, fs, gs, ts = generic_grad(f, gf, exact_quad(A), x0, j)
        plt.imshow(xk.reshape(dimen, dimen), cmap='gray')
        ax.set_title(f'{j} iterations')

    plt.show()

    pass


def q3_b():
    plt.figure(figsize=(6, 6))
    plt.imshow(original_x.reshape(dimen, dimen), cmap='gray')
    plt.show()
    plt.figure(figsize=(6, 6))
    plt.imshow(b.reshape(dimen, dimen), cmap='gray')
    plt.show()
    x0 = np.zeros(A.shape[0]).reshape(dimen * dimen, 1)
    s = 1 / (2 * np.max(np.linalg.eigvals(A.T.dot(A).toarray())))
    plt.figure(figsize=(10, 10))  # specifying the overall grid size

    for i, j in enumerate([1, 10, 100, 1000]):
        ax = plt.subplot(2, 2, i + 1)  # the number of images in the grid is 5*5 (25)
        xk, fs, gs, ts = generic_grad(f, gf, const_step(s), x0, j)
        plt.imshow(xk.reshape(dimen, dimen), cmap='gray')
        ax.set_title(f'{j} iterations')

    plt.show()


def main():
    # q3_a()
    q3_b()


if __name__ == '__main__':
    main()
