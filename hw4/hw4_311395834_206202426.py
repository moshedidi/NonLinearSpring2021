#!/usr/bin/python
# coding=utf-8
"""
Author:  moshed & eladb
Created on 22/04/2021

"""
import time

import matplotlib.pyplot as plt
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
    return lambda f, xk, gk: np.square((np.linalg.norm(gk)) / (np.linalg.norm(np.dot(A, gk))))
    # def lsearch(f, xk, gk):
    #     # return 0.5 * np.square(np.linalg.norm(gk)) / (np.dot(gk.T, A).dot(gk))
    #
    #     # return (np.dot(xk, A).dot(gk) - np.dot(gk, A).dot(xk)) / 2 * (np.dot(gk, A).dot(gk))
    #
    # return lsearch


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
    s = 1 / (2 * np.max(np.linalg.eigvals(np.dot(A.T, A))))
    x0 = np.array([1, 1, 1, 1, 1])
    eps = 1 / np.power(10, 5)

    _const = generic_grad(f, gf, const_step(s), x0, eps)
    _exact = generic_grad(f, gf, exact_quad(A), x0, eps)
    _back = generic_grad(f, gf, back(0.5, 0.5, 1), x0, eps)
    # plt.loglog(np.arange(len(_const[1])), _const[1], label="const_step")
    # plt.loglog(np.arange(len(_exact[1])), _exact[1], label="exact_quad")
    # plt.loglog(np.arange(len(_back[1])), _back[1], label="back")

    # plt.title("logarithmic difference with respect to iterations")
    # plt.legend()
    # plt.show()

    plt.loglog(np.arange(len(_const[2])), _const[2], label="const_step")
    # plt.loglog(np.arange(len(_exact[2])), _exact[2], label="exact_quad")
    plt.loglog(np.arange(len(_back[2])), _back[2], label="back")

    plt.title("Gradient norm")
    plt.legend()
    plt.show()

    plt.semilogy(_const[3], _const[2], label="const_step")
    # plt.loglog(np.arange(len(_exact[2])), _exact[2], label="exact_quad")
    plt.semilogy(_back[3], _back[2], label="back")

    plt.title("Gradient norm for time")
    plt.legend()
    plt.show()


def main():
    q1()


if __name__ == '__main__':
    main()
