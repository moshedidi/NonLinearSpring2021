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


""" Q2 """
O = np.array([[1, 0, 0, -1, -1, -1, -1, -1, 0],
              [0, 1, -1, -1, -1, -1, -1, -1, -1],
              [1, 1, -1, -1, -1, -1, -1, -1, -1],
              [0, 0, -1, 1, -1, -1, -1, -1, -1],
              [-1, -1, -1, 1, 1, 1, 1, 1, 0],
              [-1, -1, 1, 1, 1, 0, 1, 0, 1],
              [-1, -1, 1, 1, 1, 1, 1, 1, 1],
              [-1, -1, 1, 1, 0, 1, 1, 0, 0]])


def S(X):
    res = 0
    for i in range(len(X)):
        for j in range(len(X)):
            res += np.square(np.square(np.linalg.norm(X[i, :] - X[j, :])) - D(O[i, :], O[j, :]))
    return res


def gS(X):
    res = []
    for i in range(len(X)):
        res.append(
            sum([(X[i, :] - X[j, :]) *
                 (np.square(np.linalg.norm(X[i, :] - X[j, :])) - D(O[i, :], O[j, :])) for j in range(len(X))]))
    return np.array(res)


def D(o1, o2):
    return np.linalg.norm(o1 - o2)


def q2():
    s = 1 / 1000
    x0 = np.random.rand(8, 2)
    eps = 1 / np.power(10, 5)

    _const = generic_grad(S, gS, const_step(s), x0, eps)
    plt.semilogy(np.arange(len(_const[1])), _const[1], label="const_step")

    plt.title("S values")
    plt.legend()
    plt.show()

    plt.scatter(_const[0][:, 0][:4], _const[0][:, 1][:4], label="const_step", color='blue')
    plt.scatter(_const[0][:, 0][4:], _const[0][:, 1][4:], label="const_step", color='red')

    plt.title("S values")
    plt.legend()
    plt.show()


''' Q3 '''


def f_q3(x):
    return np.square(x[0]) + np.power(x[1], 4) - np.square(x[1])


def df_q3(x):
    return np.array([2 * x[0], 4 * np.power(x[1], 3) - 2 * x[1]])


def generic_grad_noise(f, gf, lsearch, x0, eps, mu, sigma):
    xk = x0
    xk_1 = xk - lsearch(f, xk, gf(xk)) * gf(xk) + np.random.normal(loc=mu, scale=sigma, size=(len(xk)))
    fs, gs, ts = [f(xk)], [np.linalg.norm(gf(xk))], [time.time()]
    while np.abs(f(xk) - f(xk_1)) > eps:
        xk = xk_1
        xk_1 = xk - lsearch(f, xk, gf(xk)) * gf(xk)
        fs.append(f(xk))
        gs.append(np.linalg.norm(gf(xk)))
        ts.append(time.time())
    return xk, fs, gs, ts


def ex3(mu, sigma, x0, epsilon):
    x0 = np.array(x0)
    res = generic_grad(f_q3, df_q3, const_step(1 / 10), x0, epsilon)
    res_noise = generic_grad_noise(f_q3, df_q3, const_step(1 / 10), x0, epsilon, mu, sigma)
    print(res)
    print(res_noise)

    plt.loglog(np.arange(len(res[1])), res[1], label="res")
    plt.loglog(np.arange(len(res_noise[1])), res_noise[1], label="res_noise")

    plt.title("EX3")
    plt.legend()
    plt.show()


def main():
    # q1()
    # q2()
    ex3(0, 0.0005, [100, 0], 1 / np.power(10, 8))


if __name__ == '__main__':
    main()
