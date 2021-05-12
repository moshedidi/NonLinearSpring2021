#!/usr/bin/python
# coding=utf-8
"""
Author:  moshed & eladb
Created on 07/05/2021

"""

import numpy as np
from numpy.linalg import eigvals


def f(A, b):
    return lambda x: - sum([np.log(b[i] - A[i].T.dot(x)) for i in range(len(b))])


def df(A, b):
    return lambda x: -A.T.dot(-1 * np.array([1 / (b[i] - A[i].T.dot(x)) for i in range(len(b))]))


def ddf(A, b):
    return lambda x: A.T.dot(np.diag(np.array([1 / np.square(b[i] - A[i].T.dot(x)) for i in range(len(b))]))).dot(A)


def armijo_newton(f, df, ddf, alpha, beta, s, xk, eps):
    xs, fs = [], []
    while np.linalg.norm(df(xk)) > eps:
        f_xk, df_xk = f(xk), df(xk)
        xs.append(xk.T)
        fs.append(f_xk)
        dk = np.linalg.solve(ddf(xk), -df_xk)
        tk = s
        while f(xk + tk * dk) >= f_xk + alpha * tk * df_xk.T.dot(dk):
            tk = beta * tk
        xk = xk + tk * dk
    if len(xs) == 0:
        xs, fs = [xk.T], [f(xk)]
    return xs, fs


def analytic_center(A, b, x0):
    if not all([(A[i].T.dot(x0) <= b[i]) for i in range(len(b))]):
        raise Exception('x0 is not in interior(P)')
    xs, fs = armijo_newton(f(A, b), df(A, b), ddf(A, b),
                           alpha=1 / 4,
                           beta=1 / 2,
                           s=2,
                           xk=x0,
                           eps=1 / np.power(10, 6))
    print(xs)
    print(fs)


def test_analytic_center():
    A = np.array([[2, 10],
                  [1, 0],
                  [-1, 3],
                  [-1, -1]])
    b = np.array([1, 0, 2, 2])
    x0 = np.array([-1.99, 0])
    analytic_center(A, b, x0)


def main():
    A = np.array([[2, 10]])
    b = np.array([0])
    t = f(A, b)
    print(np.isnan(t(np.array([1, 1]))))
    # test_analytic_center()


if __name__ == '__main__':
    main()
