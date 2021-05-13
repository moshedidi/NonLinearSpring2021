#!/usr/bin/python
# coding=utf-8
"""
Author:  moshed & eladb
Created on 07/05/2021

"""
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from numpy.linalg import eigvals, LinAlgError

""" Q1 """


def f(A, b):
    return lambda x: - sum([np.log(b[i] - A[i].T.dot(x)) for i in range(len(b))])


def f3(x, y, A, b):
    return - sum([np.log(b[i] - (A[i][0] * x + A[i][1] * y)) for i in range(len(b))])


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
    return np.array(xs), np.array(fs)


def test_analytic_center():
    A = np.array([[2, 10],
                  [1, 0],
                  [-1, 3],
                  [-1, -1]])
    b = np.array([1, 0, 2, 2])
    x0 = np.array([-1.99, 0])
    xs, fs = analytic_center(A, b, x0)

    X, Y = np.meshgrid(np.arange(-2, 0, 0.01), np.arange(-2.5, 0.5, 0.01))
    Z = X.copy()

    f_xk = f(A, b)
    for i in range(len(X)):
        for j in range(len(X[0])):
            Z[i, j] = f_xk((X[i, j], Y[i, j]))

    Z[(np.isnan(Z))] = 10 ** 10
    fig, ax = plt.subplots(1, 1)
    ax.contour(X, Y, Z, levels=fs[::-1], extend='both')
    patches = []
    polygon = Polygon([[-2, 0],
                       [-1.0604, 0.3072],
                       [-0.001, 0.1],
                       [-0.001, -1.98]], True)
    # TODO: actually find the poly
    patches.append(polygon)

    p = PatchCollection(patches, alpha=0.4)
    ax.add_collection(p)
    ax.scatter(xs[:, 0], xs[:, 1], marker='*', color='red')
    ax.plot(xs[:, 0], xs[:, 1], color='black')
    plt.title("Contour of f(x) with {f_xk}")
    plt.show()

    plt.figure()
    plt.semilogy(np.arange(len(fs) - 1), np.array(fs[0:len(fs) - 1]) - fs[-1], label="f(xk) - f(xN)")
    plt.show()


""" Q2 """


def hybrid_newton(f, gf, hf, lsearch, xk, eps):
    f_xk, df_xk, hf_xk = f(xk), gf(xk), hf(xk)
    fs, gs, ts, newton = [f_xk], [np.linalg.norm(df_xk)], [0], []
    while np.linalg.norm(gf(xk)) > eps:
        start_time = time.time()
        try:
            np.linalg.cholesky(hf(xk))
            dk = np.linalg.solve(hf_xk.astype('float64'), -df_xk.astype('float64')).astype('float64')
            tk = lsearch(xk, df_xk, [dk, 'newton'])
            newton.append(1)
        except LinAlgError:
            dk = -df_xk
            tk = lsearch(xk, df_xk.astype('float64'), [dk, 'grad'])
            newton.append(0)
        xk = xk + tk * dk
        f_xk, df_xk, hf_xk = f(xk), gf(xk), hf(xk)
        fs.append(f_xk.astype('float64'))
        gs.append(np.linalg.norm(df_xk.astype('float64')))
        ts.append(time.time() - start_time + ts[-1])
    return xk, fs, gs, ts[1:], newton


def hybrid_back(f, alpha, beta, s):
    def lsearch(xk, gk, direction):
        if direction[1] == 'newton':
            return 1
        else:
            dk = direction[0]
            tk = s
            while f(xk + tk * dk).astype('float64') >= f(xk).astype('float64') + alpha * tk * gk.T.dot(dk).astype('float64'):
                tk *= beta
            return tk

    return lsearch


def f_q2(x):
    return ((np.abs(x[0] ** 4,dtype=np.float64)) + (np.abs(x[1] ** 4,dtype=np.float64)) - (36 * x[0] * x[1]).astype('float64'))


def df_q2(x):
    return np.array([(4 * x[0] ** 3 - 36 * x[1]).astype('int64'), (4 * x[1] ** 3 - 36 * x[0]).astype('float64')])


def ddf_q2(x):
    return np.array([[(abs(12 * x[0] ** 2)).astype('int64'), -36],
                     [- 36, (abs(12 * x[1] ** 2)).astype('int64')]])


def generic_grad(f, gf, lsearch, x0, eps):
    xk = x0
    xk_1 = xk - lsearch(f, xk, gf(xk)) * gf(xk)
    f_xk, d_xk = f(xk) , gf(xk)
    fs, gs, ts = [f_xk], [d_xk], [time.time()]
    while np.abs(np.linalg.norm(gf(xk).astype('float64'))) > eps:
        xk = xk_1
        xk_1 = xk - lsearch(f, xk, gf(xk).astype('float64')) * gf(xk).astype('float64')
        fs.append(f(xk).astype('float64'))
        gs.append(np.linalg.norm(gf(xk).astype('float64')))
        ts.append(time.time())
    return xk, fs, gs, ts


def back(alpha, beta, s):
    def lsearch(_f, xk, gk):
        t = s
        while _f(xk - t * gk).astype('float64') > _f(xk).astype('float64') - alpha * t * np.square(np.linalg.norm(gk).astype('float64')):
            t *= beta
        return t

    return lsearch


def q2():
    hn = hybrid_newton(f=f_q2,
                       gf=df_q2,
                       hf=ddf_q2,
                       lsearch=hybrid_back(f=f_q2, alpha=0.25, beta=0.5, s=1),
                       xk=np.array([200, 0]).astype('int64'),
                       eps=10 ** -6)
    gd = generic_grad(f=f_q2, gf=df_q2, lsearch=back(1 / 4, 1 / 2, 1), x0=np.array([200, 0]).astype('float64'),
                     eps=10 ** -6)
    print(hn[0],hn[1],hn[2])
    print(gd[0],gd[1],gd[2])
    plt.semilogy(np.arange(1, len(hn[1]) + 1), np.array(hn[1]) + 162, label="hybrid_newton")
    plt.semilogy(np.arange(1, len(gd[1]) + 1), np.array(gd[1]) + 162, label="generic grad")
    plt.plot(np.arange(1, len(hn[4]) + 1), hn[4], label="direction type", color="red")
    plt.title("Log Scale of f(x) with respect to iteration, with iteration type")
    plt.legend()
    plt.show()


def main():
    # test_analytic_center()
    q2()
    # x, fs, gs, ts, newton = hybrid_newton(f, gf, hf, lsearch, x0, eps)


if __name__ == '__main__':
    main()
