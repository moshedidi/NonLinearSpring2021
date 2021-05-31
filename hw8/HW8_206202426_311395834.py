#!/usr/bin/python
# coding=utf-8
"""
Author:  moshed & eladb
Created on 26/05/2021

"""
import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np


def q1_a():
    n = 3
    A = np.array([[1, 1, 0],
                  [1, 2, 0],
                  [0, 0, 1]])
    b = np.array([3, -4, 0])
    st1 = np.array([[0.25, 2, 0],
                    [np.sqrt(31 / 16), 0, 0],
                    [0, 0, 0]])
    b_st1 = np.array([0, 0, 2])

    x = cp.Variable(tuple([n]))
    prob = cp.Problem(cp.Minimize(cp.quad_form(x, A) + b.T @ x),
                      [cp.norm(st1 + b_st1) + cp.quad_over_lin(np.array([1, -1, 1]) @ x + 1,
                                                               np.array([1, 1, 0]) @ x) <= 6,
                       x >= 0])
    prob.solve()

    print("\nThe optimal value is", prob.value)
    print("The optimal x is")
    print(x.value)


def q1_b():
    n = 2
    A = np.array([[1, 1],
                  [1, 1]])
    b = np.array([0, 2])
    x = cp.Variable(tuple([n]))
    prob = cp.Problem(cp.Maximize(cp.quad_form(x, A) + cp.sqrt(b.T @ x + 5) - np.array([2, 3]) @ x),
                      [cp.quad_over_lin(np.array([1, 0]) @ x, np.array([1, 1]) @ x)
                       + cp.power(cp.quad_over_lin(np.array([1, 0]) @ x, np.array([0, 1]) @ x) + 1, 8) <= 100,
                       np.array([1, 1]) @ x >= 4,
                       x[1] >= 0])
    prob.solve()

    print("\nThe optimal value is", prob.value)
    print("The optimal x is")
    print(x.value)


def q2_c():
    m, n = 50, 2
    outliers_num = 10
    np.random.seed(314)
    A = 3000 * np.random.rand(n, m)
    A[:, : outliers_num] += 3000
    p = (10 * np.random.rand(m, 1) + 10).round()
    alpha = 0.01
    gamma = 1.2
    eta1 = 20
    eta2 = 30
    mu1 = 2
    mu2 = 5

    x1 = cp.Variable(tuple([n]))
    prob1 = cp.Problem(cp.Minimize(gamma * alpha * (cp.sum([p[i] * cp.norm(A.T[i] - x1) for i in range(m)]))))
    prob1.solve()

    print("\nThe optimal value is", prob1.value)
    print("The optimal x is")
    print(x1.value)

    x2 = cp.Variable(tuple([n]))
    prob2 = cp.Problem(cp.Minimize(gamma * alpha *
                                   (cp.sum([p[i] * cp.norm(A.T[i] - x2) +
                                            cp.max(cp.vstack([0, alpha * cp.norm(A.T[i] - x2) - eta1])) * mu1 +
                                            cp.max(cp.vstack([0, alpha * cp.norm(A.T[i] - x2) - eta2])) * (mu2 - mu1)
                                            for i in range(m)]))))
    prob2.solve()

    print("\nThe optimal value is", prob2.value)
    print("The optimal x is")
    print(x2.value)

    plt.scatter(x1.value[0], x1.value[1], label="Without", color='red')
    plt.scatter(x2.value[0], x2.value[1], label="with", color='blue')
    plt.scatter(A[0], A[1], label="pos", color='green')
    plt.show()


""" q4 """


def proj_section_b():
    return lambda x: np.array([(1 - (x[1] - x[0])) / 2, (1 - (x[1] - x[0])) / 2])


def grad_proj(f, gf, proj, t, x0, eps):
    fs = []
    xk = x0
    fs.append(f(x0))
    xk_1 = proj(xk - t * gf(xk))
    fs.append(f(xk_1))
    while np.linalg.norm(xk_1 - xk) > eps:
        xk = xk_1
        xk_1 = proj(xk - t * gf(xk))
        fs.append(f(xk_1))
    return xk_1, fs


def q4():
    return grad_proj(f=lambda x: np.square(np.linalg.norm(x)),
                     gf=lambda x: 2 * x,
                     proj=proj_section_b(),
                     t=0.5,
                     x0=np.array([100, 100]),
                     eps=10 ** -8)


def main():
    # q2_c()
    print(q4())


if __name__ == '__main__':
    main()
