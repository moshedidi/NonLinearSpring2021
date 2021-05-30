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


def main():
    # Generate data.
    m = 20
    n = 15
    np.random.seed(1)
    A = np.random.randn(m, n)
    b = np.random.randn(m)

    # Define and solve the CVXPY problem.
    x = cp.Variable(n)
    cost = cp.sum_squares(A @ x - b)
    prob = cp.Problem(cp.Minimize(cost))
    prob.solve()

    # Print result.
    print("\nThe optimal value is", prob.value)
    print("The optimal x is")
    print(x.value)
    print("The norm of the residual is ", cp.norm(A @ x - b, p=2).value)


if __name__ == '__main__':
    q2_c()
