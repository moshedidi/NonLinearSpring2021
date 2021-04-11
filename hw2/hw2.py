#!/usr/bin/python
# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt

"""
Author:  moshed & eladb
Created on 06/04/2021

"""


def validation(k, l=None, u=None, eps1=None, eps2=None):
    valid = True
    error_message = ''
    if l is not None and u is not None:
        try:
            if l >= u:
                valid = False
                error_message = "The lower limit is bigger then the upper limit"
        except Exception as err:
            valid = False
            error_message += f' {str(err)},'
    if eps1 or eps2 is not None:
        try:
            if eps1 <= 0 or k <= 0:
                valid = False
                error_message = "The difference and the iterations max number must be a positive number"
        except Exception as err:
            valid = False
            error_message += f' {str(err)},'
    if error_message != '':
        print(error_message)
    return valid


def f(x):
    """Calculating f(x) """
    return np.square(x) + (np.square(x) - 3 * x + 10) / (2 + x)


def df(x):
    """Calculating the derivative for f(x)"""
    return 2 * x + (np.square(x) + 4 * x - 16) / np.square(2 + x)


def ddf(x):
    """Calculating the second derivative for f(x)"""
    return 2 + 2 / (2 + x) - 2 * (2 * x - 3) / np.square(2 + x) + 2 * (np.square(x) - 3 * x + 10) / np.power(2 + x, 3)


def generic_gs(f, l, u, eps, k):
    """Running the golden section algorithm """
    validation(l, u, eps, k)
    fv = []
    counter = 0
    tau = (3 - np.sqrt(5)) / 2
    x2 = l + tau * (u - l)
    x3 = l + (1 - tau) * (u - l)
    f_x2,f_x3 = f(x2), f(x3)
    while np.abs(u - l) >= eps and counter <= k:
        fv.append(f((u + l) / 2))
        if f_x2 < f_x3:
            u = x3
            x3 = x2
            f_x3 = f_x2
            x2 = l + tau * (u - l)
            f_x2 = f(x2)
        else:
            l = x2
            x2 = x3
            f_x2 = f_x3
            x3 = l + (1 - tau) * (u - l)
            f_x3 = f(x3)
        counter += 1
    fv.append(f((u + l) / 2))
    return (l + u) / 2, fv


def generic_hybrid(f, df, ddf, l, u, eps1, eps2, k):
    """Running the hybrid newton algorithm"""
    validation(l, u, eps1, eps2, k)
    fv = []
    x = u
    counter = 0
    while counter <= k and np.abs(df(x)) > eps2 and np.abs(u - l) >= eps1:
        fv.append(f(x))
        x_new = x - df(x) / ddf(x)
        if (l <= x_new <= u) and np.abs(df(x_new)) < 0.99 * np.abs(df(x)):
            x = x_new
        else:
            x = (l + u) / 2
        if df(u) * df(x) > 0:
            u = x
        else:
            l = x
    fv.append(f(x))
    return x, fv


def generic_newton(f, df, ddf, x0, eps, k):
    """Running the generic newton algorithm"""
    validation(eps, k)
    fv = []
    x = x0
    counter = 0
    while counter <= k and np.abs(df(x)) > eps:
        fv.append(f(x))
        counter += 1
        x = x - df(x) / ddf(x)
    fv.append(f(x))
    return x, fv


def generic_bisect(f, df, l, u, eps, k):
    """Running the generic bisect algorithm"""
    validation(l, u, eps, k)
    fv = []
    x = (u + l) / 2
    counter = 0
    while np.abs(u - l) >= eps and counter < k:
        fv.append(f(x))
        if df(u) * df(x) > 0:
            u = x
        else:
            l = x
        counter += 1
        x = (u + l) / 2
    fv.append(f(x))
    return x, fv


# def sectionC():
#     l = -1
#     u = 0
#     x0 = 0.5554
#     k = 2
#     x, fv = generic_newton(f=lambda x: -3.55 * np.power(x, 3) + 1.1 * np.square(x)
#                                        + 0.765 * x - 0.74, df=lambda x: -10.65 * np.square(x) + 2.2 * x + 0.765
#                            , ddf=lambda x: -21.3 * x + 2.2, x0=x0, eps=0.0001, k=k)
#     print("good")

def phi(t, mu, a, b, c):
    return mu * np.square(t - a) + np.abs(t - b) + np.abs(t - c)


def gs_denoise_step(mu, a, b, c):
    """Running the golden section algorithm """
    eps = 1 / np.power(10, 10)
    tau = (3 - np.sqrt(5)) / 2
    l = np.min([a, b, c]) - 1
    u = np.max([a, b, c]) + 1
    t2 = l + tau * (u - l)
    t3 = l + (1 - tau) * (u - l)
    f_t2 = phi(t2, mu, a, b, c)
    f_t3 = phi(t3, mu, a, b, c)
    while np.abs(u - l) >= eps:
        if f_t2 < f_t3:
            u = t3
            t3 = t2
            f_t3 = f_t2
            t2 = l + tau * (u - l)
            f_t2 = phi(t2, mu, a, b, c)
        else:
            l = t2
            t2 = t3
            f_t2 = f_t3
            t3 = l + (1 - tau) * (u - l)
            f_t3 = phi(t3, mu, a, b, c)
    return (l + u) / 2

def gs_denoise(s, alpha,N):
    x = s
    for i in range(N):
        x[0] = gs_denoise_step(alpha, s[0], 0, x[1])
        for k in range(1,len(s)-1):
            x[k] = gs_denoise_step(alpha, s[k], x[k-1], x[k+1])
        x[len(x)-1] = gs_denoise_step(alpha, s[len(x)-1], x[len(x)-1], 0)
    return x

def script_ex_4():
    # plotting the real discrete signal
    real_s_1 = [1.] * 40
    real_s_0 = [0.] * 40

    plt.plot(range(40), real_s_1, 'black', linewidth=0.7)
    plt.plot(range(41, 81), real_s_0, 'black', linewidth=0.7)

    # solving the problem
    s = np.array([[1.] * 40 + [0.] * 40]).T + 0.1 * np.random.randn(80, 1)  # noised signal
    # x1 = gs_denoise(s, 0.5, 100)
    # x2 = gs_denoise(s, 0.5, 1000)
    x3 = gs_denoise(s, 0.5, 10000)

    # plt.plot(range(80), s, 'cyan', linewidth=0.7,label = "s")
    # plt.plot(range(80), x1, 'red', linewidth=0.7,label = "x1")
    # plt.plot(range(80), x2, 'green', linewidth=0.7,label = "x2")
    plt.plot(range(80), x3, 'blue', linewidth=0.7,label = "x3")
    plt.legend()
    plt.show()

def main():
    l = -1
    u = 5
    x0 = (u + l) / 2
    k = 50
    eps = eps1 = eps2 = 1 / np.power(10, 6)
    x_bisect, fv_bisect = generic_bisect(f, df, l, u, eps, k)
    plt.semilogy(np.arange(len(fv_bisect)), np.array(fv_bisect) - 3.5825439993037,
                 label=" bisect ")
    x_newton, fv_newton = generic_newton(f, df, ddf, x0, eps, k)
    plt.semilogy(np.arange(len(fv_newton)), np.array(fv_newton) - 3.5825439993037,
                 label=" newton ")
    x_hybrid, fv_hybrid = generic_hybrid(f, df, ddf, l, u, eps1, eps2, k)
    plt.semilogy(np.arange(len(fv_hybrid)), np.array(fv_hybrid) - 3.5825439993037,
                 label="hybrid")
    x_gs, fv_gs = generic_gs(f, l, u, eps, k)
    plt.semilogy(np.arange(len(fv_gs)), np.array(fv_gs) - 3.5825439993037,
                 label="gs")
    plt.title("logarithmic difference with respect to iterations")
    plt.legend()
    plt.show()
    script_ex_4()


if __name__ == '__main__':
    main()
