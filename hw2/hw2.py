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
    while np.abs(u - l) >= eps and counter <= k:
        fv.append(f((u + l) / 2))
        if f(x2) < f(x3):
            u = x3
            x3 = x2
            x2 = l + tau * (u - l)
        else:
            l = x2
            x2 = x3
            x3 = l + (1 - tau) * (u - l)
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


def gs_denoise(s, alpha, N):
    x = s
    for i in range(N):
        x[0] = (gs_denoise_step(2 * alpha, s[0], x[1], x[1])) / 2
        for k in range(1, len(s) - 1):
            x[k] = gs_denoise_step(alpha, s[k], x[k - 1], x[k + 1])
        x[len(x) - 1] = (gs_denoise_step(2 * alpha, s[len(x) - 1], x[len(x) - 1], x[len(x) - 1])) / 2
    return x

def ex2(l, u, eps):
    r, fv = generic_hybrid(f=lambda x: -3.55 * np.power(x, 3) + 1.1 * np.square(x)
                                       + 0.765 * x - 0.74, df=lambda x: -10.65 * np.square(x) + 2.2 * x + 0.765
                           , ddf=lambda x: -21.3 * x + 2.2, l=l,u=u, eps1=eps, eps2=eps, k=10000)
    return r


def script_ex_4():
    # plotting the real discrete signal
    real_s_1 = [1.] * 40
    real_s_0 = [0.] * 40

    # solving the problem
    s = np.array([[1.] * 40 + [0.] * 40]).T + 0.1 * np.random.randn(80, 1)  # noised signal
    # x1 = gs_denoise(s, 0.5, 100)
    # x2 = gs_denoise(s, 0.5, 1000)
    # x3 = gs_denoise(s, 0.5, 10000)
    # print(x1)
    # print(x2)
    # print(x3)
    x1 = [[5.04112183e-01], [1.00822437e+00], [1.00822437e+00], [1.00822437e+00], [1.00822437e+00], [9.02017384e-01],
          [8.25467704e-01], [8.25467705e-01], [8.51664682e-01], [1.03814140e+00], [1.09455371e+00], [1.09455371e+00],
          [1.09455371e+00], [1.09455371e+00], [1.06443840e+00], [9.22637767e-01], [8.72084223e-01], [8.72084225e-01],
          [8.88024864e-01], [9.24215155e-01], [9.24215155e-01], [9.07821538e-01], [9.07821542e-01], [9.75585921e-01],
          [9.75585921e-01], [9.75585921e-01], [9.75585921e-01], [9.75585921e-01], [9.75585921e-01], [9.75585921e-01],
          [9.75585921e-01], [9.75585921e-01], [9.75585921e-01], [9.41086415e-01], [9.41086250e-01], [9.41086250e-01],
          [9.46398410e-01], [9.46398577e-01], [9.97757850e-01], [9.97757850e-01], [-1.26796487e-01], [-1.26797599e-01],
          [-1.26797599e-01], [-1.26797366e-01], [-9.15923196e-02], [-9.15923163e-02], [-4.77509133e-02],
          [-4.77509132e-02], [-4.77506873e-02], [1.29547903e-02], [1.29547896e-02], [-1.07556335e-02],
          [-1.07556276e-02], [1.60058057e-01], [1.60058057e-01], [-6.60287602e-02], [-6.60287600e-02],
          [-5.55697584e-02], [1.32681195e-01], [1.32681195e-01], [7.32469705e-02], [7.32469703e-02], [2.98952932e-03],
          [2.98952948e-03], [6.44036109e-03], [9.51431153e-02], [9.51431153e-02], [-1.48599487e-01], [-1.48599484e-01],
          [8.08253667e-03], [8.08253850e-03], [2.56284173e-02], [4.45328744e-02], [4.45328743e-02], [4.45328741e-02],
          [4.45328728e-02], [-1.30249899e-01], [-1.30249896e-01], [-1.55250500e-10], [4.38967466e-11]]
    x2 = [[4.54716126e-01], [9.09432260e-01], [9.39886138e-01], [9.39886138e-01], [9.39887269e-01], [9.51023479e-01],
          [1.00478965e+00], [1.08990880e+00], [1.08990880e+00], [1.02103506e+00], [1.02103146e+00], [1.02103146e+00],
          [1.02103146e+00], [9.72571755e-01], [9.54744318e-01], [9.54744318e-01], [9.56650374e-01], [9.56650374e-01],
          [9.56653940e-01], [1.02290082e+00], [1.02290082e+00], [1.02290082e+00], [1.02290082e+00], [1.02290196e+00],
          [1.03601169e+00], [1.03869591e+00], [1.03869591e+00], [1.03869591e+00], [1.03724656e+00], [1.03724619e+00],
          [1.03724619e+00], [1.03724619e+00], [1.03724619e+00], [9.23536462e-01], [9.23532909e-01], [9.23532909e-01],
          [9.20603680e-01], [9.20603683e-01], [9.57681527e-01], [9.57681527e-01], [1.22520905e-01], [-1.05571625e-01],
          [-1.05571622e-01], [-1.80807620e-02], [-1.80807618e-02], [-1.80807612e-02], [-1.80807598e-02],
          [-1.80777230e-02], [5.37510102e-02], [5.37510104e-02], [5.37535124e-02], [1.10973640e-01], [1.10973640e-01],
          [1.10973640e-01], [-4.96408885e-02], [-4.96408959e-02], [-1.99482402e-01], [-1.99482401e-01],
          [-6.49686546e-03], [-6.49686501e-03], [-6.49686500e-03], [-8.99792755e-02], [-8.99814795e-02],
          [-8.99814785e-02], [-7.84074085e-02], [-7.84074084e-02], [-7.84074080e-02], [-7.84074078e-02],
          [-7.84027750e-02], [5.40757266e-02], [5.40757269e-02], [5.40757269e-02], [5.40757269e-02], [-1.24886769e-02],
          [-1.24914722e-02], [-1.24914730e-02], [-5.42697717e-02], [-5.42697717e-02], [-5.42697570e-02],
          [-5.61065908e-11]]

    x3 = [[4.39463395e-01], [8.78926790e-01], [8.78926790e-01], [8.78926791e-01], [8.78978177e-01], [1.05486988e+00],
          [1.07800763e+00], [1.07800763e+00], [1.06395159e+00], [9.33849754e-01], [8.87111615e-01], [8.87111616e-01],
          [9.41488436e-01], [9.41514665e-01], [1.06676271e+00], [1.06676271e+00], [1.06063655e+00], [1.06063655e+00],
          [1.03604510e+00], [1.03604510e+00], [1.04911275e+00], [1.04911828e+00], [1.07201036e+00], [1.07201036e+00],
          [1.04806516e+00], [1.04806516e+00], [1.05129474e+00], [1.05131466e+00], [1.09285519e+00], [1.09285519e+00],
          [1.07271173e+00], [1.02968486e+00], [1.02965983e+00], [1.02965983e+00], [1.02965983e+00], [1.02965983e+00],
          [1.02965983e+00], [1.02965983e+00], [9.50354568e-01], [9.50354553e-01], [2.01139967e-01], [1.42122625e-01],
          [1.42122624e-01], [8.79496198e-02], [8.79416140e-02], [6.46602020e-02], [6.46601987e-02], [-1.76713293e-01],
          [-1.76713288e-01], [-2.25312093e-02], [-2.25312092e-02], [-2.25312088e-02], [-2.25312087e-02],
          [-2.25312084e-02], [-2.25089965e-02], [3.09798141e-02], [5.86183075e-02], [5.86183042e-02], [-2.27011723e-02],
          [-2.27295410e-02], [-2.28260324e-02], [-7.60676483e-02], [-7.60911696e-02], [-7.60911695e-02],
          [-7.60911694e-02], [-7.60579839e-02], [-5.11736870e-03], [7.16887567e-02], [7.16887568e-02], [7.16887566e-02],
          [7.16887566e-02], [-4.39655872e-02], [-4.39655864e-02], [-1.00304246e-02], [-1.00304244e-02],
          [-1.00304244e-02], [-3.05760815e-02], [-3.05760806e-02], [-7.60403448e-10], [2.18519361e-11]]
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(range(40), real_s_1, 'black', linewidth=0.7)
    axs[0, 0].plot(range(41, 81), real_s_0, 'black', linewidth=0.7)
    axs[0, 1].plot(range(40), real_s_1, 'black', linewidth=0.7)
    axs[0, 1].plot(range(41, 81), real_s_0, 'black', linewidth=0.7)
    axs[1, 0].plot(range(40), real_s_1, 'black', linewidth=0.7)
    axs[1, 0].plot(range(41, 81), real_s_0, 'black', linewidth=0.7)
    axs[1, 1].plot(range(40), real_s_1, 'black', linewidth=0.7)
    axs[1, 1].plot(range(41, 81), real_s_0, 'black', linewidth=0.7)
    axs[0, 0].plot(range(80), s)
    axs[0, 0].set_title('s')
    axs[0, 1].plot(range(80), x1, 'tab:orange')
    axs[0, 1].set_title('x1')
    axs[1, 0].plot(range(80), x2, 'tab:green')
    axs[1, 0].set_title('x2')
    axs[1, 1].plot(range(80), x3, 'tab:red')
    axs[1, 1].set_title('x3')
    # plt.plot(range(80), s, 'cyan', linewidth=0.7, label="s")
    # plt.plot(range(80), x1, 'red', linewidth=0.7, label="x1")
    # plt.plot(range(80), x2, 'green', linewidth=0.7, label="x2")
    # plt.plot(range(80), x3, 'blue', linewidth=0.7, label="x3")
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
    r = ex2(-1, 0, 1 / np.power(10, 5))
    print(r)


if __name__ == '__main__':
    main()
