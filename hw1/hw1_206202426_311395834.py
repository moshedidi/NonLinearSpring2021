#!/usr/bin/python
# coding=utf-8
"""
Author:  moshed & eladb
Created on 24/03/2021

"""
import matplotlib.pyplot as plt
import numpy as np


def is_valid_input(A=None, B=None, x=None):
    valid, error_message = True, ''
    n, m = 0, 0
    if A is not None:
        try:
            # Try to create a numpy array_like from given A lists
            A = np.array(A)
            # Validate that all matrix elements are numbers
            _type = np.array(list(A)).dtype
            if _type not in ['int32', 'float64']:
                valid = False
                error_message += ' Matrix A is not all numbers,'
            else:
                # Validate that the matrix is squared and has at least one element
                n_temp, m_temp = A.shape
                if n_temp != m_temp or n_temp <= 0:
                    valid = False
                    error_message += ' Matrix A dimensions are not valid,'
                else:
                    n, m = n_temp, m_temp
        except Exception as err:
            valid = False
            error_message += f' {str(err)},'
    if B is not None:
        try:
            # Try to create a numpy array_like from given B lists if B is given
            B = np.array(B)
            # Validate that all matrix elements are numbers
            _type = np.array(list(B)).dtype
            if _type not in ['int32', 'float64']:
                valid = False
                error_message += ' Matrix B is not all numbers,'
            else:
                # Validate that the matrix is squared and has at least one element and it's dimensions are the same of A
                n_temp, m_temp = B.shape
                if n_temp != m_temp or n_temp <= 0 or n_temp != n or m_temp != m:
                    valid = False
                    error_message += ' Matrix B dimensions are not valid,'
        except Exception as err:
            valid = False
            error_message += f' {str(err)},'
    if x is not None:
        try:
            # Try to create a numpy array from given x
            x = np.array(x)
            # Validate that all array elements are numbers
            _type = np.array(list(x)).dtype
            if _type not in ['int32', 'float64']:
                valid = False
                error_message += ' Array x is not all numbers,'
            else:
                # Validate that the array dimensions are the appropriate to A
                n_temp = x.shape[0]
                if n_temp != n or n_temp <= 0 or len(x.shape) != 1:
                    valid = False
                    error_message += ' Array x dimensions are not valid,'
        except Exception as err:
            valid = False
            error_message += f' {str(err)},'
    if error_message != '':
        print(error_message)
    return valid


def ex1(A, x):
    if not is_valid_input(A=A, x=x):
        return
    n = A.shape[0]
    X = np.array([x, ] * n).T
    i = np.arange(1, n + 1)
    B = np.add(A, np.multiply(X, i)).T
    np.fill_diagonal(B, 0)
    return B


def ex2(A, B, n, b):
    if not is_valid_input(A=A, B=B, x=b):
        return
    """ Create P """
    BABT = np.block([[B, A, B.T]])
    m = A.shape[0]
    P = np.block([[A, B.T, np.array([np.zeros(m), ] * (n - 2) * m).T]])
    second = np.block([[BABT, np.array([np.zeros(m), ] * (n - 3) * m).T]])
    P = np.append(P, second, axis=0)
    for i in range(1, n - 3):
        temp = np.block([[np.array([np.zeros(m), ] * i * m).T, BABT, np.array([np.zeros(m), ] * (n - i - 3) * m).T]])
        P = np.append(P, temp, axis=0)
    before_last = np.block([[np.array([np.zeros(m), ] * (n - 3) * m).T, BABT]])
    P = np.append(P, before_last, axis=0)
    last = np.block([[np.array([np.zeros(m), ] * (n - 2) * m).T, B, A]])
    P = np.append(P, last, axis=0)
    """ Create y """
    y = b.T
    for i in range(2, n + 1):
        temp = b * i
        y = np.append(y, temp, axis=0)
    """ Create Q """
    Q = np.kron(A, P)
    """ Create z """
    z = np.block([[y] * m])
    return np.linalg.solve(Q, z.T)


def create_A_and_y(X, m):
    y = X[:, 1]
    A = np.zeros((m, 5))
    A[:, 0] = np.ones(m)
    A[:, 1] = X[:, 0]
    A[:, 2] = np.square(X[:, 0])
    A[:, 3] = -np.multiply(X[:, 0], y)
    A[:, 4] = -np.multiply(np.square(X[:, 0]), y)
    return A, y


# ex_5_b
def fit_rational(X):
    m = X.shape[0]
    A, y = create_A_and_y(X, m)
    x = np.linalg.lstsq(A, y, rcond=None)
    return x


def create_A_TA(X, m):
    y = X[:, 1]
    A = np.zeros((m, 6))
    A[:, 0] = np.ones(m)
    A[:, 1] = X[:, 0]
    A[:, 2] = np.square(X[:, 0])
    A[:, 3] = -y
    A[:, 4] = -np.multiply(X[:, 0], y)
    A[:, 5] = -np.multiply(np.square(X[:, 0]), y)
    ATA = np.dot(A.T, A)
    return ATA


# ex_5_d
def fit_rational_normed(X):
    m = X.shape[0]
    ATA = create_A_TA(X, m)
    i = np.argmin(np.linalg.eig(ATA)[0])
    x = np.linalg.eig(ATA)[1][:, i]
    return x


def ex5(X):
    X = X.T
    u = fit_rational(X)[0]
    u_normed = fit_rational_normed(X)
    f_u = (u[0] + u[1] * X[:, 0] + u[2] * np.square(X[:, 0])) / (
            1 + u[3] * X[:, 0] + u[4] * np.square(X[:, 0]))
    f_u_normed = (u_normed[0] + u_normed[1] * X[:, 0] + u_normed[2] * np.square(X[:, 0])) / (
            u_normed[3] + u_normed[4] * X[:, 0] + u_normed[5] * np.square(X[:, 0]))
    plt.scatter(X[:, 0], X[:, 1], label="True data")
    plt.scatter(X[:, 0], f_u, label="f(u)")
    plt.scatter(X[:, 0], f_u_normed, label="f(u_normed)")
    plt.plot(X[:, 0], X[:, 1])
    plt.plot(X[:, 0], f_u)
    plt.plot(X[:, 0], f_u_normed)
    print("u is : " + str(u))
    print("u normed is : " + str(u_normed))
    print("The norm of the distance: " + str(np.linalg.norm(X[:, 1] - f_u)))
    print("The norm of the distance from u_norm: " + str(np.linalg.norm(X[:, 1] - f_u_normed)))
    plt.legend()
    plt.show()


def main():
    A = np.array([[1, -2, 3, 7], [4, 5, 6, 7], [-7, 8, 9, 7], [10, -11, 12, 7]])
    x = np.array([17, 6, -3, 0])
    print('EX1: ')
    print(ex1(A, x))

    n_2, m_2 = 4, 3
    A_2 = np.arange(1, m_2 * 3 + 1).reshape(3, 3)
    B_2 = A_2 + (np.ones(m_2 * 3, dtype=int) * 2).reshape(3, 3)
    b_2 = np.arange(1, m_2 + 1).T
    print('EX2: ')
    print(ex2(A=A_2, B=B_2, n=n_2, b=b_2))

    X = np.array([[-0.966175231649752, -0.920529100440521, -0.871040946427231, -0.792416754493313, -0.731997794083466,
                   -0.707678784846507, -0.594776425699584, -0.542182374657374, -0.477652051223985, -0.414002394497506,
                   -0.326351540865686, -0.301458382421319, -0.143486910424499, -0.0878464728184052, -0.0350835941699658,
                   0.0334396260398352, 0.0795033683251447, 0.202974351567305, 0.237382785959596, 0.288908922672592,
                   0.419851917880386, 0.441532730387388, 0.499570508388721, 0.577394288619662, 0.629734626483965,
                   0.690534081997171, 0.868883439039411, 0.911733893303862, 0.940260537535768, 0.962286449219438],
                  [1.61070071922315, 2.00134259950511, 2.52365719332252, 2.33863055618848, 2.46787274461421,
                   2.92596278963705, 4.49457749339454, 5.01302648557115, 5.53887922607839, 5.59614305167494,
                   5.3790027966219, 4.96873291187938, 3.56249278950514, 2.31744895283007, 2.39921966442751,
                   1.52592143155874, 1.42166345066052, 1.19058953217964, 1.23598301133586, 0.461229833080578,
                   0.940922128674924, 0.73146046340835, 0.444386541739061, 0.332335616103906, 0.285195114684272,
                   0.219953363135822, 0.234575259776606, 0.228396325882262, 0.451944920264431, 0.655793276158532]])
    print('EX5: ')
    ex5(X)


if __name__ == '__main__':
    main()
