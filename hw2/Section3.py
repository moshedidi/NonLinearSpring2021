import numpy as np
import Validation

def hybrid(f, df, ddf, l, u, eps1, eps2):
    """Running the hybrid newton algorithm"""
    Validation.validation(l=l, u=u, eps1=eps1, eps2=eps2)
    x = u
    counter = 0
    while np.abs(df(x)) > eps2 and np.abs(u - l) >= eps1:
        x_new = x - df(x) / ddf(x)
        if (l <= x_new <= u) and np.abs(df(x_new)) < 0.99 * np.abs(df(x)):
            x = x_new
        else:
            x = (l + u) / 2
        if df(u) * df(x) > 0:
            u = x
        else:
            l = x
    return x


def ex2(l, u, eps):
    r = hybrid(f=lambda x: -3.55 * np.power(x, 3) + 1.1 * np.square(x)
                                   + 0.765 * x - 0.74, df=lambda x: -10.65 * np.square(x) + 2.2 * x + 0.765
                       , ddf=lambda x: -21.3 * x + 2.2, l=l, u=u, eps1=eps, eps2=eps)
    return r

