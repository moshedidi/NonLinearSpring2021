import Section4
import Section1
import Section3
import numpy as np
import matplotlib.pyplot as plt


def main():
    ####### main for section 1 #######
    l = -1
    u = 5
    x0 = (u + l) / 2
    k = 50
    eps = eps1 = eps2 = 1 / np.power(10, 6)
    x_bisect, fv_bisect = Section1.generic_bisect(Section1.f, Section1.df, l, u, eps, k)
    plt.semilogy(np.arange(len(fv_bisect)), np.array(fv_bisect) - 3.5825439993037,
                 label=" bisect ")
    x_newton, fv_newton = Section1.generic_newton(Section1.f, Section1.df, Section1.ddf, x0, eps, k)
    plt.semilogy(np.arange(len(fv_newton)), np.array(fv_newton) - 3.5825439993037,
                 label=" newton ")
    x_hybrid, fv_hybrid = Section1.generic_hybrid(Section1.f, Section1.df, Section1.ddf, l, u, eps1, eps2, k)
    plt.semilogy(np.arange(len(fv_hybrid)), np.array(fv_hybrid) - 3.5825439993037,
                 label="hybrid")
    x_gs, fv_gs = Section1.generic_gs(Section1.f, l, u, eps, k)
    plt.semilogy(np.arange(len(fv_gs)), np.array(fv_gs) - 3.5825439993037,
                 label="gs")
    plt.title("logarithmic difference with respect to iterations")
    plt.legend()
    plt.show()
    ####### main for section 3 #######
    r = Section3.ex2(-1, 0, 1 / np.power(10, 5))
    print("the argmin for p(x) is :" + str(r))
    ####### main for section 4 #######
    Section4.script_ex_4()


if __name__ == '__main__':
    main()
