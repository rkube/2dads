#!/usr/bin/python
#-*- Encoding: UTF-8 -*-

"""
Check the convergence of the inverse laplace solver

Load output from test_laplace for a list of Nx.
Compute the second derivative via finite differences and compare as
((d^2/dx^2 + d^2/dy^2) arr2 - arr1)

See test_laplace.cu for names of output arrays

"""

import numpy as np
import matplotlib.pyplot as plt
from codestash.num.fd import d2_dx2

#Nx_arr = np.array([64, 128, 256, 512, 1024, 2048, 4096], dtype='int')
Nx_arr = np.array([128], dtype='int')
L2_arr = np.zeros(Nx_arr.shape[0], dtype='float64')
L = 20.

def sol_an(x, y):
    return (np.exp(-0.5 * (x * x + y * y)) * (-2.0 + x * x + y * y))


for idx, Nx in enumerate(Nx_arr):
    arr1 = np.loadtxt("test_laplace_arr1_%d.dat" % (Nx))[:Nx, :Nx]
    sol_num = np.loadtxt("test_laplace_solnum_%d.dat" % (Nx))[:Nx, :Nx]

    dx = L / float(Nx)
    xrg = -0.5 * L + (np.arange(Nx) + 0.5) * dx
    yrg = -0.5 * L + (np.arange(Nx) + 0.5) * dx

    xx, yy = np.meshgrid(xrg, yrg)

    res = (sol_an(xx, yy) - sol_num)

    L2_arr[idx] = np.sqrt((res * res).sum() / float(res.shape[0] * res.shape[1]))

    plt.figure()
    plt.contourf(arr1)
    plt.colorbar()
    plt.title('sol_in')

    plt.figure()
    plt.contourf(sol_num)
    plt.colorbar()
    plt.title('sol_num')

    title_str = r"result, Nx = %d, max = %e, min = %e, L2 = %e" % (Nx, res.max(), res.min(), L2_arr[idx])
    plt.figure()
    plt.contourf(res)
    plt.colorbar()
    plt.title(title_str)

plt.figure()
plt.loglog(Nx_arr, L2_arr, 'o-')
plt.xlabel(r"$N_x$")
plt.ylabel(r"$L_2$")


plt.show()

# End of file laplace_test_convergence.py
