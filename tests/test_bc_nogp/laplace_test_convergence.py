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

def fin_an(x, y):
    return (np.exp(-0.5 * (x * x + y * y)) * (-2.0 + x * x + y * y))

def fout_an(x, y):
    return(np.exp(-0.5 * (x * x + y * y)))


for idx, Nx in enumerate(Nx_arr):
    arr1 = np.loadtxt("test_laplace_arr1_%d.dat" % (Nx))[:Nx, :Nx]
    arr2 = np.loadtxt("test_laplace_arr2_%d.dat" % (Nx))[:Nx, :Nx]

    dx = L / float(Nx)
    xrg = -0.5 * L + (np.arange(Nx) + 0.5) * dx
    yrg = -0.5 * L + (np.arange(Nx) + 0.5) * dx

    xx, yy = np.meshgrid(xrg, yrg)

    arr2_xx = d2_dx2(arr2, dx=dx, axis=0)
    arr2_yy = d2_dx2(arr2, dx=dx, axis=1)

    sol_an = fout_an(xx, yy)
    sol_in = fin_an(xx, yy)

    res = (sol_an - arr2)

    L2_arr[idx] = np.sqrt((res * res).sum() / float(res.size))

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
