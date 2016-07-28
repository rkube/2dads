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

Nx_arr = np.array([128, 256, 512, 1024], dtype='int')
L2_arr = np.zeros(Nx_arr.shape[0], dtype='float64')
L = 20.

def f_an(x, y):
    return(np.exp(-0.5 * (x * x + y * y)))


for idx, Nx in enumerate(Nx_arr):
    arr1 = np.loadtxt("test_laplace_arr1_%d.dat" % (Nx))[:Nx, :Nx]
    arr2 = np.loadtxt("test_laplace_arr2_%d.dat" % (Nx))[:Nx, :Nx]

    dx = 20. / float(Nx)

    arr2_xx = d2_dx2(arr2, dx=dx, axis=0)
    arr2_yy = d2_dx2(arr2, dx=dx, axis=1)

    res = (arr1 - arr2_xx - arr2_yy)[1:-1, 1:-1]

    L2_arr[idx] = np.sqrt((res * res).sum() / float(res.size))


    plt.figure()
    plt.contourf(arr1)
    plt.colorbar()
    plt.title("arr1, %d" % Nx)

plt.figure()
plt.loglog(Nx_arr, L2_arr, 'o-')
plt.xlabel(r"$N_x$")
plt.ylabel(r"$L_2$")


plt.show()

# End of file laplace_test_convergence.py
