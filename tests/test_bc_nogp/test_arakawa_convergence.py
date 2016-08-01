#!/usr/bin/python
#-*- Encoding: UTF-8 -*-

"""
Load input and output from test_arakawa.cu

Compute {f,g} = -f_y g_x + g_y f_x

Input:
    f(x, y) = x - sin(2 pi y)
    f_x = 1
    f_y = - 2 pi cos(2 pi y)

    -> initializes arr1

    g(x, y) = sin(2 pi x) + y
    g_x = 2 pi cos(2 pi x)
    g_y = 1

    -> initializes arr2

    {f,g} = (2 pi)^2 cos(2 pi x) cos(2 pi y) + 1

    -> stored in arr3

"""
import numpy as np
import matplotlib.pyplot as plt

def fout_an(x, y):
    return(4.0 * np.pi * np.pi * np.cos(2. * np.pi * x) * np.cos(2. * np.pi * y) + 1.)

Nx_arr = np.array([128, 256, 512], dtype='int')
L2_arr = np.zeros(Nx_arr.shape[0], dtype='float64')
L = 2.0

for idx, Nx in enumerate(Nx_arr):
    arr1 = np.loadtxt("test_arakawa_arr1_%d_in.dat" % (Nx))[:Nx, :Nx]
    arr2 = np.loadtxt("test_arakawa_arr2_%d_in.dat" % (Nx))[:Nx, :Nx]
    arr3 = np.loadtxt("test_arakawa_arr3_%d_out.dat" % (Nx))[:Nx, :Nx]

    dx = L / float(Nx)
    xrg = -0.5 * L + (np.arange(Nx) + 0.5) * dx
    yrg = -0.5 * L + (np.arange(Nx) + 0.5) * dx

    xx, yy = np.meshgrid(xrg, yrg)

    res = (fout_an(xx, yy) - arr3)[1:-1, 1:-1]
    L2_arr[idx] = np.sqrt((res * res).sum() / float(res.size))

    #plt.figure()
    #plt.contourf(fout_an(xx, yy))
    #plt.colorbar()
    #plt.title('Analytic solution')

    #plt.figure();
    #plt.contourf(arr3)
    #plt.colorbar()
    #plt.title('Numerical solution')

    title_str = r"result, Nx=%d, max=%e, min=%e, L2=%e" % (Nx, res.max(), res.min(), L2_arr[idx])

    plt.figure()
    plt.contourf(res)
    plt.colorbar()
    plt.title(title_str)

plt.figure()
plt.loglog(Nx_arr, L2_arr, 'o-')
plt.xlabel(r"$N_x$")
plt.ylabel(r"$L_2$")

plt.show()
# End of file test_arakawa_convergence.py
