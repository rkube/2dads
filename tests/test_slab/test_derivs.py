#!/usr/bin/env python
# -*- Encoding: UTF-8 -*-

import numpy as np
import ctypes
from twodads.twodads import input2d
import matplotlib.pyplot as plt
from num.fd import d_dx, d2_dx2

np.set_printoptions(linewidth=267, precision=4)

# Set up simulation domain and data arrays
inp = input2d('.')

Nx, My = inp['Nx'], inp['My']
Lx, Ly = inp['Lx'], inp['Ly']
dx, dy = inp['deltax'], inp['deltay']


theta = np.zeros(Nx * My, dtype='float64')
theta_x = np.zeros(Nx * My, dtype='float64')
theta_y = np.zeros(Nx * My, dtype='float64')

# get pointers from ndarrays

ptr_theta = ctypes.c_void_p(theta.ctypes.data)
ptr_theta_x = ctypes.c_void_p(theta_x.ctypes.data)
ptr_theta_y = ctypes.c_void_p(theta_y.ctypes.data)

# call library function
slab_derivs_dll = ctypes.cdll.LoadLibrary('test_derivs.so')

slab_derivs_dll.test_derivs(ptr_theta, ptr_theta_x, ptr_theta_y)

theta = theta.reshape([Nx, My])
theta_x = theta_x.reshape([Nx, My])
theta_y = theta_y.reshape([Nx, My])

t1 = np.loadtxt('theta1.dat')
t2 = np.loadtxt('theta2.dat')

tx_num = d_dx(theta, axis=1, dx=dx)


plt.figure()
plt.contourf(theta, 32)
plt.colorbar()
plt.title('theta')

plt.figure()
plt.contourf(theta_x, 32)
plt.colorbar()
plt.contour(theta)
plt.title('theta_x')

plt.figure()
plt.contourf(theta_y, 32)
plt.colorbar()
plt.contour(theta)
plt.title('theta_y')

plt.show()
# End of file test_derivs.py
