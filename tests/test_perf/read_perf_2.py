#!/usr/bin/env python
# -*- Encoding: UTF-8 -*-

""""
read output form test_perf_derivs.cu to verify correctness
of derivatives using kmap field
"""

import numpy as np
import matplotlib as mpl
mpl.use('AGG')
from num.fd import d_dx, d2_dx2
import matplotlib.pyplot as plt

rarr = np.loadtxt('r_arr.dat')
rarr_x = np.loadtxt('r_arr_x.dat')
rarr_x2 = np.loadtxt('r_arr_x2.dat')
rarr_y = np.loadtxt('r_arr_y.dat')
rarr_y2 = np.loadtxt('r_arr_y2.dat')

Nx, My = rarr.shape
dx = 10. / float(Nx)
dy = 10. / float(My)

rarr_x_fd = d_dx(rarr, dx=dx, axis=1)
rarr_x2_fd = d2_dx2(rarr, dx=1.0, axis=1)

rarr_y_fd = d_dx(rarr, dx=dy, axis=0)
rarr_y2_fd = d2_dx2(rarr, dx=dy, axis=0)

# ###########################################################
# test d/dx
fig, axgr = plt.subplots(1, 3, figsize=(14, 10))
p1 = axgr[0].contourf(rarr_x, 64)
axgr[0].set_title('d/dx, spectral')
fig.colorbar(p1, ax=axgr[0])

p2 = axgr[1].contourf(rarr_x_fd, 64)
axgr[1].set_title('d/dx, fd')
fig.colorbar(p2, ax=axgr[1])

p3 = axgr[2].contourf(np.abs(rarr_x - rarr_x_fd), 64)
axgr[2].set_title('d/dx, abs(spectral - fd)')
fig.colorbar(p3, ax=axgr[2])

fig.savefig('fig_ddx.png')

# ###########################################################
# test d/dy
fig, axgr = plt.subplots(1, 3, figsize=(14, 10))
p1 = axgr[0].contourf(rarr_y, 64)
axgr[0].set_title('d/dy, spectral')
fig.colorbar(p1, ax=axgr[0])

p2 = axgr[1].contourf(rarr_y_fd, 64)
axgr[1].set_title('d/dy, fd')
fig.colorbar(p2, ax=axgr[1])

p3 = axgr[2].contourf(np.abs(rarr_y - rarr_y_fd), 64)
axgr[2].set_title('d/dy, abs(spectral - fd)')
fig.colorbar(p3, ax=axgr[2])

fig.savefig('fig_ddy.png')
# ###########################################################
# test d2/dx2
fig, axgr = plt.subplots(1, 3, figsize=(14, 10))
p1 = axgr[0].contourf(rarr_x2, 64)
axgr[0].set_title('d2/dx2, spectral')
fig.colorbar(p1, ax=axgr[0])

p2 = axgr[1].contourf(rarr_x2_fd, 64)
axgr[1].set_title('d2/dx2, fd')
fig.colorbar(p2, ax=axgr[1])

p3 = axgr[2].contourf(np.abs(rarr_x2 - rarr_x2_fd), 64)
axgr[2].set_title('d/dy, abs(spectral - fd)')
fig.colorbar(p3, ax=axgr[2])

fig.savefig('fig_ddx2.png')
# ###########################################################
# test d2/dy2
fig, axgr = plt.subplots(1, 3, figsize=(14, 10))
p1 = axgr[0].contourf(rarr_y2, 64)
axgr[0].set_title('d2/dy2, spectral')
fig.colorbar(p1, ax=axgr[0])

p2 = axgr[1].contourf(rarr_y2_fd, 64)
axgr[1].set_title('d2/dy2, fd')
fig.colorbar(p2, ax=axgr[1])

p3 = axgr[2].contourf(np.abs(rarr_y2 - rarr_y2_fd), 64)
axgr[2].set_title('d2/dy2, abs(spectral - fd)')
fig.colorbar(p3, ax=axgr[2])
fig.savefig('fig_ddy2.png')


plt.show()

# End of file read_perf_1.py
