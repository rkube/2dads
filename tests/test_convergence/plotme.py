#!/usr/bin/env python
# -*- Encoding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from num.fd import d_dx

f = np.loadtxt("f.dat")
fx = np.loadtxt("f_x.dat")
fy = np.loadtxt("f_y.dat")

g = np.loadtxt("g.dat")
gx = np.loadtxt("g_x.dat")
gy = np.loadtxt("g_y.dat")

My, Nx = f.shape
dx = 10.0 / float(Nx)
dy = 10.0 / float(My)

print 'Lx = 10.0, Nx = %d, dx = %e' % (Nx, dx)

fx_num = d_dx(f, dx=dx, axis=1)
fy_num = d_dx(f, dx=dy, axis=0)

L2_fx = np.sqrt(np.sum((fx - fx_num) ** 2.0)) / float(fx.size)
L2_fx_str = '|fx - num(f_x)|: L2 =  %e' % L2_fx

L2_fy = np.sqrt(np.sum((fy - fy_num) ** 2.0)) / float(fy.size)
L2_fy_str = '|fy - num(f_y)|: L2 =  %e' % L2_fy


# #######################################################
# Load numerical and analytical solution of the poisson bracket, compute L2
# error
sol_num = np.loadtxt("sol_num.dat")
sol_an = np.loadtxt("sol_an.dat")

L2_pb = np.sqrt(((sol_num - sol_an) ** 2.0).sum()) / float(sol_num.size)
L2_pb_str = '|pb(num) - pb(an)|_2 = %e' % (L2_pb)
print L2_pb_str

# ######################################################
# Test whether the derivatives are computed correctly

fig1, axarr = plt.subplots(3, 2)
p0 = axarr[0, 0].contourf(fx, 64)
axarr[0, 0].set_title('f_x')
fig1.colorbar(p0, ax=axarr[0, 0])

p1 = axarr[0, 1].contourf(fy, 64)
axarr[0, 1].set_title('f_y')
fig1.colorbar(p1, ax=axarr[0, 1])

p2 = axarr[1, 0].contourf(fx_num, 64)
axarr[1, 0].set_title('f_x (fd)')
fig1.colorbar(p2, ax=axarr[1, 0])

p3 = axarr[1, 1].contourf(fy_num, 64)
axarr[1, 1].set_title('f_y (fd)')
fig1.colorbar(p3, ax=axarr[1, 1])

p4 = axarr[2, 0].contourf(fx - fx_num, 64)
axarr[2, 0].set_title(L2_fx_str)
fig1.colorbar(p4, ax=axarr[2, 0])

p5 = axarr[2, 1].contourf(fy - fy_num, 64)
axarr[2, 1].set_title(L2_fy_str)
fig1.colorbar(p5, ax=axarr[2, 1])

print L2_fx_str
print L2_fy_str




vmax = max(f.max(), fx.max(), fy.max(), g.max(), gx.max(), gy.max())
vmin = min(f.min(), fx.min(), fy.min(), g.min(), gx.min(), gy.min())
levs = np.linspace(vmin, vmax, 128)

fig_pb, axarr = plt.subplots(3, 2)

p0 = axarr[0, 0].contourf(f, levs)
axarr[0, 0].set_title('f')
fig_pb.colorbar(p0, ax=axarr[0, 0])

p1 = axarr[1, 0].contourf(fx, levs)
axarr[1, 0].set_title('fx')
fig_pb.colorbar(p1, ax=axarr[1, 0])

p2 = axarr[2, 0].contourf(fy, levs)
axarr[2, 0].set_title('fy')
fig_pb.colorbar(p2, ax=axarr[2, 0])

p3 = axarr[0, 1].contourf(g, levs)
axarr[0, 1].set_title('g')
fig_pb.colorbar(p3, ax=axarr[0, 1])

p4 = axarr[1, 1].contourf(gx, levs)
axarr[1, 1].set_title('gx')
fig_pb.colorbar(p4, ax=axarr[1, 1])

p5 = axarr[2, 1].contourf(gy, levs)
axarr[2, 1].set_title('gy')
fig_pb.colorbar(p2, ax=axarr[2, 1])

levs = np.linspace(sol_an.min(), sol_an.max(), 128)

fig_pb2, axarr = plt.subplots(1, 3)
p0 = axarr[0].contourf(sol_num, levs)
axarr[0].set_title('pb, numerical')
fig_pb2.colorbar(p0, ax=axarr[0])

p1 = axarr[1].contourf(sol_an, levs)
axarr[1].set_title('pb, analytical')
fig_pb2.colorbar(p1, ax=axarr[1])

p2 = axarr[2].contourf(sol_an - sol_num, 64)
axarr[2].set_title(L2_pb_str)
fig_pb2.colorbar(p2, ax=axarr[2])


plt.show()

# End of file plotme.py
