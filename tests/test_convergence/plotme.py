#!/usr/bin/env python
# -*- Encoding: UTF-8 -*-

import numpy as np
import matplotlib as mpl
mpl.use('AGG')
import matplotlib.pyplot as plt
#from num.fd import d_dx

f = np.loadtxt("f.dat")
fx_num = np.loadtxt("fx_num.dat")
fx_an = np.loadtxt("fx_an.dat")
fy_num = np.loadtxt("fy_num.dat")
fy_an = np.loadtxt("fy_an.dat")

g = np.loadtxt("g.dat")
gx_num = np.loadtxt("gx_num.dat")
gx_an = np.loadtxt("gx_an.dat")
gy_num = np.loadtxt("gy_num.dat")
gy_an = np.loadtxt("gy_an.dat")

lapl_f_num = np.loadtxt("lapl_f_num.dat")
lapl_f_an = np.loadtxt("lapl_f_an.dat")

# ########################################################################
pb1_num = np.loadtxt("pb_num1.dat")
pb2_num = np.loadtxt("pb_num2.dat")
pb3_num = np.loadtxt("pb_num3.dat")
pb4_num = np.loadtxt("pb_num4.dat")
pb_an = np.loadtxt("pb_an.dat")

My, Nx = f.shape
dx = 10.0 / float(Nx)
dy = 10.0 / float(My)

print 'Lx = 10.0, Nx = %d, dx = %e' % (Nx, dx)

# #######################################################
# error

L2_fx = np.sqrt(np.sum((fx_num - fx_an) ** 2.0)) / float(Nx * My)
L2_fx_str = '|fx - num(f_x)|: L2 =  %e' % L2_fx
print L2_fx_str

L2_fy = np.sqrt(np.sum((fy_num - fy_an) ** 2.0)) / float(Nx * My)
L2_fy_str = '|fy - num(f_y)|: L2 =  %e' % L2_fy
print L2_fy_str

L2_gx = np.sqrt(np.sum((gx_num - gx_an) ** 2.0)) / float(Nx * My)
L2_gx_str = '|gx - num(g_x)|: L2 =  %e' % L2_gx
print L2_gx_str

L2_gy = np.sqrt(np.sum((gy_num - gy_an) ** 2.0)) / float(Nx * My)
L2_gy_str = '|gy - num(g_y)|: L2 =  %e' % L2_gy
print L2_gy_str

L2_pb1 = np.sqrt(((pb1_num - pb_an) ** 2.0).sum()) / float(Nx * My)
L2_pb1_str = '|pb1(num) - pb(an)|_2 = %e' % (L2_pb1)
print L2_pb1_str
print 'max |pb(num) - pb(an)|_2 = %e' % (pb1_num - pb_an).max()
print 'min |pb(num) - pb(an)|_2 = %e' % (pb1_num - pb_an).min()
print 'mean |pb(num) - pb(an)|_2 = %e' % (pb1_num - pb_an).mean()

L2_pb2 = np.sqrt(((pb2_num - pb_an) ** 2.0).sum()) / float(Nx * My)
L2_pb2_str = '|pb2(num) - pb(an)|_2 = %e' % (L2_pb2)
print L2_pb2_str
print 'max |pb2(num) - pb(an)|_2 = %e' % (pb2_num - pb_an).max()
print 'min |pb2(num) - pb(an)|_2 = %e' % (pb2_num - pb_an).min()
print 'mean |pb2(num) - pb(an)|_2 = %e' % (pb2_num - pb_an).mean()

L2_pb3 = np.sqrt(((pb3_num - pb_an) ** 2.0).sum()) / float(Nx * My)
L2_pb3_str = '|pb3(num) - pb(an)|_2 = %e' % (L2_pb3)
print L2_pb3_str
print 'max |pb3(num) - pb(an)|_2 = %e' % (pb3_num - pb_an).max()
print 'min |pb3(num) - pb(an)|_2 = %e' % (pb3_num - pb_an).min()
print 'mean |pb3(num) - pb(an)|_2 = %e' % (pb3_num - pb_an).mean()

L2_pb4 = np.sqrt(((pb4_num - pb_an) ** 2.0).sum()) / float(Nx * My)
L2_pb4_str = '|pb4(num) - pb(an)|_2 = %e' % (L2_pb4)
print L2_pb4_str
print 'max |pb4(num) - pb(an)|_2 = %e' % (pb4_num - pb_an).max()
print 'min |pb4(num) - pb(an)|_2 = %e' % (pb4_num - pb_an).min()
print 'mean |pb4(num) - pb(an)|_2 = %e' % (pb4_num - pb_an).mean()



L2_lapl = np.sqrt(((lapl_f_num - lapl_f_an) ** 2.0).sum()) / float(Nx * My)
L2_lapl_str = '|lapl(num) - lapl(an)|_2 = %e' % (L2_lapl)
print L2_lapl_str
print 'max (lapl(num) - lapl(an)) = %e' % (lapl_f_num - lapl_f_an).max()
print 'min (lapl(num) - lapl(an))| = %e' % (lapl_f_num - lapl_f_an).min()
print 'mean (lapl(num) - lapl(an))| = %e' % (lapl_f_num - lapl_f_an).mean()

# ######################################################
# Test if the derivatives computed correctly
lmin = min(fx_an.min(), fx_num.min(), fy_an.min(), fy_num.min())
lmax = max(fx_an.max(), fx_num.max(), fy_an.max(), fy_num.max())
levs = np.linspace(lmin, lmax, 64)

fig1, axarr = plt.subplots(3, 2, figsize=(20, 14))

p0 = axarr[0, 0].contourf(fx_an, levs)
axarr[0, 0].set_title('f_x, analytically')
fig1.colorbar(p0, ax=axarr[0, 0])

p1 = axarr[0, 1].contourf(fy_an, levs)
axarr[0, 1].set_title('f_y, analytically')
fig1.colorbar(p1, ax=axarr[0, 1])

p2 = axarr[1, 0].contourf(fx_num, levs)
axarr[1, 0].set_title('f_x (spectral))')
fig1.colorbar(p2, ax=axarr[1, 0])

p3 = axarr[1, 1].contourf(fy_num, levs)
axarr[1, 1].set_title('f_y (spectral)')
fig1.colorbar(p3, ax=axarr[1, 1])

p4 = axarr[2, 0].contourf(fx_an - fx_num, 64)
axarr[2, 0].set_title(L2_fx_str)
fig1.colorbar(p4, ax=axarr[2, 0])

p5 = axarr[2, 1].contourf(fy_an - fy_num, 64)
axarr[2, 1].set_title(L2_fy_str)
fig1.colorbar(p5, ax=axarr[2, 1])

print 'saving...'
fig1.savefig('fig_derivs.png')

#vmax = max(f.max(), fx_num.max(), fy_num.max(), g.max(), gx_num.max(), gy_num.max())
#vmin = min(f.min(), fx_num.min(), fy_num.min(), g.min(), gx_num.min(), gy_num.min())
#levs = np.linspace(vmin, vmax, 128)

# #####################################################
# derivatives of g
fig_pb, axarr = plt.subplots(3, 2, figsize=(20, 14))

lmin = min(gx_an.min(), gx_num.min(), gy_an.min(), gy_num.min())
lmax = max(gx_an.max(), gx_num.max(), gy_an.max(), gy_num.max())
levs = np.linspace(lmin, lmax, 64)

p0 = axarr[0, 0].contourf(gx_an, levs)
axarr[0, 0].set_title('gx, an.')
fig_pb.colorbar(p0, ax=axarr[0, 0])

p1 = axarr[1, 0].contourf(gx_num, levs)
axarr[1, 0].set_title('gx (spectral)')
fig_pb.colorbar(p1, ax=axarr[1, 0])

p2 = axarr[2, 0].contourf(np.abs(gx_num - gx_an), 64)
axarr[2, 0].set_title(L2_gx_str);
fig_pb.colorbar(p2, ax=axarr[2, 0])

p3 = axarr[0, 1].contourf(gy_an, levs)
axarr[0, 1].set_title('gy, an.')
fig_pb.colorbar(p3, ax=axarr[0, 1])

p4 = axarr[1, 1].contourf(gy_num, levs)
axarr[1, 1].set_title('gy, spectral')
fig_pb.colorbar(p4, ax=axarr[1, 1])

p5 = axarr[2, 1].contourf(np.abs(gy_num - gy_an), 64)
axarr[2, 0].set_title(L2_gy_str);
fig_pb.colorbar(p5, ax=axarr[2, 1])

print 'saving...'
fig_pb.savefig('fig_pb.png')

# ######################################################
# Test whether the piosson bracket is computed correctly
levs = np.linspace(pb_an.min(), pb_an.max(), 128)

fig_pb2, axarr = plt.subplots(4, 3, figsize=(20, 14))

p0 = axarr[0, 0].contourf(pb_an, levs)
axarr[0, 0].set_title('pb, analytical')
fig_pb2.colorbar(p0, ax=axarr[0, 0])

p1 = axarr[0, 1].contourf(pb1_num, levs)
axarr[0, 1].set_title('pb1 = f_x g_y - f_y g_x')
fig_pb2.colorbar(p0, ax=axarr[0, 1])

p2 = axarr[0, 2].contourf(pb_an - pb1_num, 64)
axarr[0, 2].set_title(L2_pb1_str)
fig_pb2.colorbar(p2, ax=axarr[0, 2])

#
p3 = axarr[1, 1].contourf(pb2_num, levs)
axarr[1, 1].set_title('pb2 = (f g_y)_x - (f g_x)_y')
fig_pb2.colorbar(p3, ax=axarr[1, 1])

p4 = axarr[1, 2].contourf(pb_an - pb2_num, 64)
axarr[1, 2].set_title(L2_pb2_str)
fig_pb2.colorbar(p4, ax=axarr[1, 2])

#
p5 = axarr[2, 1].contourf(pb3_num, levs)
axarr[2, 1].set_title('pb3 = (f_x g)_y - (f_y g)_x')
fig_pb2.colorbar(p5, ax=axarr[2, 1])

p6 = axarr[2, 2].contourf(pb_an - pb3_num, 64)
axarr[2, 2].set_title(L2_pb3_str)
fig_pb2.colorbar(p6, ax=axarr[2, 2])

#
p7 = axarr[3, 1].contourf(pb4_num, levs)
axarr[3, 1].set_title('pb3 = (pb1 + pb2 + pb) / 3')
fig_pb2.colorbar(p5, ax=axarr[3, 1])

p8 = axarr[3, 2].contourf(pb_an - pb3_num, 64)
axarr[3, 2].set_title(L2_pb4_str)
fig_pb2.colorbar(p6, ax=axarr[3, 2])

print 'saving...'
fig_pb2.savefig('fig_pb.png')

# ########################################################################
# Test if the Laplacian is computed correctly

lmin = min(lapl_f_num.min(), lapl_f_an.min())
lmax = max(lapl_f_num.max(), lapl_f_an.max())

levs = np.linspace(lmin, lmax, 64)

fig_lp, axarr = plt.subplots(1, 3, figsize=(20, 14))

p0, axarr[0].contourf(lapl_f_num, levs)
axarr[0].set_title('laplace f, num.')
fig_lp.colorbar(p0, ax=axarr[0])

p1, axarr[1].contourf(lapl_f_an, levs)
axarr[1].set_title('laplace f, num.')
fig_lp.colorbar(p1, ax=axarr[1])

p2, axarr[2].contourf(lapl_f_num - lapl_f_an, 64)
axarr[2].set_title(L2_lapl_str)
fig_lp.colorbar(p0, ax=axarr[2])

print 'saving...'
fig_lp.savefig('fig_lp.png')



# ########################################################################
#


plt.show()

# End of file plotme.py
