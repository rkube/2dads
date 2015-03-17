#!/usr/bin/env python
#-*- Encoding: UTF-8 -*-

import numpy as np
np.set_printoptions(precision=4, linewidth=240)

Nx, My = 16, 16
Lx, Ly = 1.0, 1.0
dx = 2. * Lx / Nx
dy = 2. * Ly / My

x_rg = np.arange(-Lx, Lx, dx)
y_rg = np.arange(-Ly, Ly, dy)

xx, yy = np.meshgrid(x_rg, y_rg)


z = 0.3 * xx + np.sin(2. * np.pi * yy)

print z

print 'max = ', z.max()
print 'min = ', z.min()
print 'sum = ', z.sum()
print 'mean = ', z.mean()

print 'sum(axis=0)'
print z.sum(axis=0)

print 'sum(axis=1)'
print z.sum(axis=1)


# End of file foo.py
