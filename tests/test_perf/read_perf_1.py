#!/usr/bin/env python
# -*- Encoding: UTF-8 -*-

""""
read output form test_perf_derivs.cu to verify correctness
of derivatives using kmap field
"""

import numpy as np
import matplotlib.pyplot as plt

rarr = np.loadtxt('r_arr.dat')
rarr_x = np.loadtxt('r_arr_x.dat')
rarr_y = np.loadtxt('r_arr_y.dat')

plt.figure()
plt.contourf(rarr)
plt.title('rarr')
plt.colorbar()

plt.figure()
plt.contourf(rarr_x)
plt.title('rarr_x')
plt.colorbar()

plt.figure()
plt.contourf(rarr_y)
plt.title('rarr_y')
plt.colorbar()


plt.show()

# End of file read_perf_1.py
