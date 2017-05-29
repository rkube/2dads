#!/opt/local/bin/python
#-*- Encoding: UTF-8 -*-


import numpy as np
import matplotlib.pyplot as plt

Nx = 64
tlevs = 4
num_t = 50

arr_t0 = np.zeros([4, Nx, Nx], dtype='float')

for tstep in np.arange(0, num_t, 5):
    plt.figure(figsize=(18,4))
    for tl in np.arange(tlevs):
        arr_t0[tl, :, :] = np.loadtxt('test_stiff_solnum_%d_a%1d_t%d_device.dat' % (Nx, tl, tstep))

        plt.subplot(1, 4, tl + 1)
        plt.contourf(arr_t0[tl, :, :], 32)
        plt.colorbar()
    plt.title('arr, tlev=%d, t=%d' % (tl, tstep))




#    if t > 0:
#        plt.figure()
#        plt.contourf(arr1_t[t, :, :] - arr1_t[t - 1, :, :])
#        plt.colorbar()
#        plt.title('arr1 t%d - t%d' % (t, t - 1))

plt.show()


# End of file verify_stiff.py
