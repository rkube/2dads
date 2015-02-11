#!/usr/bin/env python
# -*- Encoding: UTF-8 -*-

import numpy as np
import ctypes

Nx, My = 128, 128
real_add_out = np.zeros(Nx * My, dtype='float64')
real_sub_out = np.zeros(Nx * My, dtype='float64')
real_mul_out = np.zeros(Nx * My, dtype='float64')
real_div_out = np.zeros(Nx * My, dtype='float64')

ptr_real_add_out = ctypes.c_void_p(real_add_out.ctypes.data)
ptr_real_sub_out = ctypes.c_void_p(real_sub_out.ctypes.data)
ptr_real_mul_out = ctypes.c_void_p(real_mul_out.ctypes.data)
ptr_real_div_out = ctypes.c_void_p(real_div_out.ctypes.data)


np.set_printoptions(linewidth=267, precision=4)
cuda_array_dll = ctypes.cdll.LoadLibrary('test_array4.so')

# Test real addition, 1+2
cuda_array_dll.real_add(ctypes.c_double(1.0), ctypes.c_double(2.0),
                        ptr_real_add_out,
                        ctypes.c_uint(My), ctypes.c_uint(Nx));
real_add_out = real_add_out.reshape([My, Nx])

# Test real subtraction, 1-2
cuda_array_dll.real_sub(ctypes.c_double(1.0), ctypes.c_double(2.0),
                        ptr_real_sub_out,
                        ctypes.c_uint(My), ctypes.c_uint(Nx));
real_sub_out = real_sub_out.reshape([My, Nx])

# Test real multiplication, 1*2
cuda_array_dll.real_mul(ctypes.c_double(1.0), ctypes.c_double(2.0),
                        ptr_real_mul_out,
                        ctypes.c_uint(My), ctypes.c_uint(Nx));
real_mul_out = real_mul_out.reshape([My, Nx])

# Test real division, 1/2
cuda_array_dll.real_div(ctypes.c_double(1.0), ctypes.c_double(2.0),
                        ptr_real_div_out,
                        ctypes.c_uint(My), ctypes.c_uint(Nx));
real_div_out = real_div_out.reshape([My, Nx])


print real_add_out
print '(r1 + r2): max=%f\tmin=%f\tmean=%f' % (real_add_out.max(),
                                              real_add_out.min(),
                                              real_add_out.mean())
print real_sub_out
print '(r1 - r2): max=%f\tmin=%f\tmean=%f' % (real_sub_out.max(),
                                              real_sub_out.min(),
                                              real_sub_out.mean())
print real_mul_out
print '(r1 * r2): max=%f\tmin=%f\tmean=%f' % (real_mul_out.max(),
                                              real_mul_out.min(),
                                              real_mul_out.mean())
print real_div_out
print '(r1 / r2): max=%f\tmin=%f\tmean=%f' % (real_div_out.max(),
                                              real_div_out.min(),
                                              real_div_out.mean())

################################################################################

# End of file test_array.py
