// Initialization routines

#ifndef INITIALIZE_H
#define INITIALIZE_H

#include "cuda_types.h"
#include "cuda_array4.h"
#include <curand_kernel.h>
#include <vector>
#include <ctime>

typedef cuda_array<cuda::cmplx_t> cuda_arr_cmplx;
typedef cuda_array<cuda::real_t> cuda_arr_real;

void init_simple_sine(cuda_arr_real*, vector<double>, cuda::slab_layout_t);

void init_gaussian(cuda_arr_real*, vector<double>, cuda::slab_layout_t, bool);

void init_invlapl(cuda_arr_real*, vector<double>, cuda::slab_layout_t);

void init_mode(cuda_arr_cmplx*, const vector<double>, const cuda::slab_layout_t, const uint tlev);

void init_turbulent_bath(cuda_arr_cmplx*, cuda::slab_layout_t, uint);

#endif //INITIALIZE_H
