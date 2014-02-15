// Initialization routines

#ifndef INITIALIZE_H
#define INITIALIZE_H

#include "include/cuda_types.h"
#include "include/cuda_array3.h"
#include <vector>

typedef cuda_array<cuda::cmplx_t, cuda::real_t> cuda_arr_cmplx;
typedef cuda_array<cuda::real_t, cuda::real_t> cuda_arr_real;

void init_simple_sine(cuda_arr_real*, vector<double>, cuda::slab_layout_t);

void init_gaussian(cuda_arr_real*, vector<double>, cuda::slab_layout_t, bool);

void init_invlapl(cuda_arr_real*, vector<double>, cuda::slab_layout_t);

void init_mode(cuda_arr_cmplx*, vector<double>, cuda::slab_layout_t, uint tlev);

void init_all_modes(cuda_arr_cmplx*, vector<double>, cuda::slab_layout_t, uint);

#endif //INITIALIZE_H
