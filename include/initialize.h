// Initialization routines
#include "include/cuda_array2.h"
#include <vector>

void init_simple_sine(cuda_array<cuda::real_t>*, vector<double>,
        const double, const double, const double, const double);

void init_gaussian(cuda_array<cuda::real_t>*, vector<double>,
        const double, const double, const double, const double, bool);

void init_invlapl(cuda_array<cuda::real_t>*, vector<double>,
        const double, const double, const double, const double);

void init_mode(cuda_array<cuda::cmplx_t>*, vector<double>,
        const double, const double, const double, const double);


