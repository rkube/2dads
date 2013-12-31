#ifndef CUDA_TYPES
#define CUDA_TYPES

#include "vector_types.h"

namespace cuda
{
    //typedef cufftDoubleComplex cmplx_t;
    const double PI = 3.14159265358979323846264338327950288;
    typedef double2 cmplx_t;
    typedef double real_t;
    const int cuda_blockdim_nx = 1;
    const int cuda_blockdim_my = 8;
    struct slab_layout
    {
        double x_left;
        double delta_x;
        double y_lo;
        double delta_y;
        unsigned int Nx;
        unsigned int My;
    };
};


#endif //CUDA_TYPES
