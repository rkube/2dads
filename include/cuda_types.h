#ifndef CUDA_TYPES
#define CUDA_TYPES

#include "vector_types.h"

namespace cuda
{
    //typedef cufftDoubleComplex cmplx_t;
    typedef double2 cmplx_t;
    typedef double real_t;
    const int cuda_blockdim_nx = 1;
    const int cuda_blockdim_my = 64;

    struct slab_layout
    {
        double x_left;
        double delta_x;
        double y_lo;
        double delta_y;
        int Nx;
        int My;
    };

};


#endif //CUDA_TYPES
