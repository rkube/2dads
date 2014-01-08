#ifndef CUDA_TYPES
#define CUDA_TYPES

#include "vector_types.h"
#include "cufft.h"

namespace cuda
{
    //typedef cufftDoubleComplex cmplx_t;
    typedef double2 cmplx_t;
    typedef double real_t;
    const unsigned int cuda_blockdim_nx = 1;
    const unsigned int cuda_blockdim_my = 256;
    //const real_t PI = 3.14159265358979323846264338327950288;
    const real_t PI = 3.141592653589793;
    const real_t TWOPI = 6.283185307179586;
    const real_t FOURPIS = 39.47841760435743;

    // Align slab_layout_t at 8 byte boundaries(as for real_t)
    // Do this, otherwise you get differently aligned structures when
    // compiling in g++ and nvcc. Also, don't forget the -maligned-double flag
    // in g++
    struct slab_layout_t
    {
        real_t x_left;
        real_t delta_x;
        real_t y_lo;
        real_t delta_y;
        unsigned int Nx;
        unsigned int My;
    } __attribute__ ((aligned (8)));

    struct stiff_params_t
    {
        real_t delta_t;
        real_t length_x;
        real_t length_y;
        real_t diff;
        real_t S;
        unsigned int Nx;
        unsigned int My;
        unsigned int level;
    } __attribute__ ((aligned (8)));

    const cmplx_t ss3_alpha_c[3][4] = {{make_cuDoubleComplex(1.0, 0.0), make_cuDoubleComplex(1.0, 0.0), make_cuDoubleComplex(0.0, 0.0), make_cuDoubleComplex(0.0, 0.0)},
                                       {make_cuDoubleComplex(1.5, 0.0), make_cuDoubleComplex(2.0, 0.0), make_cuDoubleComplex(-0.5, 0.0), make_cuDoubleComplex(0.0, 0.0)},
                                       {make_cuDoubleComplex(11.0/6.0, 0.0), make_cuDoubleComplex(3.0, 0.0), make_cuDoubleComplex(-1.5, 0.0), make_cuDoubleComplex(1.0 / 3.0, 0.0)}};
    const cmplx_t ss3_beta_c[3][3] = {{make_cuDoubleComplex(1.0, 0.0), make_cuDoubleComplex(0.0, 0.0), make_cuDoubleComplex(0.0, 0.0)}, 
                                      {make_cuDoubleComplex(2.0, 0.0), make_cuDoubleComplex(-1.0, 0.0), make_cuDoubleComplex(0.0, 0.0)}, 
                                      {make_cuDoubleComplex(3.0, 0.0), make_cuDoubleComplex(-3.0, 0.0), make_cuDoubleComplex(1.0, 0.0)}};
};

#endif //CUDA_TYPES
