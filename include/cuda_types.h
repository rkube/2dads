#ifndef CUDA_TYPES
#define CUDA_TYPES

#include "vector_types.h"
#include "cufft.h"

namespace cuda
{
    //typedef cufftDoubleComplex cmplx_t;
    typedef double2 cmplx_t;
    typedef double real_t;
    const unsigned int cuda_blockdim_nx = 1; ///< Block dimension in radial (x) direction
    const unsigned int cuda_blockdim_my = 256; ///< Block dimension in poloidal(y) direction
    //const real_t PI = 3.14159265358979323846264338327950288;
    const real_t PI = 3.141592653589793; ///< Pi
    const real_t TWOPI = 6.283185307179586; ///< 2.0 * pi
    const real_t FOURPIS = 39.47841760435743; ///< 4.0 * pi * pi

    /// Align slab_layout_t at 8 byte boundaries(as for real_t)
    /// Do this, otherwise you get differently aligned structures when
    /// compiling in g++ and nvcc. Also, don't forget the -maligned-double flag
    /// in g++
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

    const real_t ss3_alpha_r[3][4] = {{1.0, 1.0, 0.0, 0.0}, {1.5, 2.0, -0.5, 0.0}, {11./6., 3., -1.5, 1./3.}}; ///< Coefficients for implicit part in time integration
    const real_t ss3_beta_r[3][3] = {{1., 0., 0.}, {2., -1., 0.}, {3., -3., 1.}}; ///< Coefficients for explicit part in time integration
};

#endif //CUDA_TYPES
