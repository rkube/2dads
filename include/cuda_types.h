#ifndef CUDA_TYPES
#define CUDA_TYPES

#include "vector_types.h"
#include "cufft.h"
#include "cucmplx.h"
#include <ostream>
#include <assert.h>
#include <vector>


namespace cuda
{
    //typedef cufftDoubleComplex cmplx_t;
    //typedef double2 cmplx_t;
    typedef double real_t;
    typedef CuCmplx<real_t> cmplx_t;
    const unsigned int cuda_blockdim_nx = 1; ///< Block dimension in radial (x) direction
    const unsigned int cuda_blockdim_my = 16; ///< Block dimension in poloidal(y) direction
    //const real_t PI = 3.14159265358979323846264338327950288;
    const real_t PI = 3.141592653589793; ///< $\pi$
    const real_t TWOPI = 6.283185307179586; ///< $2.0 \pi$
    const real_t FOURPIS = 39.47841760435743; ///< $4.0 * \pi^2$

    const int max_initc = 6; /// < Maximal number of initial conditions

    /// Align slab_layout_t at 8 byte boundaries(as for real_t)
    /// Do this, otherwise you get differently aligned structures when
    /// compiling in g++ and nvcc. Also, don't forget the -maligned-double flag
    /// in g++
    class slab_layout_t
    {
    public:
        // Provide standard ctor for pre-C++11
        slab_layout_t(real_t xl, real_t dx, real_t yl, real_t dy, unsigned int n, unsigned int m) :
            x_left(xl), delta_x(dx), y_lo(yl), delta_y(dy), Nx(n), My(m) {};
        const real_t x_left;
        const real_t delta_x;
        const real_t y_lo;
        const real_t delta_y;
        const unsigned int Nx;
        const unsigned int My;

        friend std::ostream& operator<<(std::ostream& os, const slab_layout_t s)
        {
            os << "x_left = " << s.x_left << "\t";
            os << "delta_x = " << s.delta_x << "\t";
            os << "y_lo = " << s.y_lo << "\t";
            os << "delta_y = " << s.delta_y << "\t";
            os << "Nx = " << s.Nx << "\t";
            os << "My = " << s.My << "\n";
            return os;
        }
    } __attribute__ ((aligned (8)));

    /// Class to store initialization parameters, up to 6 doubles
    /// Using this datatype avoids copying an array with values for init functions
    /// And the init functions can have the same number of arguments
    class init_params_t
    {
    public:
        init_params_t() : i1(0.0), i2(0.0), i3(0.0), i4(0.0), i5(0.0), i6(0.0) {};
        init_params_t(std::vector<real_t> initc) {
            assert (initc.size() == 6);
            i1 = initc[0];
            i2 = initc[1];
            i3 = initc[2];
            i4 = initc[3];
            i5 = initc[4];
            i6 = initc[5];
        }
        real_t i1;
        real_t i2;
        real_t i3;
        real_t i4;
        real_t i5;
        real_t i6;
    } __attribute__ ((aligned (8)));


    class stiff_params_t
    {
    public:
        // Provide a standard constructor for pre-C++11
        stiff_params_t(real_t dt, real_t lx, real_t ly, real_t d, real_t h, unsigned int n, unsigned int m, unsigned int l) :
        delta_t(dt), length_x(lx), length_y(ly), diff(d), hv(h), Nx(n), My(m), level(l) {};
        const real_t delta_t;
        const real_t length_x;
        const real_t length_y;
        const real_t diff;
        const real_t hv;
        const unsigned int Nx;
        const unsigned int My;
        const unsigned int level;
        friend std::ostream& operator<<(std::ostream& os, const stiff_params_t s)
        {
            os << "delta_t = " << s.delta_t << "\t";
            os << "length_x = " << s.length_x << "\t";
            os << "length_y = " << s.length_y << "\t";
            os << "diff = " << s.diff << "\t";
            os << "hv = " << s.hv << "\t";
            os << "Nx = " << s.Nx << "\t";
            os << "My = " << s.My << "\t";
            os << "level = " << s.level << "\n";
            return os;
        }

    } __attribute__ ((aligned (8)));

    const real_t ss3_alpha_r[3][4] = {{1.0, 1.0, 0.0, 0.0}, {1.5, 2.0, -0.5, 0.0}, {11./6., 3., -1.5, 1./3.}}; ///< Coefficients for implicit part in time integration
    const real_t ss3_beta_r[3][3] = {{1., 0., 0.}, {2., -1., 0.}, {3., -3., 1.}}; ///< Coefficients for explicit part in time integration
};

#endif //CUDA_TYPES
