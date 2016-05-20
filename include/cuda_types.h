#ifndef CUDA_TYPES
#define CUDA_TYPES

#include "vector_types.h"
#include "cufft.h"
#include "cucmplx.h"

#include <ostream>
#include <assert.h>
#include <vector>
#include <map>

#ifdef _CUFFT_H_
const std::map<cufftResult, std::string> cufftGetErrorString
{
    {CUFFT_SUCCESS, std::string("CUFFT_SUCCESS")},
    {CUFFT_INVALID_PLAN, std::string("CUFFT_INVALID_PLAN")},
    {CUFFT_ALLOC_FAILED, std::string("CUFFT_ALLOC_FAILED")},
    {CUFFT_INVALID_TYPE, std::string("CUFFT_INVALID_TYPE")},
    {CUFFT_INVALID_VALUE, std::string("CUFFT_INVALID_VALUE")},
    {CUFFT_INTERNAL_ERROR, std::string("CUFFT_INTERNAL_ERROR")},
    {CUFFT_EXEC_FAILED, std::string("CUFFT_EXEC_FAILED")},
    {CUFFT_SETUP_FAILED, std::string("CUFFT_SETUP_FAILED")},
    {CUFFT_INVALID_SIZE, std::string("CUFFT_INVALID_SIZE")},
    {CUFFT_UNALIGNED_DATA, std::string("CUFFT_UNALIGNED_DATA")}
};

#endif


namespace cuda
{
    typedef double real_t;
    typedef CuCmplx<real_t> cmplx_t;
    //constexpr unsigned int blockdim_nx{32}; ///< Block dimension in radial (x) direction, columns
    //constexpr unsigned int blockdim_my{1};  ///< Block dimension in poloidal(y) direction, rows
    constexpr unsigned int blockdim_col{4}; ///< Block dimension for consecutive elements (y-direction)
    constexpr unsigned int blockdim_row{1}; ///< Block dimension for non-consecutive elements (x-direction)

    constexpr unsigned int blockdim_nx_max{1024};
    constexpr unsigned int blockdim_my_max{1024};

    constexpr unsigned int griddim_nx_max{1024};
    constexpr unsigned int griddim_my_max{1024};

    constexpr unsigned int num_gp_x{4};                 ///< Number of stored ghost-points in x-direction
    constexpr unsigned int gp_offset_x{num_gp_x / 2};   ///< Offset for cell values in x-direction
    constexpr unsigned int num_gp_y{4};                 ///< Number of stored ghost-points in y-direction
    constexpr unsigned int gp_offset_y{num_gp_y / 2};   ///< Offset for cell values in y-direction


    constexpr real_t PI{3.1415926535897932384};     ///< $\pi$
    constexpr real_t TWOPI{6.2831853071795864769};  ///< $2.0 \pi$
    constexpr real_t FOURPIS{39.47841760435743};    ///< $4.0 * \pi^2$
    constexpr real_t epsilon{1e-10};
    constexpr int max_initc{6}; /// < Maximal number of initial conditions

    constexpr int io_w{10}; //width of fields used in cout
    constexpr int io_p{4}; //precision when printing with cout

    enum class bc_t {bc_dirichlet, bc_neumann, bc_periodic};

    /// Align slab_layout_t at 8 byte boundaries(as for real_t)
    /// Do this, otherwise you get differently aligned structures when
    /// compiling in g++ and nvcc. Also, don't forget the -maligned-double flag
    /// in g++
    class slab_layout_t
    {
    public:
        // Provide standard ctor for pre-C++11
        slab_layout_t(real_t xl, real_t dx, real_t yl, real_t dy, real_t dt, unsigned int my, unsigned int nx) :
            x_left(xl), delta_x(dx), y_lo(yl), delta_y(dy), delta_t(dt), My(my), Nx(nx) {};
        const real_t x_left;
        const real_t delta_x;
        const real_t y_lo;
        const real_t delta_y;
        const real_t delta_t;
        const unsigned int My;
        const unsigned int Nx;

        friend std::ostream& operator<<(std::ostream& os, const slab_layout_t s)
        {
            os << "x_left = " << s.x_left << "\t";
            os << "delta_x = " << s.delta_x << "\t";
            os << "y_lo = " << s.y_lo << "\t";
            os << "delta_y = " << s.delta_y << "\t";
            os << "delta_t = " << s.delta_t << "\t";
            os << "My = " << s.My << "\t";
            os << "Nx = " << s.Nx << "\n";
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


    template <typename T>
    struct bvals
    {
        // The boundary conditions on the domain border
        bc_t bc_left;
        bc_t bc_right;
        bc_t bc_top;
        bc_t bc_bottom;

        // The boundary values on the domain border
        T bval_left;
        T bval_right;
        T bval_top;
        T bval_bottom;
    };


    class stiff_params_t
    {
    public:
        // Provide a standard constructor for pre-C++11
        stiff_params_t(real_t dt, real_t lx, real_t ly, real_t d, real_t h, unsigned int my, unsigned int nx21, unsigned int l) :
        delta_t(dt), length_x(lx), length_y(ly), diff(d), hv(h), My(my), Nx21(nx21), level(l) {};
        const real_t delta_t;
        const real_t length_x;
        const real_t length_y;
        const real_t diff;
        const real_t hv;
        const int My;
        const int Nx21;
        const int level;
        friend std::ostream& operator<<(std::ostream& os, const stiff_params_t s)
        {
            os << "delta_t = " << s.delta_t << "\t";
            os << "length_x = " << s.length_x << "\t";
            os << "length_y = " << s.length_y << "\t";
            os << "diff = " << s.diff << "\t";
            os << "hv = " << s.hv << "\t";
            os << "My = " << s.My << "\t";
            os << "Nx21 = " << s.Nx21 << "\t";
            os << "level = " << s.level << "\n";
            return os;
        }

    } __attribute__ ((aligned (8)));

    //constexpr real_t ss3_alpha_r[3][4] = {{1.0, 1.0, 0.0, 0.0}, {1.5, 2.0, -0.5, 0.0}, {11./6., 3., -1.5, 1./3.}}; ///< Coefficients for implicit part in time integration
    constexpr real_t ss3_alpha_r[3][3] = {{1., 0., 0.}, {2., -0.5, 0.}, {3., -1.5, 1./3.}}; ///< Coefficients for implicit part in time integration
    constexpr real_t ss3_beta_r[3][3] =  {{1., 0., 0.}, {2., -1. , 0.}, {3., -3.,  1.   }}; ///< Coefficients for explicit part in time integration

#ifdef __CUDACC__
    //__constant__ const real_t ss3_alpha_d[3][4] = {{1.0, 1.0, 0.0, 0.0}, {1.5, 2.0, -0.5, 0.0}, {11./6., 3., -1.5, 1./3.}};
    __constant__ const real_t ss3_alpha_d[3][3] = {{1., 0., 0.}, {2., -0.5, 0.}, {3., -1.5, 1./3.}};
    __constant__ const real_t ss3_beta_d[3][3]  = {{1., 0., 0.}, {2., -1. , 0.}, {3., -3.,  1.  }};
#endif //__CUDACC__
};

#endif //CUDA_TYPES
