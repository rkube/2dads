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

#ifdef __CUDACC__
#define CUDAMEMBER __host__ __device__
#else
#define CUDAMEMBER
#endif

namespace cuda
{
    typedef double real_t;
    typedef CuCmplx<real_t> cmplx_t;
    //constexpr unsigned int blockdim_nx{32}; ///< Block dimension in radial (x) direction, columns
    //constexpr unsigned int blockdim_my{1};  ///< Block dimension in poloidal(y) direction, rows
    constexpr unsigned int blockdim_col{4}; ///< Block dimension for consecutive elements (y-direction)
    constexpr unsigned int blockdim_row{4}; ///< Block dimension for non-consecutive elements (x-direction)

    constexpr unsigned int blockdim_nx_max{1024};
    constexpr unsigned int blockdim_my_max{1024};

    constexpr unsigned int griddim_nx_max{1024};
    constexpr unsigned int griddim_my_max{1024};

    //constexpr unsigned int num_gp_x{4};                 ///< Number of stored ghost-points in x-direction
    //constexpr unsigned int gp_offset_x{num_gp_x / 2};   ///< Offset for cell values in x-direction
    //constexpr unsigned int num_gp_y{4};                 ///< Number of stored ghost-points in y-direction
    //constexpr unsigned int gp_offset_y{num_gp_y / 2};   ///< Offset for cell values in y-direction

    constexpr size_t num_pad_y{2};                      ///< Number of rows to pad for in-place DFT

    constexpr real_t PI{3.1415926535897932384};     ///< $\pi$
    constexpr real_t TWOPI{6.2831853071795864769};  ///< $2.0 \pi$
    constexpr real_t FOURPIS{39.47841760435743};    ///< $4.0 * \pi^2$
    constexpr real_t epsilon{1e-10};
    constexpr size_t max_initc{6}; /// < Maximal number of initial conditions

    constexpr size_t io_w{16}; //width of fields used in cout
    constexpr size_t io_p{10}; //precision when printing with cout

    // Boundary conditions: Dirichlet, Neumann, periodic
    enum class bc_t {bc_dirichlet, bc_neumann, bc_periodic};
    // vertex centered: x = x_left + n * delta_x
    // cell centered: x = x_left + (n + 1/2) * delta_x
    enum class grid_t {vertex_centered, cell_centered};
    // 1d DFT (in last dimension), 2d DFT
    enum class dft_t {dft_1d, dft_2d};

    /// Align slab_layout_t at 8 byte boundaries(as for real_t)
    /// Do this, otherwise you get differently aligned structures when
    /// compiling in g++ and nvcc. Also, don't forget the -maligned-double flag
    /// in g++
    class slab_layout_t
    {
    public:
        slab_layout_t(real_t _xl, real_t _dx, real_t _yl, real_t _dy, size_t _nx, size_t _pad_x, size_t _my, size_t _pad_y) :
            x_left(_xl), delta_x(_dx), y_lo(_yl), delta_y(_dy), Nx(_nx), pad_x(_pad_x), My(_my), pad_y(_pad_y) {};

        const real_t x_left;
        const real_t delta_x;
        const real_t y_lo;
        const real_t delta_y;
        const size_t Nx;
        const size_t pad_x;
        const size_t My;
        const size_t pad_y;

        CUDAMEMBER inline real_t get_xleft() const {return(x_left);};
        CUDAMEMBER inline real_t get_deltax() const {return(delta_x);};
        CUDAMEMBER inline real_t get_xright() const {return(x_left + static_cast<real_t>(Nx) * delta_x);};
        CUDAMEMBER inline real_t get_Lx() const {return(static_cast<real_t>(Nx) * delta_x);};
        
        CUDAMEMBER inline real_t get_ylo() const {return(y_lo);};
        CUDAMEMBER inline real_t get_deltay() const {return(delta_y);};
        CUDAMEMBER inline real_t get_yup() const {return(y_lo + static_cast<real_t>(My) * delta_y);};
        CUDAMEMBER inline real_t get_Ly() const {return(static_cast<real_t>(My) * delta_y);};

        CUDAMEMBER inline size_t get_nx() const {return(Nx);};
        CUDAMEMBER inline size_t get_pad_x() const {return(pad_x);};

        CUDAMEMBER inline size_t get_my() const {return(My);};
        CUDAMEMBER inline size_t get_pad_y() const {return(pad_y);};
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
    class bvals_t
    {
        public:
            CUDAMEMBER bvals_t(bc_t _bc_left, bc_t _bc_right, bc_t _bc_top, bc_t _bc_bottom, T _bv_l, T _bv_r, T _bv_t, T _bv_b)
                               : bc_left(_bc_left), bc_right(_bc_right), bc_top(_bc_top), bc_bottom(_bc_bottom),
                                 bval_left(_bv_l), bval_right(_bv_r), bval_top(_bv_t), bval_bottom(_bv_b) {};

            CUDAMEMBER inline bc_t get_bc_left() const {return(bc_left);};
            CUDAMEMBER inline bc_t get_bc_right() const {return(bc_right);};
            CUDAMEMBER inline bc_t get_bc_top() const {return(bc_top);};
            CUDAMEMBER inline bc_t get_bc_bottom() const {return(bc_bottom);};

            CUDAMEMBER inline T get_bv_left() const {return(bval_left);};
            CUDAMEMBER inline T get_bv_right() const {return(bval_right);};
            CUDAMEMBER inline T get_bv_top() const {return(bval_top);};
            CUDAMEMBER inline T get_bv_bottom() const {return(bval_bottom);};

            CUDAMEMBER inline bool operator==(const bvals_t rhs) const
            {
                if((bc_left  == rhs.bc_left) 
                   && (bc_right == rhs.bc_right)
                   && (bc_top == rhs.bc_top)
                   && (bc_bottom == rhs.bc_bottom)
                   && (abs(bval_left - rhs.bval_left) < cuda::epsilon)
                   && (abs(bval_right - rhs.bval_right) < cuda::epsilon)
                   && (abs(bval_top - rhs.bval_top) < cuda::epsilon)
                   && (abs(bval_bottom - rhs.bval_bottom) < cuda::epsilon))
                {
                    return (true);
                }
                return (false);
            }

            CUDAMEMBER inline bool operator!=(const bvals_t rhs) const
            {
                return(!(*this == rhs));
            }

        private: 
            // The boundary conditions on the domain border
            const bc_t bc_left;
            const bc_t bc_right;
            const bc_t bc_top;
            const bc_t bc_bottom;

            // The boundary values on the domain border
            const T bval_left;
            const T bval_right;
            const T bval_top;
            const T bval_bottom;
    };


    class stiff_params_t
    {
    public:
        // Provide a standard constructor for pre-C++11
        stiff_params_t(real_t dt, real_t lx, real_t ly, real_t d, real_t h, size_t my, size_t nx21, size_t l) :
        delta_t(dt), length_x(lx), length_y(ly), diff(d), hv(h), My(my), Nx21(nx21), level(l) {};
        const real_t delta_t;
        const real_t length_x;
        const real_t length_y;
        const real_t diff;
        const real_t hv;
        const size_t My;
        const size_t Nx21;
        const size_t level;
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
