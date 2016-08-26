/*! 
 * \brief User defined types for twodads
 */

#ifndef TWODADS_TYPES
#define TWODADS_TYPES

#include <iostream>
#include <cassert>
#include <vector>

#ifdef __CUDACC__
#define CUDAMEMBER __host__ __device__
#else
#define CUDAMEMBER
#endif

#include "cucmplx.h"
#ifdef DEVICE
#include "cufft.h"
#endif //DEVICE

#ifdef HOST
#include <fftw3.h>
#endif //HOST


namespace twodads {
    using real_t = double;
    using cmplx_t = CuCmplx<real_t>;

    #ifdef DEVICE
    using fft_handle_t = cufftHandle;
    //using real_arr = cuda_array_bc_nogp<T, allocator_device>;
    //using cmplx_arr = cuda_array_bc_nogp<CuCmplx<T>, allocator_device>;
    #endif //DEVICE

    #ifdef HOST
    using fft_handle_t = fftw_plan;
    //using real_arr = cuda_array_bc_nogp<T, allocator_host>;
    //using cmplx_arr = cuda_array_bc_nogp<CuCmplx<T>, allocator_host>;
    #endif //HOST

    // Use PI from math.h
    constexpr real_t PI{3.141592653589793};
    constexpr real_t TWOPI{6.2831853071795864769};  ///< $2.0 \pi$
    constexpr real_t FOURPIS{39.47841760435743};    ///< $4.0 * \pi^2$
    constexpr real_t epsilon{0.000001}; //10^-6
    constexpr real_t Sigma{-3.185349};

    constexpr size_t max_initc{6}; /// < Maximal number of initial conditions
    
    constexpr int io_w{12}; //width of fields used in cout
    constexpr int io_p{8}; //precision when printing with cout


    // Coefficients for stiffly stable time integration
    constexpr real_t alpha[3][4] = {{1.0, 1.0, 0.0, 0.0}, {1.5, 2.0, -0.5, 0}, {11.0/6.0, 3.0, -1.5, 1.0/3.0}};
    constexpr real_t beta[3][3] = {{1.0, 0.0, 0.0}, {2.0, -1.0, 0.0}, {3.0, -3.0, 1.0}};

    // To access the alpha and beta arrays in cuda, we need a wrapper crutch.
    // Accessing alpha and beta in vec.apply in device code fails without error :/
    // http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#constexpr-variables
    //constexpr __device__ real_t get_alpha(size_t t, size_t k) {return(alpha[t][k]);};
    //constexpr __device__ real_t get_beta(size_t t, size_t k) {return(beta[t][k]);};

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
    struct slab_layout_t
    {
        slab_layout_t(real_t _xl, real_t _dx, real_t _yl, real_t _dy, 
                size_t _nx, size_t _pad_x, size_t _my, size_t _pad_y, 
                twodads::grid_t _grid) :
            x_left(_xl), delta_x(_dx), y_lo(_yl), delta_y(_dy), 
            Nx(_nx), pad_x(_pad_x), My(_my), pad_y(_pad_y),
            grid(_grid), 
            cellshift(grid == twodads::grid_t::vertex_centered ? 0.0 : 0.5) 
            {};

        const real_t x_left;
        const real_t delta_x;
        const real_t y_lo;
        const real_t delta_y;
        const size_t Nx;
        const size_t pad_x;
        const size_t My;
        const size_t pad_y;
        const twodads::grid_t grid;
        const real_t cellshift;

        friend std::ostream& operator<< (std::ostream& os, const slab_layout_t& sl)
        {
            os << "x_left = " << sl.get_xleft();
            os << "\tdelta_x = " << sl.get_deltax();
            os << "\ty_lo = " << sl.get_ylo();
            os << "\tdelta_y = " << sl.get_deltay();
            os << "\tNx = " << sl.get_nx();
            os << "\tpad_x = " << sl.get_pad_x();
            os << "\tMy = " << sl.get_my();
            os << "\tpad_y = " << sl.get_pad_y();
            os << "\tcellshift = " << sl.get_cellshift();
            os << std::endl;
            return(os);
        }

        CUDAMEMBER inline bool operator==(const slab_layout_t& rhs)
        {
            return((get_nx() == rhs.get_nx()) &&
                   (get_my() == rhs.get_my()) &&
                   (get_pad_x() == rhs.get_pad_x()) &&
                   (get_pad_y() == rhs.get_pad_y())); 
        }

        CUDAMEMBER inline real_t get_xleft() const {return(x_left);};
        CUDAMEMBER inline real_t get_deltax() const {return(delta_x);};
        CUDAMEMBER inline real_t get_xright() const {return(x_left + static_cast<real_t>(Nx) * get_deltax());};
        CUDAMEMBER inline real_t get_Lx() const {return(static_cast<real_t>(Nx) * get_deltax());};
        CUDAMEMBER inline real_t get_x(size_t n) const {return(get_xleft() + (static_cast<real_t>(n) + get_cellshift()) * get_deltax());};
        CUDAMEMBER inline real_t get_y(size_t m) const {return(get_ylo() + (static_cast<real_t>(m) + get_cellshift()) * get_deltay());};

        CUDAMEMBER inline real_t get_ylo() const {return(y_lo);};
        CUDAMEMBER inline real_t get_deltay() const {return(delta_y);};
        CUDAMEMBER inline real_t get_yup() const {return(y_lo + static_cast<real_t>(My) * delta_y);};
        CUDAMEMBER inline real_t get_Ly() const {return(static_cast<real_t>(My) * delta_y);};

        CUDAMEMBER inline size_t get_nx() const {return(Nx);};
        CUDAMEMBER inline size_t get_pad_x() const {return(pad_x);};

        CUDAMEMBER inline size_t get_my() const {return(My);};
        CUDAMEMBER inline size_t get_pad_y() const {return(pad_y);};


        CUDAMEMBER inline size_t get_nelem_per_t() const {return((get_nx() + get_pad_x()) * (get_my() + get_pad_y()));};
        CUDAMEMBER inline twodads::grid_t get_grid() const {return(grid);};
        CUDAMEMBER inline real_t get_cellshift() const {return(cellshift);};
    } __attribute__ ((aligned (8)));

    /// Class to store initialization parameters, up to 6 doubles
    /// Using this datatype avoids copying an array with values for init functions
    /// And the init functions can have the same number of arguments
    class init_params_t
    {
    public:
        init_params_t() : i1(0.0), i2(0.0), i3(0.0), i4(0.0), i5(0.0), i6(0.0) {};
        init_params_t(std::vector<twodads::real_t> initc) {
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
            CUDAMEMBER bvals_t(bc_t _bc_left, bc_t _bc_right, T _bv_l, T _bv_r)
                               : bc_left(_bc_left), bc_right(_bc_right), 
                                 bval_left(_bv_l), bval_right(_bv_r) {};

            CUDAMEMBER inline bc_t get_bc_left() const {return(bc_left);};
            CUDAMEMBER inline bc_t get_bc_right() const {return(bc_right);};

            CUDAMEMBER inline T get_bv_left() const {return(bval_left);};
            CUDAMEMBER inline T get_bv_right() const {return(bval_right);};

            CUDAMEMBER inline bool operator==(const bvals_t& rhs) const
            {
                if((bc_left  == rhs.bc_left) 
                   && (bc_right == rhs.bc_right)
                   && (abs(bval_left - rhs.bval_left) < twodads::epsilon)
                   && (abs(bval_right - rhs.bval_right) < twodads::epsilon))
                {
                    return (true);
                }
                return (false);
            }

            CUDAMEMBER inline bool operator!=(const bvals_t& rhs) const
            {
                return(!(*this == rhs));
            }

        private: 
            // The boundary conditions on the domain border
            const bc_t bc_left;
            const bc_t bc_right;

            // The boundary values on the domain border
            const T bval_left;
            const T bval_right;
    };


    class stiff_params_t
    {
        public:
            // Provide a standard constructor for pre-C++11
            CUDAMEMBER stiff_params_t(real_t dt, real_t lx, real_t ly, real_t d, real_t h, size_t my, size_t nx21, size_t l) :
                delta_t(dt), length_x(lx), length_y(ly), diff(d), hv(h), My(my), Nx21(nx21), level(l) {};
            CUDAMEMBER real_t get_deltat() const {return(delta_t);};
            CUDAMEMBER real_t get_lengthx() const {return(length_x);};
            CUDAMEMBER real_t get_lengthy() const {return(length_y);};
            CUDAMEMBER real_t get_diff() const {return(diff);};
            CUDAMEMBER real_t get_hv() const {return(hv);};
            CUDAMEMBER size_t get_my() const {return(My);};
            CUDAMEMBER size_t get_nx21() const {return(Nx21);};
            CUDAMEMBER size_t get_tlevs() const {return(level);};
        private:
            const real_t delta_t;
            const real_t length_x;
            const real_t length_y;
            const real_t diff;
            const real_t hv;
            const size_t My;
            const size_t Nx21;
            const size_t level;
    } __attribute__ ((aligned (8)));



    // Dynamic fields
    // c++11 can used scoped enums, i.e. in a cpp it reads dyn_field_t::f_theta
    // in c++98 (used by nvcc) just use f_theta
    // Make sure, all members of enums have a different name
    enum class dyn_field_t {f_theta, ///< theta, thermodynamic variable
                            f_tau, ///<tau, electron temperature
    		                f_omega  ///< oemga, vorticity
                            };

    /*! \brief Field types
     * \param f_theta
     *
     */
    enum class field_t {f_theta, ///< \f$\theta\f$
        f_theta_x,  ///< \f$\theta_x  \f$, radial derivative of theta
        f_theta_y,  ///< \f$ \theta_y \f$, poloidal derivative of theta
        f_omega,    ///< \f$ \Omega   \f$
        f_omega_x,  ///< \f$ \Omega_x \f$
        f_omega_y,  ///< \f$ \Omega_x \f$ 
        f_tau,      ///< \f$ \tau \f$
        f_tau_x,    ///< \f$ \tau_x \f$
        f_tau_y,    ///< \f$ \tau_x \f$ 
        f_strmf,    ///< \f$ \phi     \f$  
        f_strmf_x,  ///< \f$ \phi_x   \f$ 
        f_strmf_y,  ///< \f$ \phi_y   \f$   
        f_tmp,      ///< tmp field
        f_tmp_x,    ///< tmp field
        f_tmp_y,    ///< tmp field
        f_theta_rhs,///< rhs for time integration
        f_tau_rhs,///< rhs for time integration
        f_omega_rhs ///< rhs for time integration
    };

    /// List of fields of Fourier coefficients
    enum class field_k_t {f_theta_hat,  ///< Fourier coefficients of theta, tlevs time levels
        f_theta_x_hat,  ///< Fourier coefficients of radial derivative of theta, only current time level
        f_theta_y_hat, ///<  Fourier coefficients of poloidal derivative of theta, only current time level
        f_tau_hat, f_tau_x_hat, f_tau_y_hat,
        f_omega_hat, f_omega_x_hat, f_omega_y_hat,
        f_strmf_hat, f_strmf_x_hat, f_strmf_y_hat,
        f_theta_rhs_hat, f_tau_rhs_hat, f_omega_rhs_hat, f_tmp_hat};
               
    /// Initialization function
    enum class init_fun_t {init_NA, ///< Not available, throws an error
        init_gaussian,  ///< Initializes gaussian profile 
        init_constant, ///< Initialize field with constant value
        init_sine, ///< Initializes sinusoidal profile for theta
        init_mode, ///< Initializes single modes for theta_hat
        init_turbulent_bath, ///< Initialize all modes randomly
        init_lamb_dipole ///<Lamb Dipole
    };
    /*
        init_tau_gaussian,  ///< Initializes gaussian profile for tau, init_function = theta_gaussian
        init_both_gaussian, ///< Initializes gaussian profile for both, theta and tau
        init_theta_sine,  ///< Initializes sinusoidal profile for theta
        init_omega_sine,  ///< Initializes sinusoidal profile for omega
        init_both_sine,  ///< Initializes sinusoidal profile for theta and omega
        init_turbulent_bath, ///< Initialize all modes randomly
        init_test, ///< Who knows?
        init_theta_mode, ///< Initializes single modes for theta_hat
        init_omega_mode, ///< Initializes single modes for omega_hat
        init_both_mode, ///< Initializes single modes for theta_hat and omega_hat
        init_lamb,
        init_file, ///< Input from input.h5
    };*/

    /*!
     * Defines the right hand side to use
     */
    enum class rhs_t {
        theta_rhs_ns,          ///< Navier-Stokes equation
        theta_rhs_lin,         ///< interchange model, linear
        theta_rhs_log,         ///< Logarithmic interchange model
        theta_rhs_hw,          ///< Hasegawa-Wakatani model
        theta_rhs_full,        ///< Density continuity equation with compression of ExB and diamagnetic drift
        theta_rhs_hwmod,       ///< Modified Hasegawa-Wakatani model
        theta_rhs_null,        ///< No explicit terms
        theta_rhs_sheath_nlin, ///< Non-linear sheath losses
        omega_rhs_ns,          ///< Navier-Stokes equation
        omega_rhs_hw,          ///< Hasegawa-Wakatani model
        omega_rhs_hwmod,       ///< Modified Hasegawa-Wakatani model
        omega_rhs_hwzf,        ///< Modified Hasegawa-Wakatani model, supressed zonal flow
        omega_rhs_ic,          ///< Interchange turbulence
        omega_rhs_sheath_nlin, ///< Non-linear sheath losses
        omega_rhs_null,        ///< No explicit terms (passive advection)
        tau_rhs_sheath_nlin,   ///< Non-linear sheath losses
        tau_rhs_null           ///< No explicit terms
    };

    /*!
     * Enumerate diagnostic functions
     */
    enum class diagnostic_t {diag_blobs,  ///< Information on blob dynamics
        diag_com_theta,     ///< COM dynamics on theta field
        diag_com_tau,       ///< COM dynamics of tau field
        diag_max_theta,     ///< max dynamics on theta
        diag_max_tau,       ///< max dynamics on tau
        diag_max_omega,     ///< max dynamics on omega
        diag_max_strmf,     ///< max dynamics on strmf
        diag_energy,        ///< Energetics for turbulence simulations
        diag_energy_ns,     ///< Energetics for turbulence simulations
        diag_energy_local,  ///< Energetics for local model energy theorem
        diag_probes,        ///< Time series from probes
        diag_consistency,   ///< Consistency of Laplace solver
        diag_mem            ///< GPU memory usage information
    };
    
    /*!
     *  Available full output of fields
     */
    enum class output_t {o_theta,  ///< Theta, thermodynamic variable
        o_theta_x, ///< Radial derivative of theta
        o_theta_y, ///< Poloidal derivative of theta
        o_omega, o_omega_x, o_omega_y, 
        o_tau, o_tau_x, o_tau_y,
        o_strmf, o_strmf_x, o_strmf_y, 
        o_theta_rhs, o_tau_rhs, o_omega_rhs};

    /*!
     * Domain layout passed to diagnostic functions
     */
    struct diag_data_t
    {
        uint Nx;
        uint My;
        uint runnr;
        real_t x_left;
        real_t delta_x;
        real_t y_lo;
        real_t delta_y;
        real_t delta_t;
    } __attribute__ ((aligned (8)));
};


#endif //2dads_types
//End of file 2dads_types.h
