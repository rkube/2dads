/*! 
 * \brief User defined types for twodads
 */

#ifndef TWODADS_TYPES
#define TWODADS_TYPES

#ifdef __CUDACC__
#define CUDAMEMBER __host__ __device__
#else
#define CUDAMEMBER
#endif


namespace twodads {
    using real_t = double;
    using cmplx_t = std::complex<twodads::real_t>;
    //typedef std::complex<twodads::real_t> cmplx_t;

    // Use PI from math.h
    constexpr real_t PI{3.141592653589793};
    constexpr real_t epsilon{0.000001}; //10^-6

    constexpr real_t Sigma{-3.185349};
    
    constexpr int io_w{8}; //width of fields used in cout
    constexpr int io_p{4}; //precision when printing with cout

    // Coefficients for stiffly stable time integration
    constexpr real_t alpha[3][4] = {{1.0, 1.0, 0.0, 0.0}, {1.5, 2.0, -0.5, 0}, {11.0/6.0, 3.0, -1.5, 1.0/3.0}};
    constexpr real_t beta[3][3] = {{1.0, 0.0, 0.0}, {2.0, -1.0, 0.0}, {3.0, -3.0, 1.0}};

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
