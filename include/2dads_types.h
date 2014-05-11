// User defined types for twodads

#ifndef TWODADS_TYPES
#define TWODADS_TYPES

#include <complex>

namespace twodads {
    //using real_t = double;
    //using m_cmplx = std::complex<twodads::real_t>;
    typedef double real_t;
    typedef std::complex<twodads::real_t> cmplx_t;

    // Use PI from math.h
    //constexpr real_t PI = 3.14159265358979323846264338327950288;
    const double PI = 3.14159265358979323846264338327950288;

    // Coefficients for stiffly stable time integration
    const real_t alpha[3][4] = {{1.0, 1.0, 0.0, 0.0}, {1.5, 2.0, -0.5, 0}, {11.0/6.0, 3.0, -1.5, 1.0/3.0}};
    const real_t beta[3][3] = {{1.0, 0.0, 0.0}, {2.0, -1.0, 0.0}, {3.0, -3.0, 1.0}};

    // Dynamic fields
    // c++11 can used scoped enums, i.e. in a cpp it reads dyn_field_t::f_theta
    // in c++98 (used by nvcc) just use f_theta
    // Make sure, all members of enums have a different name
    //enum class dyn_field_t {f_theta, f_omega};
    enum dyn_field_t {d_theta, ///< theta, thermodynamic variable
                      d_omega  ///< oemga, vorticity
                      };
    /// List of real fields
    //enum class field_t {f_theta, f_theta_x, f_theta_y, f_omega, f_omega_x,
    //                    f_omega_y, f_strmf, f_strmf_x, f_strmf_y};
    enum field_t {f_theta, ///< theta
        f_theta_x,  ///< radial derivative of theta
        f_theta_y, ///< poloidal derivative of theta
        f_omega, f_omega_x, f_omega_y, f_strmf, f_strmf_x, f_strmf_y, f_tmp, f_theta_rhs, f_omega_rhs};

    /// List of fields of Fourier coefficients
    //enum class field_k_t {f_theta_hat, f_theta_x_hat, f_theta_y_hat,
    //                      f_omega_hat, f_omega_x_hat, f_omega_y_hat,
    //                      f_strmf_hat, f_strmf_x_hat, f_strmf_y_hat,
    //                      f_theta_rhs_hat, f_omega_rhs_hat, f_tmp_hat};
    enum field_k_t {f_theta_hat,  ///< Fourier coefficients of theta, tlevs time levels
        f_theta_x_hat,  ///< Fourier coefficients of radial derivative of theta, only current time level
        f_theta_y_hat, ///<  Fourier coefficients of poloidal derivative of theta, only current time level
        f_omega_hat, f_omega_x_hat, f_omega_y_hat,
        f_strmf_hat, f_strmf_x_hat, f_strmf_y_hat,
        f_theta_rhs_hat, f_omega_rhs_hat, f_tmp_hat};
               
    /// Initialization function
    //enum class init_fun_t {init_NA, init_theta_gaussian, init_omega_random_k, init_omega_random_k2,
    //                       init_both_random_k, init_omega_exp_k, init_omega_dlayer, init_const_k,
    //                       init_simple_sine, init_theta_mode, init_theta_const, init_file};
    enum init_fun_t {init_NA, ///< Not available, throws an error
        init_theta_gaussian,  ///< Initializes gaussian profile for theta, init_function = theta_gaussian
        init_both_gaussian, ///< Initializes gaussian profile for both, theta and omega
        init_theta_sine,  ///< Initializes sinusoidal profile for theta
        init_omega_sine,  ///< Initializes sinusoidal profile for omega
        init_both_sine,  ///< Initializes sinusoidal profile for theta and omega
        init_test, ///< Who knows?
        init_theta_mode, ///< Initializes single modes for theta_hat
        init_omega_mode, ///< Initializes single modes for omega_hat
        init_both_mode, ///< Initializes single modes for theta_hat and omega_hat
        init_file, ///< Input from input.h5
    };

    //enum class rhs_t {theta_rhs_lin, theta_rhs_log, theta_rhs_hw, theta_rhs_hwmod, theta_rhs_ic, theta_rhs_NA,
    //                  omega_rhs_std, omega_rhs_hw, omega_rhs_hwmod, omega_rhs_ic, omega_rhs_NA,
    //                  rhs_null};

    enum rhs_t {theta_rhs_lin,  ///< Linear interchange model
        theta_rhs_log, ///< Logarithmic interchange model
        theta_rhs_hw, ///< Hasegawa-Wakatani model
        theta_rhs_hwmod, ///< Modified Hasegawa-Wakatani model
        theta_rhs_ic, ///< Interhange turbulence
        theta_rhs_NA, ///< Not available, throws an error
        omega_rhs_std, ///< who knows?
        omega_rhs_hw, ///< Hasegawa-Wakatani model
        omega_rhs_hwmod,  ///< Modified Hasegawa-Wakatani model
        omega_rhs_hwzf, ///< Modified Hasegawa-Wakatani model, supressed zonal flow
        omega_rhs_ic, ///< Interchange turbulence
        omega_rhs_NA, ///< Set if RHS specified in input.ini is not available, throws an error
        rhs_ns, ///< Navier-Stokes equation
        rhs_null ///< No explicit terms
    };

    //enum class diagnostic_t {diag_blobs, diag_energy, diag_probes};
    /// Available diagnostic functions 
    enum diagnostic_t {diag_blobs,  ///< Information on blob dynamics
        diag_energy, ///< Energetics for turbulence simulations
        diag_probes, ///< Time series from probes
    };
    /// Available full output of fields
    //enum class output_t {theta, theta_x, theta_y, omega, omega_x, omega_y, strmf};
    enum output_t {o_theta,  ///< Theta, thermodynamic variable
        o_theta_x, ///< Radial derivative of theta
        o_theta_y, ///< Poloidal derivative of theta
        o_omega, o_omega_x, o_omega_y, o_strmf, o_strmf_x, o_strmf_y, o_theta_rhs, o_omega_rhs};


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
