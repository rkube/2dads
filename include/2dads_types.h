// User defined types for twodads

#ifndef TWODADS_TYPES
#define TWODADS_TYPES

#include <tuple>   
#include <complex>

namespace twodads {
    using real_t = double;
    using m_cmplx = std::complex<twodads::real_t>;

    // Use PI from math.h
    constexpr real_t PI = 3.14159265358979323846264338327950288;

    // Coefficients for stiffly stable time integration
    constexpr real_t alpha[3][4] = {{1.0, 1.0, 0.0, 0.0}, {1.5, 2.0, -0.5, 0}, {11.0/6.0, 3.0, -1.5, 1.0/3.0}};
    constexpr real_t beta[3][3] = {{1.0, 0.0, 0.0}, {2.0, -1.0, 0.0}, {3.0, -3.0, 1.0}};

    // Dynamic fields
    enum class dyn_field_t {f_theta, f_omega};
    // List of real fields
    enum class field_t {f_theta, f_theta_x, f_theta_y, f_omega, f_omega_x,
                        f_omega_y, f_strmf, f_strmf_x, f_strmf_y};
    // List of fields of Fourier coefficients
    enum class field_k_t {f_theta_hat, f_theta_x_hat, f_theta_y_hat,
                          f_omega_hat, f_omega_x_hat, f_omega_y_hat,
                          f_strmf_hat, f_strmf_x_hat, f_strmf_y_hat,
                          f_theta_rhs_hat, f_omega_rhs_hat, f_tmp_hat};
               
    // Initialization function
    enum class init_fun_t {init_NA, init_theta_gaussian, init_omega_random_k, init_omega_random_k2,
                           init_both_random_k, init_omega_exp_k, init_omega_dlayer, init_const_k,
                           init_simple_sine, init_theta_mode, init_theta_const, init_file};

    enum class rhs_t {theta_rhs_lin, theta_rhs_log, theta_rhs_hw, theta_rhs_hwmod, theta_rhs_null, theta_rhs_ic, theta_rhs_NA,
                      omega_rhs_std, omega_rhs_hw, omega_rhs_hwmod, omega_rhs_ic, omega_rhs_null, omega_rhs_NA};

    // Diagnostic functions
    enum class diagnostic_t {diag_blobs, diag_energy, diag_probes};
    // Output fields
    enum class output_t {theta, theta_x, theta_y, omega, omega_x, omega_y, strmf};

    typedef std::tuple<int, int> mode;
};
#endif //2dads_types

//End of file 2dads_types.h