/*
 * diagnostics_cu.h
 */

#ifndef DIAGNOSTICS_CU_H_
#define DIAGNOSTICS_CU_H_

#include "2dads_types.h"
#include "cuda_types.h"
#include "error.h"
#include "derivatives.h"
#include "slab_config.h"
#include "slab_cuda.h"
#include "cuda_darray.h"

using namespace std;

///@brief Structure to hold cnter-of-mass position and dispersion tensor elements
struct com_t
{
    twodads::real_t X_c;
    twodads::real_t Y_c;

    twodads::real_t W_xx;
    twodads::real_t W_yy;

    friend std::ostream& operator<<(std::ostream& os, com_t com)
    {
        os << "X_c = " << com.X_c << ",\tY_c = " << com.Y_c << ",\tW_xx = " << com.W_xx << ",\tW_yy = " << com.W_yy;
        return (os);
    }
};

class diagnostics_cu {
	public:
		typedef void (diagnostics_cu::*diag_func_ptr)(twodads::real_t const);
        typedef cuda_darray<cuda::real_t> cuda_darr;
        /// Default constructor, pass a slab_config object.
        /// This initializes the diagnostic arrays and diagnostic variables
		diagnostics_cu(const slab_config&);
        /// does nothing
		~diagnostics_cu();

        /// Update the diag_array members from GPU memory. They are used to
        /// compute diagnostic quantities
        void update_arrays(const slab_cuda&);

        /// Call all diagnostic routines, as specified in the slab_config object
        void write_diagnostics(const twodads::real_t, const slab_config&);

    private:
		/// Initialize diagnostic routines: Write file headers
		void init_diagnostic_output(string, string);

		/// @brief Print GPU memry usage to terminal
		void diag_mem(const twodads::real_t);

        ///@brief Writes COM dynamics of theta field
        inline void diag_com_theta(const twodads::real_t time); 
        ///@brief Writes COM dynamics of tau field
        inline void diag_com_tau(const twodads::real_t time); 

        ///@brief Compute COM dynamics of a thermodynamic field
        ///@param fname Name of field \f$\theta\f$, either theta or tau
        ///@param time Time of output
        ///@detailed Writes blobs_theta.dat / blobs_tau.dat
        ///@detailed Columns: t X_c Y_c VX_c VY_c wxx wyy dxx wyy
        ///@detailed \f$ X_c = \int \mathrm{d}A\, x \cdot (\theta - \theta_0) / \int \mathrm{d}A, (\theta - \theta_0) \f$
        ///@detailed \f$ Y_c = \int \mathrm{d}A\, x \cdot (\theta - \theta_0) / \int \mathrm{d}A, (\theta - \theta_0) \f$
        void diag_com(const twodads::field_t fname, com_t&, const twodads::real_t time);
      

        ///@brief Compute energy integrals for local thermal convection
        ///@param time Time of output
        ///@detailed Model Equations: \f$\partial_t n + \{\phi, n\} = \kappa \nabla_\perp^2 n$\f
        ///@detailed and \f$\partial_t \Omega + \{\phi, \Omega\} + \partial_y n = \kappa \nabla \perp^2 \Omega$\f
        ///@detailed \f$ G = \int \mathrm{d}A\, x n $\f
        ///@detailed \f$ E = \int \mathrm{d}A\, 1/2 \left( \nabla_\perp^2 \phi \right)^2 $\f
        ///@detailed \f$ \Gamma = -\int \mathrm{d}A\, n \partial_y \phi $\f
        ///@detailed \f$ \partial_t E(t) - G(t) = 0 $\f
        void diag_energy_local(const twodads::real_t time);

        ///@brief Writes max dynamics of theta field
        inline void diag_max_theta(const twodads::real_t time) {diag_max(twodads::field_t::f_theta, time);};
        ///@brief Writes max dynamics of tau field
        inline void diag_max_tau(const twodads::real_t time) {diag_max(twodads::field_t::f_tau, time);};
        ///@brief Writes max dynamics of omega field
        inline void diag_max_omega(const twodads::real_t time) {diag_max(twodads::field_t::f_omega, time);};
        ///@brief Writes max dynamics of strmf field
        inline void diag_max_strmf(const twodads::real_t time) {diag_max(twodads::field_t::f_strmf, time);};
        
        ///@brief Writes max dynamics
        ///@param fname Name of the field
        ///@param time Time of output
        void diag_max(const twodads::field_t fname, const twodads::real_t time);

        ///@brief Computes COM diagnostics 
        void compute_com(const twodads::field_t, twodads::real_t, twodads::real_t, twodads::real_t, twodads::real_t);

        ///@brief Compute energy integrals for various turbulence models
        ///@param time Time of output
        ///@detailed energy.dat: t E K T U W D1 D2 D3 D4 D5 D6 D7 D8 D9 D10 D11 D12 D13 <br>
        ///@detailed \f$E = \frac{1}{2} \int \mathrm{d}A\, \left( \widetilde{n} + \phi_x^2  + \phi_y^ \right)^2 \f$ <br>
        ///@detailed \f$D_{1} = \frac{1}{2A} \int \mathrm{d}A\, n^2 \f$  <br>
        ///@detailed \f$D_{2} = \frac{1}{2A} \int \mathrm{d}A\, \left( \nabla_\perp \phi \right)^2 \f$  <br>
        ///@detailed \f$D_{3} = \frac{1}{2A} \int \mathrm{d}A\, \Omega^2\f$ <br>
        ///@detailed \f$D_{4} = -\frac{1}{A} \int \mathrm{d}A\, \widetilde{n}\widetilde{\phi_y}\f$ <br>
        ///@detailed \f$D_{5} = \frac{1}{A} \int \mathrm{d}A\, \left( \overline{n} - \overline{\phi} \right)^2\f$ <br>
        ///@detailed \f$D_{6} = \frac{1}{A} \int \mathrm{d}A\, \left( \widetilde{n} - \widetilde{\phi} \right)^2\f$ <br>
        ///@detailed \f$D_{7} = \frac{1}{A} \int \mathrm{d}A\, \bar{\phi} \left( \bar{\phi} - \bar{n} \right)\f$ <br>
        ///@detailed \f$D_{8} = \frac{1}{A} \int \mathrm{d}A\, \widetilde{\phi} \left( \widetilde{\phi} - \widetilde{n} \right) \f$ <br>
        ///@detailed \f$D_{9} = \frac{1}{A} \int \mathrm{d}A\, 0\f$ <br>
        ///@detailed \f$D_{10} = \frac{1}{A} \int \mathrm{d}A\, \nabla_\perp^2 n\f$ <br>
        ///@detailed \f$D_{11} = \frac{1}{A} \int \mathrm{d}A\, \nabla_\perp \phi \nabla_\perp \Omega\f$ <br>
        ///@detailed \f$D_{12} = \frac{1}{A} \int \mathrm{d}A\, n_{xxx}^2 + n_{yyy}^2\f$ <br>
        ///@detailed \f$D_{13} = \frac{1}{A} \int \mathrm{d}A\, \phi_{xxx} \Omega_{xxx} + \phi_{yyy} \Omega_{yyy}\f$ <br>
        void diag_energy(const twodads::real_t);

        ///@brief Conservation integrals for Navier-Stokes system
        ///@param time Time of output
        ///@detailed Compute and conservation laws for navier-stokes equation
        ///@ energy_ns.dat: t V O E
        ///@ \f$V = \int \mathrm{d}A\, \Omega \f$ <br>
        ///@ \f$O = \frac{1}{2} \int \mathrm{d}A \Omega^2 \f$<br>
        ///@ \f$E = \frac{1}{2} \int \left(\phi_x^2 + \phi_y^2\right) \f$<br>
        ///@ \f$\mathrm{CFL} = \max(\frac{\phi_x \triangle_t}{\triangle_x}) + \max(\frac{\phi_x \triangle_t}{\triangle_x}) \f$
        void diag_energy_ns(const twodads::real_t);

        ///@brief Check consistency of numerics
        ///@detailed: Reconstruct phi from Omega and compare against strmf
        ///@detailed: Reconstruct Omega from strmf and compare against Omega
        void diag_consistency(const twodads::real_t);

        /// @brief write output for probes
        /// @detailed Probe layout is in a square grid. Specifying num_probes = N_pr gives
        /// @detailed N_pr * N_pr probes in a equidistant grid, starting at n=m=0
        /// @detailed probe write n_tilde, n, phi, phi_tilde, omega, omega_tilde, phi_y_tilde, phi_x, phy_x_tilde
        void diag_probes(const twodads::real_t);
        /// @brief Write simulation parameters to log file
        //void diag_rhs(twodads::real_t const);
        /// @brief Creates a log file
		void write_logfile();


        // Private data members
        slab_config config;
        cuda::slab_layout_t slab_layout;

        cuda_darray<twodads::real_t> theta, theta_x, theta_y;
        cuda_darray<twodads::real_t> tau, tau_x, tau_y;
        cuda_darray<twodads::real_t> omega, omega_x, omega_y;
        cuda_darray<twodads::real_t> strmf, strmf_x, strmf_y;
        cuda_darray<twodads::real_t> theta_rhs, tau_rhs, omega_rhs;

        cuda_darray<twodads::real_t> theta_xx, theta_yy;
        cuda_darray<twodads::real_t> omega_xx, omega_yy;

        cuda_array<cuda::cmplx_t> tmp_array;

        twodads::real_t* x_vec;
        twodads::real_t* y_vec;
        cuda_darray<twodads::real_t> x_arr, y_arr;

        ///// @brief Compute spatial derivatives
        derivs<cuda::real_t> der;

		twodads::real_t time; /// Simulation time time
		twodads::real_t dt_diag; /// Time step between diagnostic output

        unsigned int n_probes; /// Number of probes
		bool use_log_theta; /// Is the density field logarithmic? If true, use exp(theta) in all routines
        twodads::real_t theta_bg; /// Subtract uniform background on theta

        const map<twodads::field_t, cuda_darray<cuda::real_t>*> get_darr_by_name;
        // Initialize map with pointer to member functions as values...
        // Works, but is this right??
        const map<twodads::diagnostic_t, diag_func_ptr> get_dfunc_by_name;
};
#endif /* DIAGNOSTICS_CU_H_ */
