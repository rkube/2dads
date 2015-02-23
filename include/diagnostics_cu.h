/*
 * diagnostics_cu.h
 *
 *  Created on: Feb 5, 2015
 *      Author: rku000
 */

#ifndef DIAGNOSTICS_CU_H_
#define DIAGNOSTICS_CU_H_

#include "2dads_types.h"
#include "cuda_types.h"
#include "error.h"
#include "slab_config.h"
#include "slab_cuda.h"
#include "cuda_darray.h"

using namespace std;

class diagnostics_cu {
	public:
		typedef void (diagnostics_cu::*diag_func_ptr)(twodads::real_t const);
        /// Default constructor, pass a slab_config object.
        /// This initializes the diagnostic arrays and diagnostic variables
		diagnostics_cu(const slab_config&);
        /// does nothing
		~diagnostics_cu();

        /// Update the diag_array members from GPU memory. They are used to
        /// compute diagnostic quantities
        void update_arrays(const slab_cuda&);
		//void update_array(const cuda_array<double>*, const twodads::field_t);

        /// Call all diagnostic routines, as specified in the slab_config object
        void write_diagnostics(const twodads::real_t, const slab_config&);

    private:
		/// Initialize diagnostic routines: Write file headers
		void init_diagnostic_output(string, string);

		/// @brief Print GPU memry usage to terminal
		void diag_mem(const twodads::real_t);
		/// @brief Write blob diagnostics:
        /// @detailed blobs.dat: t theta_max theta_max_x theta_max_y strmf_max strmf_max_x strmf_max_y int(theta) int(theta *x) int(theta_y) V_(X,COM) V(Y,COM) WXX WYY DXX DYY
        void diag_blobs(const twodads::real_t);
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
        /// @brief write output for probes
        /// @detailed Probe layout is in a square grid. Specifying num_probes = N_pr gives
        /// @detailed N_pr * N_pr probes in a equidistant grid, starting at n=m=0
        /// @detailed probe write n_tilde, n, phi, phi_tilde, omega, omega_tilde, phi_y_tilde, phi_x, phy_x_tilde
        void diag_probes(const twodads::real_t);
        /// @brief Write simulation parameters to log file
        //void diag_rhs(twodads::real_t const);
        /// @brief Creates a log file
		void write_logfile();

        twodads::diag_data_t slab_layout;

        cuda_darray<twodads::real_t> theta, theta_x, theta_y;
        cuda_darray<twodads::real_t> omega, omega_x, omega_y;
        cuda_darray<twodads::real_t> strmf, strmf_x, strmf_y;
        cuda_darray<twodads::real_t> theta_rhs, omega_rhs;

        twodads::real_t* x_vec;
        twodads::real_t* y_vec;
        cuda_darray<twodads::real_t> x_arr, y_arr;

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
