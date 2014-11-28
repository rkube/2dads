///
///  To write diagnostics for turbulence simulations
/// 


#ifndef DIAGNOSTICS_H
#define DIAGNOSTICS_H

#include "2dads_types.h"
#include "cuda_types.h"
#include "error.h"
#include "slab_config.h"
#include "slab_cuda.h"
#include "diag_array.h"


using namespace std;

class diagnostics {
	public:
		
        /// Default constructor, pass a slab_config object.
        /// This initializes the diagnostic arrays and diagnostic variables
		diagnostics(slab_config const);
        /// does nothing
		~diagnostics();

        /// Update the diag_array members from GPU memory. They are used to
        /// compute diagnostic quantities    
        void update_arrays(slab_cuda&);

        /// Call all diagnostic routines, as specified in the slab_config object
        void write_diagnostics(twodads::real_t const, const slab_config&);

    private:
		/// Initialize diagnostic routines: Write file headers
		void init_diagnostic_output(string, string, bool&);

		/// @brief Write blob diagnostics:
        /// @detailed blobs.dat: t theta_max theta_max_x theta_max_y strmf_max strmf_max_x strmf_max_y int(theta) int(theta *x) int(theta_y) V_(X,COM) V(Y,COM) WXX WYY DXX DYY
        void diag_blobs(twodads::real_t const);
        ///@brief Compute energy integrals for various turbulence models
        ///@param time Time of output
        ///@detailed energy.dat: t E K T U W D1 D2 D3 D4 D5 D6 D7 D8 D9 D10 D11 D12 D13 <br>
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
        void diag_energy(twodads::real_t const);
        /// @brief write output for probes
        /// @detailed Probe layout is in a square grid. Specifying num_probes = N_pr gives
        /// @detailed N_pr * N_pr probes in a equidistant grid, starting at n=m=0
        /// @detailed probe write n_tilde, n, phi, phi_tilde, omega, omega_tilde, phi_y_tilde, phi_x, phy_x_tilde
        void diag_probes(twodads::real_t const);
        /// @brief Write simulation parameters to log file
        void diag_rhs(twodads::real_t const);
        /// @brief Creates a log file
		void write_logfile();		
	   
        twodads::diag_data_t slab_layout;

        diag_array<double> theta, theta_x, theta_y;
        diag_array<double> omega, omega_x, omega_y;
        diag_array<double> strmf, strmf_x, strmf_y;

		twodads::real_t time; /// Real time 
		twodads::real_t old_com_x; /// Old radial COM velocity of the blob
		twodads::real_t old_com_y; /// Old poloidal COM velocity of the blob
		twodads::real_t old_wxx; /// Old dispersion tensor of the blob
		twodads::real_t old_wyy; /// Old dispersion tensor of the blob
		twodads::real_t t_probe; /// Time for probe diagnostics
	
        unsigned int n_probes; /// Number of probes
		bool use_log_theta; /// Is the density field logarithmic? If true, use exp(theta) in all routines
        twodads::real_t theta_bg; /// Subtract uniform background on theta
		// Flags which output files have been initialized
		bool init_flag_blobs; /// True if blobs.dat has been initialized
        bool init_flag_energy; /// True if energy.dat has been initialized
        bool init_flag_particles; /// Not implemented yet
        bool init_flag_tprobe; /// Not implemented yet
        bool init_flag_oprobe; /// Not implemented yet
};

#endif //DIAGNOSTICS_H	
