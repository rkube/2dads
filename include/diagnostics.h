///
///  To write diagnostics for turbulence simulations
/// 


#ifndef DIAGNOSTICS_H
#define DIAGNOSTICS_H

#include "include/2dads_types.h"
#include "include/cuda_types.h"
#include "include/error.h"
#include "include/slab_config.h"
#include "include/slab_cuda.h"
#include "include/diag_array.h"


using namespace std;

class diagnostics {
	public:
		
        /// Default constructor, pass a slab_config object
		diagnostics(slab_config const);
        /// does nothing
		~diagnostics();
	
        void update_arrays(slab_cuda&);

        void write_diagnostics(twodads::real_t const, const slab_config&);

    private:
		/// Initialize diagnostic routines: Write file headers
		void init_diagnostic_output(string, string, bool&);

		/// @brief Write blob diagnostics 
        void diag_blobs(twodads::real_t const);
        /// @brief Energy diagnostics
        void diag_energy(twodads::real_t const);
        /// @brief Write time series of physical variables
        void diag_probes(twodads::real_t const);
        /// @brief Write simulation parameters to log file
        void diag_rhs(twodads::real_t const);
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
