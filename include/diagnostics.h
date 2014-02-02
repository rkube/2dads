///
///  diagnostics - To write diagnostics for turbulence simulations
/// 


#ifndef __DIAGNOSTICS_H
#define __DIAGNOSTICS_H

#include "include/2dads_types.h"
#include "include/cuda_types.h"
#include "include/slab_config.h"
#include "include/diag_array.h"
#include "include/error.h"
#include "include/cuda_array2.h"
#include "include/slab_cuda.h"


using namespace std;

class diagnostics {
	public:
		
        /// Default constructor, pass a slab_config object
		diagnostics(slab_config const);
        /// does nothing
		~diagnostics();
	
		/// Initialize diagnostic routines: Write file headers
		void init_diagnostic_output(string, string, bool&);
        void update_arrays(slab_cuda&);

        void write_diagnostics(twodads::real_t const, const slab_config&);

    private:

		/// @brief Write blob diagnostics 
        void diag_blobs(twodads::real_t const);
        /// Energy diagnostics
        void diag_energy(twodads::real_t const);
        //void particles_full(tracker const*, slab const * , double);
        /// Write time series of physical variables
        void diag_probes(twodads::real_t const);
	    //void strmf_max(slab_cuda const*, double const);	
        /// Write simulation parameters to log file
		void write_logfile();		
	   
        // This is a fix. applying operator?= on any diag_array returns an array_base
        //twodads::real_t get_mean(diag_array<twodads::real_t>&);    
		
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
	
        unsigned int n_probes; /// Number of blobs
		bool use_log_theta; /// Is the density field logarithmic? If true, use exp(theta) in all routines
        twodads::real_t theta_bg; /// Subtract uniform background on theta
		// Flags which output files have been initialized
		bool init_flag_blobs; /// True if blobs.dat has been initialized
		bool init_flag_kinetic; /// True if kinetic.dat has been initialized
        bool init_flag_thermal; /// True if thermal.dat has been initialized
        bool init_flag_flow; /// True of flows.dat has been initialized
        bool init_flag_particles;
        bool init_flag_tprobe; 
        bool init_flag_oprobe;
};

#endif //__DIAGNOSTICS_H	
