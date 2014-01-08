/*
 *  diagnostic.h
 *  2dads-oo
 *
 *  Created by Ralph Kube on 05.11.10.
 *  Copyright 2010 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __DIAGNOSTICS_H
#define __DIAGNOSTICS_H

//#include "2dads.h"
//#include "slab.h"
//#include "config.h"
//#include "tracker.h"
//#include "error.h"
//#include "fftw_array.h"
//#include <vector>
#include "include/2dads_types.h"
#include "include/cuda_types.h"
#include "include/slab_config.h"
#include "include/error.h"

using namespace std;

class diagnostics {
	public:
		
        // Default constructor with initialization list
		diagnostics(slab_config const);
		~diagnostics();
	
		// Pointers to diagnostic member functions 
        //typedef boost::function<void( slab const *, double const)> dfun_ptr;

		// Initialize diagnostic routines
		void init_diagnostic_output(string, string, bool&);
	
		// Diagnostic routines
		void blobs(twodads::real_t const);
		void energy(twodads::real_t const);
        //void particles_full(tracker const*, slab const * , double);
        void probes(twodads::real_t const);
	    //void strmf_max(slab_cuda const*, double const);	
		void write_logfile();		
		// Run specified diagnostic routines
		//void write_diagnostics(slab const * , double);
		
		
	private:
        twodads::diag_data_t slab_layout;
    
		twodads::real_t time;
		twodads::real_t old_com_x;
		twodads::real_t old_com_y;
		twodads::real_t old_wxx;
		twodads::real_t old_wyy;
		twodads::real_t t_probe;		
		
		bool use_log_theta;
        twodads::real_t theta_bg;		
		// Flags which output files have been initialized
		bool init_flag_blobs;
		bool init_flag_kinetic;
        bool init_flag_thermal;
        bool init_flag_flow;
        bool init_flag_particles;
        bool init_flag_tprobe;
        bool init_flag_oprobe;
};

#endif //__DIAGNOSTICS_H	
