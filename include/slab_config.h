/*
 *  config.h
 *  2dads-oo
 *
 *  Created by Ralph Kube on 09.12.10.
 *  Copyright 2010 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __CONFIG_H
#define __CONFIG_H

#include "2dads_types.h"
#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <string>

using namespace std;

/// Store variables of 2dads-oo configuration files.



class slab_config {

public:
	/// Configuration file parsing.
    slab_config();
	~slab_config();

	// Inspectors for private memers
	inline unsigned int get_runnr() const { return runnr; };
	/// Left domain boundary.
	inline double get_xleft() const { return xleft; };
	/// Right domain boundary.
	inline double get_xright() const { return xright; };
	/// Upper domain boundary.
	inline double get_yup() const { return yup; };
	/// Lower domain boundary.
	inline double get_ylow() const { return ylow; };
	/// Number of grid points in x direction.
	inline unsigned int get_nx() const { return Nx; };
	/// Number of grid points in y direction.
	inline unsigned int get_my() const { return My; };
	/// Order of time integration scheme.
	inline unsigned int get_tlevs() const { return tlevs; };
	/// Length of one timestep.
	inline double get_deltat() const { return deltat; };
	/// Simulation time end
	inline double get_tend() const { return tend; };
	/// Time between diagnostic output.
	inline double get_tdiag() const { return tdiag; };
	/// Time between full output.
	inline double get_tout() const { return tout; };
	
	/// Use logarithmic formulation of thermodynamic variable?.
	inline bool get_log_theta() const { return log_theta; };
	/// Dealias spectral arrays?
	inline bool dealias() const { return false;};
	/// Use particle tracking?.
    inline int do_particle_tracking() const { return particle_tracking; };
    /// Randomize modes?
    //bool randomize_modes() const;
    /// Number of radial probes
    inline unsigned int get_nprobe() const { return nprobes; };

	/// Number of particles tracked.
    int get_particle_tracking() const;	
	/// List of diagnostic functions to run.
	inline vector<twodads::diagnostic_t> get_diagnostics() const { return diagnostics; };
	/// List of variables to include in full output.
	inline vector<twodads::output_t> get_output() const { return output; };
    /// List of modes to randomize.
    //inline vector<twodads::mode> get_mode_list() const { return mode_list; };

	/// Vector of all model parameters.
	inline vector<double> get_model_params() const { return model_params; };
	/// Value of a specific model parameter.
	inline double get_model_params(int i) const { return model_params[i]; };

	/// Vector of all initial conditions.
	inline vector<double> get_initc() const { return initc; };
	/// Value of a specific initial condition.
	inline double get_initc(int i) const { return initc[i]; };
	//string get_initial_conditions_str() const;

	/// Number of threads.
	inline int get_nthreads() const { return nthr; };
	
	/// Return the initialization routine.
	inline twodads::init_fun_t get_init_function() const { return init_function; };
	/// Return type of RHS for theta equation.
	inline twodads::rhs_t get_theta_rhs_type() const { return theta_rhs; };
	/// Return type of RHS for omega equation.
	inline twodads::rhs_t get_omega_rhs_type() const { return omega_rhs; };
	
	/// Check configuration consistency.
	int consistency();
	
	/// Domain length in x direction.
	inline double get_lengthx() const { return (xright - xleft); };
	/// Domain length in y direction.
	inline double get_lengthy() const { return (yup - ylow); };
	/// Grid spacing in x direction.
	inline double get_deltax() const { return( (xright - xleft) / double(Nx) ); };
	/// Grid spacing in y direction.
	inline double get_deltay() const { return( (yup - ylow) / double(My) ); };

    void test_modes() const;


private:
	// Configuration variables
	unsigned int runnr;
	double xleft;
	double xright;
	double ylow;
	double yup;
	unsigned int Nx;
	unsigned int My;
	unsigned int tlevs;
    unsigned int nthr;
	string scheme;
	double deltat;
	double tend;
	double tdiag;
	double tout;
	bool log_theta;
	bool do_dealiasing;
    bool do_randomize_modes;
    int particle_tracking;
    unsigned int nprobes;

    twodads::rhs_t theta_rhs;
    twodads::rhs_t omega_rhs;
    twodads::init_fun_t init_function;

    vector<twodads::diagnostic_t> diagnostics;
    vector<twodads::output_t> output;

	//vector<string> diagnostics;
	//vector<string> output;
	vector<double> initc;
	vector<double> model_params;
	string initial_conditions_str;
    string shift_modes_str;
	int nthreads;
	int chunksize;
    double diff;

    //vector<twodads::mode> mode_list;
	
    // helper functions
	void split_at_whitespace(const string, vector<double>*);
};

#endif
