/*
 *  config.h
 *  2dads-oo
 *
 *  Created by Ralph Kube on 09.12.10.
 *  Copyright 2010 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef CONFIG_H
#define CONFIG_H

#include <iostream>
#include <map>
#include "2dads_types.h"

using namespace std;

/*! \brief store simulation domain information
 *  \ingroup configuration
 */



class slab_config {

public:
    /*!
     * \brief Standard constructor reads input.ini from pwd,
     * call slab_config('.') as delegating constructor
     */
    slab_config() : slab_config(string("input.ini")) {};

    /*
     * \brief Constructor, reads input.ini in directory, specified by argument
     */
    slab_config(string);

	~slab_config() {};

    // Inspectors for private members
	inline unsigned int get_runnr() const { return runnr; };
	/*! 
     * \brief Left domain boundary.
     * \return Left domain boundary (double)
     */
	inline double get_xleft() const { return xleft; };
	/*! 
     * \brief Right domain boundary.
     * \return Right domain boundary (double)
     */
	inline double get_xright() const { return xright; };
	/*! 
     * \brief Upper domain boundary.
     * \returns Upper domain boundary (double)
     */
	inline double get_yup() const { return yup; };
	/*! 
     * \brief Lower domain boundary.
     * \returns Lower domain bonudary (double)
     */
	inline double get_ylow() const { return ylow; };
	/*! 
     * \brief Number of grid points in x direction.
     * \returns Number of grid points in x direction
     */
	inline unsigned int get_nx() const { return Nx; };
	/*!
     * \brief Number of grid points in y direction.
     * \returns Number of grid points in y direction
     */
	inline unsigned int get_my() const { return My; };
	/*! 
     * \brief Order of time integration scheme.
     * \returns (currently only 4th order Karniadakis scheme is supported
     */
	inline unsigned int get_tlevs() const { return tlevs; };
	/*!
     * \brief Length of one timestep.
     * \returns \f$ \mathrm{d}t \f$
     */
	inline double get_deltat() const { return deltat; };
	/*! 
     * \brief Simulation time end
     * \returns \f$ T_{\mathrm{end}} \f$
     */
	inline double get_tend() const { return tend; };
	/*! 
     * \brief Time between diagnostic output.
     * \returns \f$ \triangle_{\mathrm{tdiag}} \f$
     */
	inline double get_tdiag() const { return tdiag; };
	/*! 
     * \brief Time between full output.
     * \returns \f$ \triangle_{\mathrm{tout}} \f$
     */
	inline double get_tout() const { return tout; };
	
	/// Use logarithmic formulation of density variable.
	inline bool get_log_theta() const { return log_theta; };
	/// Use logarithmic formulation of temperature variable.
	inline bool get_log_tau() const { return log_tau; };

    inline static bool get_log_omega() { return false; };
	/// Use particle tracking?.
    inline int do_particle_tracking() const { return particle_tracking; };
    /// Number of radial probes
    inline unsigned int get_nprobe() const { return nprobes; };

    /// Number of threads for diagnostic_arrays
    inline unsigned int get_nthreads() const {return nthreads; };

	// Number of particles tracked.
    //int get_particle_tracking() const;	
	/// List of diagnostic functions to run.
	inline vector<twodads::diagnostic_t> get_diagnostics() const { return diagnostics; };
	/// List of variables to include in full output.
    /// theta, theta_x, theta_y, omega[_xy], strmf[_xy]
	inline vector<twodads::output_t> get_output() const { return output; };

	/// Vector of all model parameters.
	inline vector<double> get_model_params() const { return model_params; };
	/// Value of a specific model parameter.
	inline double get_model_params(int i) const { return model_params[i]; };

	/// Vector of initial conditions for theta
	inline vector<double> get_initc_theta() const { return initc_theta; };

	/// Vector of initial conditions for tau
	inline vector<double> get_initc_tau() const { return initc_tau; };

	/// Vector of initial conditions for omega
	inline vector<double> get_initc_omega() const { return initc_omega; };

	/*!
     * \brief Value of a specific initial condition.
     * \returns Value of initial parameter i 
     */
	inline double get_initc_theta(int i) const { return initc_theta[i]; };
	
	/*!
     * \brief Value of a specific initial condition.
     * \returns Value of initial parameter i 
     */
	inline double get_initc_tau(int i) const { return initc_tau[i]; };
	
	/*!
     * \brief Value of a specific initial condition.
     * \returns Value of initial parameter i 
     */
	inline double get_initc_omega(int i) const { return initc_omega[i]; };
	
	/// Return the initialization routine.
	inline string get_init_function_theta_str() const {return init_function_theta_str;};
	inline twodads::init_fun_t get_init_function_theta() const { return init_function_theta; };

	/// Return the initialization routine.
	inline string get_init_function_tau_str() const {return init_function_tau_str;};
	inline twodads::init_fun_t get_init_function_tau() const { return init_function_tau; };

	/// Return the initialization routine.
	inline string get_init_function_omega_str() const {return init_function_omega_str;};
	inline twodads::init_fun_t get_init_function_omega() const { return init_function_omega; };

    /*!
	 \brief Return type of RHS for theta equation. <br>
     \detailed theta_gaussian: \f$\theta(x,y) = i_0 + i_1 \exp( -(x - i_3)^2 / (2 i_4)^2 - (y - i_4)^2 / (2 i_5^2))\f$, \f$ \Omega(x,y) = 0 \f$ <br>
     \detailed omega_gaussian: \f$ \theta(x,y) = 0\f$, \f$\Omega(x,y) = i_0 + i_1 \exp( -(x - i_3)^2 / (2 i_4)^2 - (y - i_4)^2 / (2 i_5^2))\f$ <br>
     \detailed both_gaussian: both of the above <br>
     \detailed theta_sine: \f$ k_x = 2 \pi i_0 / (Nx \triangle_x)\f$, \f$k_y = 2 \pi i_1 (My \triangle_y)\f$, \f$ \theta(x,y) = \sin(k_x x) + \sin(k_y y) \f$ <br>
     \detailed omega_sine: \f$ k_x = 2 \pi i_0 / (Nx \triangle_x)\f$, \f$k_y = 2 \pi i_1 (My \triangle_y)\f$, \f$ \Omega(x,y) = \sin(k_x x) + \sin(k_y y) \f$ <br>
     \detailed both_sine: Both of the above, using same \f$ k_x \f$ and \f$ k_y \f$ for \f$ \theta \f$ and \f$ \Omega \f$ <br>
     \detailed theta_mode: \f$ \widehat{\theta}(n_x, m_y) = \sum\limits_{k = 0}^{\mathrm{length(initc)}/4} i_{k/4} \times \exp \left[ -\frac{(n_x - i_{(k+1)/4})^2}{i_{(k+3)/4}^2} - \frac{(m_y - i_{(k+2)/4})^2}{i_{(k+3)/4}^2} \right] \f$ In input file: initc = mode0 amplitude, mode0 x, mode0 y, mode0 sigma, mode1 amplitude, mode1 x, ...<br>
     \detailed omega_mode: \f$ \widehat{\Omega}(n_x, m_y) = \sum\limits_{k = 0}^{\mathrm{length(initc)}/4} i_{k/4} \times \exp \left[ -\frac{(n_x - i_{(k+1)/4})^2}{i_{(k+3)/4}^2} - \frac{(m_y - i_{(k+2)/4})^2}{i_{(k+3)/4}^2} \right] \f$ In input file: initc = mode0 amplitude, mode0 x, mode0 y, mode0 sigma, mode1 amplitude, mode1 x, ...<br>
     \detailed both_mode: both of the above <br>
    */
	inline twodads::rhs_t get_theta_rhs_type() const { return theta_rhs; };
	/*! 
     * \brief Return type of RHS for omega equation.
     * \returns twodads::rhs_t
     */
	inline twodads::rhs_t get_omega_rhs_type() const { return omega_rhs; };
	/*! 
     * \brief Return type of RHS for tau equation.
     * \returns twodads::rhs_t
     */
	inline twodads::rhs_t get_tau_rhs_type() const { return tau_rhs; };
	
	/// Check configuration consistency.
	int consistency();
	
	/*! 
     * \brief Domain length in x direction.
     */
	inline double get_lengthx() const { return (xright - xleft); };
	/*!
     * \brief Domain length in y direction.
     */
	inline double get_lengthy() const { return (yup - ylow); };
	/*!
     * \brief Grid spacing in x direction.
     */
	inline double get_deltax() const { return( (xright - xleft) / double(Nx) ); };
	/*!
     * \brief Grid spacing in y direction.
     */
	inline double get_deltay() const { return( (yup - ylow) / double(My) ); };

    /*!
     * \brief Prints all parameters
     */

    void print_config() const;

    void test_modes() const;
private:
	// Configuration variables
	unsigned int runnr;
	double xleft;
	double xright;
	double ylow;
	double yup;
	unsigned int My;
	unsigned int Nx;
	unsigned int tlevs;
	string scheme;
	double deltat;
	double tend;
	double tdiag;
	double tout;
	bool log_theta;
    bool log_tau;
	bool do_dealiasing;
    bool do_randomize_modes;
    int particle_tracking;
    unsigned int nprobes;
    unsigned int nthreads;

    twodads::rhs_t theta_rhs;
    twodads::rhs_t omega_rhs;
    twodads::rhs_t tau_rhs;
    twodads::init_fun_t init_function_theta;
    twodads::init_fun_t init_function_tau;
    twodads::init_fun_t init_function_omega;
    string init_function_theta_str;
    string init_function_tau_str;
    string init_function_omega_str;

    vector<twodads::diagnostic_t> diagnostics;
    vector<twodads::output_t> output;

	vector<double> initc_theta;
	vector<double> initc_tau;
	vector<double> initc_omega;

	vector<double> model_params;
	string initial_conditions_str;
    string shift_modes_str;
	int chunksize;
    double diff;
	
    // helper functions
	void split_at_whitespace(const string, vector<double>*);
    // Mappings from values in input.ini to enums in twodads.h
    static const map<std::string, twodads::output_t> output_map;
    static const map<std::string, twodads::diagnostic_t> diagnostic_map;
    static const map<std::string, twodads::init_fun_t> init_func_map;
    static const map<std::string, twodads::rhs_t> rhs_func_map;
};

/*
 * Safely return a value from a map. Catch errors and throw a nice error message
 * Use .at() for map access. Throws out of range errors, when keys are requested that are not initialized
 */

template<typename R>
R map_safe_select(string varname, map<std::string, R> mymap)
{
    R ret_val;
    try{
        ret_val = mymap.at(varname);
    } catch (const std::out_of_range& oor)
    {
        stringstream err_msg;
        err_msg << "Invalid selection" << varname << endl;
        err_msg << "Valid strings are: " << endl;
        for(auto const &it : mymap)
            err_msg << it.first << endl;
        throw out_of_range(err_msg.str());
    }
    return (ret_val);
}

#endif //CONFIG_H
