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
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/filesystem.hpp>


#include "2dads_types.h"
#include "error.h"

// For inverse map lookups
// http://stackoverflow.com/questions/5749073/reverse-map-lookup
template<typename T>
class finder
{
    public:
        finder(const T _val) : val(_val) {}
        T get_val() const {return(val);};

        template <typename U>
        bool operator()(const std::pair<U, T> cmp) const
        {
            return (std::get<1>(cmp) == get_val());
        }
    private:
        const T val;
};



class slab_config_js
{
    public:
        slab_config_js(std::string); 

        boost::property_tree::ptree& get_pt()  {return(pt);};

        size_t get_runnr() const {return(pt.get<size_t>("2dads.runnr"));};
        twodads::real_t get_xleft() const {return(pt.get<twodads::real_t>("2dads.geometry.xleft"));};
        twodads::real_t get_xright() const {return(pt.get<twodads::real_t>("2dads.geometry.xright"));};
        twodads::real_t get_Lx() const {return(get_xright() - get_xleft());};
        twodads::real_t get_ylow() const {return(pt.get<twodads::real_t>("2dads.geometry.ylow"));};
        twodads::real_t get_yup() const {return(pt.get<twodads::real_t>("2dads.geometry.yup"));};
        twodads::real_t get_Ly() const {return(get_ylow() - get_yup());};

        size_t get_my() const {return(pt.get<size_t>("2dads.geometry.My"));};
        size_t get_pad_y() const {return(pt.get<size_t>("2dads.geometry.pady"));};
        size_t get_my21() const {return((get_my() + get_pad_y()) / 2 + 1);};
        size_t get_nx() const {return(pt.get<size_t>("2dads.geometry.Nx"));};
        size_t get_pad_x() const {return(pt.get<size_t>("2dads.geometry.padx"));};
        size_t get_tlevs() const {return(pt.get<size_t>("2dads.integrator.level"));};

        twodads::real_t get_deltax() const {return(twodads::real_t(get_nx()) / get_Lx());};
        twodads::real_t get_deltay() const {return(twodads::real_t(get_my()) / get_Ly());};

        twodads::grid_t get_grid_type() const {return(grid_map.at(pt.get<std::string>("2dads.geometry.grid_type")));};
        
        twodads::bvals_t<twodads::real_t> get_bvals(const twodads::field_t fname) const;

        twodads::stiff_params_t get_tint_params(const twodads::dyn_field_t) const;

        twodads::slab_layout_t get_geom() const
        {
            twodads::slab_layout_t sl(get_xleft(), get_deltax(), get_ylow(), get_deltay(),
                                      get_nx(), get_pad_x(), get_my(), get_pad_y(), get_grid_type());
            return(sl);
        };

        std::string get_scheme() const {return(pt.get<std::string>("2dads.integrator.scheme"));};
        twodads::real_t get_deltat() const {return(pt.get<twodads::real_t>("2dads.integrator.deltat"));};
        twodads::real_t get_tend() const {return(pt.get<twodads::real_t>("2dads.integrator.deltat"));};
        twodads::real_t get_tdiag() const {return(pt.get<twodads::real_t>("2dads.diagnostics.xleft"));};
        twodads::real_t get_tout() const {return(pt.get<twodads::real_t>("2dads.output.tout"));};

        bool get_log_theta() const {return(log_theta);};
        bool get_log_tau() const {return(log_tau);};

        twodads::rhs_t get_rhs_t(const twodads::dyn_field_t) const;
        twodads::init_fun_t get_init_func_t(const twodads::dyn_field_t) const;
        std::vector<twodads::real_t> get_initc(const twodads::dyn_field_t) const;

        std::vector<twodads::diagnostic_t> get_diagnostics() const;
        std::vector<twodads::output_t> get_output() const;
        std::vector<twodads::real_t> get_model_params() const;

    private:
        boost::property_tree::ptree pt;
	    bool log_theta;
        bool log_tau;
	    //bool do_dealiasing;
        //bool do_randomize_modes;
        //bool particle_tracking;
        //size_t nprobes;

        //std::vector<twodads::diagnostic_t> diagnostics;
        //std::vector<twodads::output_t> output;
	    //std::vector<twodads::real_t> model_params;
    
        // Mappings from values in input.ini to enums in twodads.h
        static const std::map<std::string, twodads::field_t> fname_map;
        static const std::map<std::string, twodads::dyn_field_t> dyn_fname_map;
        static const std::map<std::string, twodads::output_t> output_map;
        static const std::map<std::string, twodads::diagnostic_t> diagnostic_map;
        static const std::map<std::string, twodads::init_fun_t> init_func_map;
        static const std::map<std::string, twodads::rhs_t> rhs_func_map;
        static const std::map<std::string, twodads::bc_t> bc_map;
        static const std::map<std::string, twodads::grid_t> grid_map;
};


/*! \brief store simulation domain information
 *  \ingroup configuration
 */

class slab_config {

public:
    /*!
     * \brief Standard constructor reads input.ini from pwd,
     * call slab_config('.') as delegating constructor
     */
    slab_config() : slab_config(std::string("input.ini")) {};

    /*
     * \brief Constructor, reads input.ini in directory, specified by argument
     */
    slab_config(std::string);

	~slab_config() {};

    // Inspectors for private members
	inline size_t get_runnr() const { return runnr; };
	/*! 
     * \brief Left domain boundary.
     * \return Left domain boundary (twodads::real_t)
     */
	inline twodads::real_t get_xleft() const { return xleft; };
	/*! 
     * \brief Right domain boundary.
     * \return Right domain boundary (twodads::real_t)
     */
	inline twodads::real_t get_xright() const { return xright; };
	/*! 
     * \brief Upper domain boundary.
     * \returns Upper domain boundary (twodads::real_t)
     */
	inline twodads::real_t get_yup() const { return yup; };
	/*! 
     * \brief Lower domain boundary.
     * \returns Lower domain bonudary (twodads::real_t)
     */
	inline twodads::real_t get_ylow() const { return ylow; };
	/*! 
     * \brief Number of grid points in x direction.
     * \returns Number of grid points in x direction
     */
	inline size_t get_nx() const { return Nx; };
	/*!
     * \brief Number of grid points in y direction.
     * \returns Number of grid points in y direction
     */
	inline size_t get_my() const { return My; };
	/*! 
     * \brief Order of time integration scheme.
     * \returns (currently only 4th order Karniadakis scheme is supported
     */
	inline size_t get_tlevs() const { return tlevs; };
	/*!
     * \brief Length of one timestep.
     * \returns \f$ \mathrm{d}t \f$
     */
	inline twodads::real_t get_deltat() const { return deltat; };
	/*! 
     * \brief Simulation time end
     * \returns \f$ T_{\mathrm{end}} \f$
     */
	inline twodads::real_t get_tend() const { return tend; };
	/*! 
     * \brief Time between diagnostic output.
     * \returns \f$ \triangle_{\mathrm{tdiag}} \f$
     */
	inline twodads::real_t get_tdiag() const { return tdiag; };
	/*! 
     * \brief Time between full output.
     * \returns \f$ \triangle_{\mathrm{tout}} \f$
     */
	inline twodads::real_t get_tout() const { return tout; };
	
	/// Use logarithmic formulation of density variable.
	inline bool get_log_theta() const { return log_theta; };
	/// Use logarithmic formulation of temperature variable.
	inline bool get_log_tau() const { return log_tau; };

    inline static bool get_log_omega() { return false; };
	/// Use particle tracking?.
    inline int do_particle_tracking() const { return particle_tracking; };
    /// Number of radial probes
    inline size_t get_nprobe() const { return nprobes; };

    /// Number of threads for diagnostic_arrays
    inline size_t get_nthreads() const {return nthreads; };

	// Number of particles tracked.
    //int get_particle_tracking() const;	
	/// List of diagnostic functions to run.
	inline std::vector<twodads::diagnostic_t> get_diagnostics() const { return diagnostics; };
	/// List of variables to include in full output.
    /// theta, theta_x, theta_y, omega[_xy], strmf[_xy]
	inline std::vector<twodads::output_t> get_output() const { return output; };

	/// Vector of all model parameters.
	inline std::vector<twodads::real_t> get_model_params() const { return model_params; };
	/// Value of a specific model parameter.
	inline twodads::real_t get_model_params(int i) const { return model_params[i]; };

	/// Vector of initial conditions for theta
	inline std::vector<twodads::real_t> get_initc_theta() const { return initc_theta; };

	/// Vector of initial conditions for tau
	inline std::vector<twodads::real_t> get_initc_tau() const { return initc_tau; };

	/// Vector of initial conditions for omega
	inline std::vector<twodads::real_t> get_initc_omega() const { return initc_omega; };

	/*!
     * \brief Value of a specific initial condition.
     * \returns Value of initial parameter i 
     */
	inline twodads::real_t get_initc_theta(int i) const { return initc_theta[i]; };
	
	/*!
     * \brief Value of a specific initial condition.
     * \returns Value of initial parameter i 
     */
	inline twodads::real_t get_initc_tau(int i) const { return initc_tau[i]; };
	
	/*!
     * \brief Value of a specific initial condition.
     * \returns Value of initial parameter i 
     */
	inline twodads::real_t get_initc_omega(int i) const { return initc_omega[i]; };
	
	/// Return the initialization routine.
	inline std::string get_init_function_theta_str() const {return init_function_theta_str;};
	inline twodads::init_fun_t get_init_function_theta() const { return init_function_theta; };

	/// Return the initialization routine.
	inline std::string get_init_function_tau_str() const {return init_function_tau_str;};
	inline twodads::init_fun_t get_init_function_tau() const { return init_function_tau; };

	/// Return the initialization routine.
	inline std::string get_init_function_omega_str() const {return init_function_omega_str;};
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
	inline twodads::real_t get_lengthx() const { return (xright - xleft); };
	/*!
     * \brief Domain length in y direction.
     */
	inline twodads::real_t get_lengthy() const { return (yup - ylow); };
	/*!
     * \brief Grid spacing in x direction.
     */
	inline twodads::real_t get_deltax() const { return( (xright - xleft) / twodads::real_t(Nx) ); };
	/*!
     * \brief Grid spacing in y direction.
     */
	inline twodads::real_t get_deltay() const { return( (yup - ylow) / twodads::real_t(My) ); };

    /*!
     * \brief Prints all parameters
     */

    void print_config() const;

    void test_modes() const;
private:
	// Configuration variables
	size_t runnr;
	twodads::real_t xleft;
	twodads::real_t xright;
	twodads::real_t ylow;
	twodads::real_t yup;
	size_t My;
	size_t Nx;
	size_t tlevs;
	std::string scheme;
	twodads::real_t deltat;
	twodads::real_t tend;
	twodads::real_t tdiag;
	twodads::real_t tout;
	bool log_theta;
    bool log_tau;
	bool do_dealiasing;
    bool do_randomize_modes;
    int particle_tracking;
    size_t nprobes;
    size_t nthreads;

    twodads::rhs_t theta_rhs;
    twodads::rhs_t omega_rhs;
    twodads::rhs_t tau_rhs;
    twodads::init_fun_t init_function_theta;
    twodads::init_fun_t init_function_tau;
    twodads::init_fun_t init_function_omega;
    std::string init_function_theta_str;
    std::string init_function_tau_str;
    std::string init_function_omega_str;

    std::vector<twodads::diagnostic_t> diagnostics;
    std::vector<twodads::output_t> output;

	std::vector<twodads::real_t> initc_theta;
	std::vector<twodads::real_t> initc_tau;
	std::vector<twodads::real_t> initc_omega;

	std::vector<twodads::real_t> model_params;
	std::string initial_conditions_str;
    std::string shift_modes_str;
	int chunksize;
    twodads::real_t diff;
	
    // helper functions
	void split_at_whitespace(const std::string, std::vector<twodads::real_t>*);
    // Mappings from values in input.ini to enums in twodads.h
    static const std::map<std::string, twodads::output_t> output_map;
    static const std::map<std::string, twodads::diagnostic_t> diagnostic_map;
    static const std::map<std::string, twodads::init_fun_t> init_func_map;
    static const std::map<std::string, twodads::rhs_t> rhs_func_map;
};

/*
 * Safely return a value from a map. Catch errors and throw a nice error message
 * Use .at() for map access. Throws out of range errors, when keys are requested that are not initialized
 */

template<typename R>
R map_safe_select(std::string varname, std::map<std::string, R> mymap)
{
    R ret_val;
    try{
        ret_val = mymap.at(varname);
    } catch (const std::out_of_range& oor)
    {
        std::stringstream err_msg;
        err_msg << "Invalid selection: " << varname << std::endl;
        err_msg << "Valid strings are: " << std::endl;
        for(auto const &it : mymap)
            err_msg << it.first << std::endl;
        throw std::out_of_range(err_msg.str());
    }
    return (ret_val);
}

#endif //CONFIG_H
