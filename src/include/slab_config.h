/*
 * slab_config.h
 *
 * Run configuration is stored in json format.
 * This class parses the json file and builds the structures defined in 2dads_types.h
 * from the json file when called.
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



class slab_config_js
{
    /**
     .. cpp:class:: slab_config_js

     Store slab configuration.

    */
    public:
        slab_config_js(std::string); 

        const boost::property_tree::ptree& get_pt() const  {return(pt);};

        /**
         .. cpp:function:: size_t get_runnr() const

         Returns the run number.
        
        */
        size_t get_runnr() const {return(pt.get<size_t>("2dads.runnr"));};

        /**
         .. cpp:function:: twodads::real_t get_xleft() const

         Returns the location of the left simulation domain boundary.

        */
        twodads::real_t get_xleft() const {return(pt.get<twodads::real_t>("2dads.geometry.xleft"));};

        /**
         .. cpp:function:: twodads::real_t get_xright() const

         Returns the location of the right simulation domain boundary.

        */
        twodads::real_t get_xright() const {return(pt.get<twodads::real_t>("2dads.geometry.xright"));};

        /**
         .. cpp:function:: twodads::real_t get_Lx() const

         Returns the length of the simulation domain in the x-direction.

        */
        twodads::real_t get_Lx() const {return(get_xright() - get_xleft());};

        /**
         .. cpp:function:: twodads::real_t get_ylow() const

         Returns the location of the lower simulation domain boundary.

        */
        twodads::real_t get_ylow() const {return(pt.get<twodads::real_t>("2dads.geometry.ylow"));};

        /**
         .. cpp:function:: twodads::real_T get_yup() const

         Returns the location of the upper simulation domain boundary.

        */
        twodads::real_t get_yup() const {return(pt.get<twodads::real_t>("2dads.geometry.yup"));};

        /**
         .. cpp:function:: twodads::real_t get_Ly() const

         Returns the length of the simulation domain in the y-direction.

        */
        twodads::real_t get_Ly() const {return(get_yup() - get_ylow());};

        /**
         .. cpp:function:: size_t get_my() const

         Returns the number of discretization points in the y-direction.

        */
        size_t get_my() const {return(pt.get<size_t>("2dads.geometry.My"));};

        /**
         .. cpp:function:: size_t get_pad_y() const

         Returns the number of padding elements in the y-direction.

        */
        size_t get_pad_y() const {return(pt.get<size_t>("2dads.geometry.pady"));};

        /**
         .. cpp:function:: size_t get_my21() const

         Returns my / 2 + 1.

        */
        size_t get_my21() const {return((get_my() + get_pad_y()) / 2 + 1);};

        /**
         .. cpp:function:: size_t get_nx() const

         Returns the number of discretization points in the x-direction.

        */
        size_t get_nx() const {return(pt.get<size_t>("2dads.geometry.Nx"));};

        /**
         .. cpp:function:: size_t get_pad_x() const

         Returns the number of padding elements in the x-direction.

        */
        size_t get_pad_x() const {return(pt.get<size_t>("2dads.geometry.padx"));};

        /**
         .. cpp:function:: size_t get_tlevs() const

         Returns the number of previous time steps that needs to be stored in the dynamical fields.

        */
        size_t get_tlevs() const {return(pt.get<size_t>("2dads.integrator.level"));};

        /**
         .. cpp:function:: twodads::real_t get_deltax() const

         Returns the distance between two discretization points in the x-direction.

        */
        twodads::real_t get_deltax() const {return(get_Lx() / twodads::real_t(get_nx()));};

        /**
         .. cpp:function:: twodads::real_t get_deltay() const

         Returns the distance between two discretization points in the y-direction.

        */
        twodads::real_t get_deltay() const {return(get_Ly() / twodads::real_t(get_my()));};

        /**
         .. cpp:function:: twodads::grid_t get_grid_type() const

         Returns the grid type to be used.

        */
        twodads::grid_t get_grid_type() const {return(grid_map.at(pt.get<std::string>("2dads.geometry.grid_type")));};

        /**
         .. cpp:function:: twodads::dft_t get_dft_t() const

         Returns the DFT type to be used (1d/2d).
        
        */
        twodads::dft_t get_dft_t() const;

        /**
         .. cpp:function:: twodads::bvals_t<twodads::real_t> get_bvals(const twodads::field_t fname) const

          :param const field_t fname: Name of the field.

         Returns the boundary condition for the specified field.

        */
        twodads::bvals_t<twodads::real_t> get_bvals(const twodads::field_t fname) const;

        /**
         .. cpp:function:: twodads::stiff_params_t get_tint_params(const twodads::dyn_field_t fname) const

          :param const dyn_field_t fname: Dynamic field name

        Returns the parameters for karniadakis time integration for the given field.

        */
        twodads::stiff_params_t get_tint_params(const twodads::dyn_field_t) const;

        /**
         .. cpp:function:: slab_layout_t get_geom() const

          Returns a slab_layout_t for the defined simulation parameters.
        
        */
        twodads::slab_layout_t get_geom() const
        {
            twodads::slab_layout_t sl(get_xleft(), get_deltax(), get_ylow(), get_deltay(),
                                      get_nx(), get_pad_x(), get_my(), get_pad_y(), get_grid_type());
            return(sl);
        };

        std::string get_scheme() const {return(pt.get<std::string>("2dads.integrator.scheme"));};

        /**
         .. cpp:function:: twodads::real_t get_deltat() const

         Returns the size of the time step.

        */
        twodads::real_t get_deltat() const {return(pt.get<twodads::real_t>("2dads.integrator.deltat"));};

        /**
         .. cpp:function:: twodads::real_t get_tend() const

         Returns the end time for time integration.

        */
        twodads::real_t get_tend() const {return(pt.get<twodads::real_t>("2dads.integrator.tend"));};

        /** 
         .. cpp:function:: twodads::real_t get_tdiag() const

         Returns the time step for diagnostic output.
        
        */
        twodads::real_t get_tdiag() const {return(pt.get<twodads::real_t>("2dads.diagnostics.tdiag"));};

        /**
         .. cpp:function:: twodads::real_t get_tout() const

         Returns the time step for full output.

        */
        twodads::real_t get_tout() const {return(pt.get<twodads::real_t>("2dads.output.tout"));};

        /**
         .. cpp:function:: bool get_log_theta() const

         Returns true if a logarithmic formulation for the density field is to be used.
         Otherwise returns false.

        */
        bool get_log_theta() const {return(log_theta);};

        /**
         .. cpp:function:: bool get_log_tau() const

         Returns true if a logarithmic formulation for the temperature field is to be used.
         Otherwise returns false.

        */
        bool get_log_tau() const {return(log_tau);};

        /**
         .. cpp:function:: twodads::rhs_t get_rhs_t(const twodads::dyn_field_t fname) const

         :param const dyn_field_t fname: Name of the field.

         Returns the type of RHS for the given field name.

        */
        twodads::rhs_t get_rhs_t(const twodads::dyn_field_t) const;

        /**
         .. cpp:function:: twodads::init_fun_t get_init_func_t(const twodads::dyn_field_t fname) const

         :param const dyn_field_t fname: Name of the field.

         Returns the initial function for the given field name.
         
        */
        twodads::init_fun_t get_init_func_t(const twodads::dyn_field_t) const;

        /**
         .. cpp:function:: std::vector<twodads::real_t> get_initc(const twodads::dyn_field_t fname) const

         :param const dyn_field_t fname: Name of the field

        Returns a vector with parameter values for the initial function.

        */
        std::vector<twodads::real_t> get_initc(const twodads::dyn_field_t) const;

        /**
         .. cpp:function:: std::vector<twodads::diagnostic_t> get_diagnostics() const

         Returns a vector with the specified diagnostic functions.
        
        */
        std::vector<twodads::diagnostic_t> get_diagnostics() const;

        /**
         .. cpp:function:: std::vector<twodads::output_t> get_output() const

         Returns a vector with the specified fields to write to the binary output file.

        */
        std::vector<twodads::output_t> get_output() const;

        /**
         .. cpp:function:: std::vector<twodads::real_t> get_model_params(const twodads::dyn_field_t fname) const

            :param const dyn_field_t fname: Name of the field.

         Returns a vector with the parameter values for the RHS of fname.
        */
        std::vector<twodads::real_t> get_model_params(const twodads::dyn_field_t) const;

        /**
         .. cpp:function:: bool check_consistency() const

        Checks simulation configuration for self-consistency.
        */
        bool check_consistency() const;

    private:
        boost::property_tree::ptree pt;
	    bool log_theta;
        bool log_tau;
    
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

#endif //CONFIG_H
