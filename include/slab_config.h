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
#include <boost/property_tree/json_parser.hpp>
//#include <boost/property_tree/ptree.hpp>

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
    public:
        slab_config_js(std::string); 

        const boost::property_tree::ptree& get_pt()  {return(pt);};

        size_t get_runnr() const {return(pt.get<size_t>("2dads.runnr"));};
        twodads::real_t get_xleft() const {return(pt.get<twodads::real_t>("2dads.geometry.xleft"));};
        twodads::real_t get_xright() const {return(pt.get<twodads::real_t>("2dads.geometry.xright"));};
        twodads::real_t get_Lx() const {return(get_xright() - get_xleft());};
        twodads::real_t get_ylow() const {return(pt.get<twodads::real_t>("2dads.geometry.ylow"));};
        twodads::real_t get_yup() const {return(pt.get<twodads::real_t>("2dads.geometry.yup"));};
        twodads::real_t get_Ly() const {return(get_yup() - get_ylow());};

        size_t get_my() const {return(pt.get<size_t>("2dads.geometry.My"));};
        size_t get_pad_y() const {return(pt.get<size_t>("2dads.geometry.pady"));};
        size_t get_my21() const {return((get_my() + get_pad_y()) / 2 + 1);};
        size_t get_nx() const {return(pt.get<size_t>("2dads.geometry.Nx"));};
        size_t get_pad_x() const {return(pt.get<size_t>("2dads.geometry.padx"));};
        size_t get_tlevs() const {return(pt.get<size_t>("2dads.integrator.level"));};

        twodads::real_t get_deltax() const {return(get_Lx() / twodads::real_t(get_nx()));};
        twodads::real_t get_deltay() const {return(get_Ly() / twodads::real_t(get_my()));};

        twodads::grid_t get_grid_type() const {return(grid_map.at(pt.get<std::string>("2dads.geometry.grid_type")));};
        twodads::dft_t get_dft_t() const;

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
        twodads::real_t get_tend() const {return(pt.get<twodads::real_t>("2dads.integrator.tend"));};
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

        bool check_consistency() const;

    private:
        boost::property_tree::ptree pt;
	    bool log_theta;
        bool log_tau;
	    //bool do_dealiasing;
        //bool do_randomize_modes;
        //bool particle_tracking;
        //size_t nprobes;

        //std::vector<twodads::diagnostic_t> diagnostics;
    
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
