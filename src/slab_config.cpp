/*
 *  slab_config.cpp
 */


#include "slab_config.h"


const std::map<std::string, twodads::output_t> slab_config_js::output_map 
{
    {"theta", twodads::output_t::o_theta},
    {"theta_x", twodads::output_t::o_theta_x},
    {"theta_y", twodads::output_t::o_theta_y},
    {"omega", twodads::output_t::o_omega},
    {"omega_x", twodads::output_t::o_omega_x},
    {"omega_y", twodads::output_t::o_omega_y},
    {"tau", twodads::output_t::o_tau},
    {"tau_x", twodads::output_t::o_tau_x},
    {"tau_y", twodads::output_t::o_tau_y},
    {"strmf", twodads::output_t::o_strmf},
    {"strmf_x", twodads::output_t::o_strmf_x},
    {"strmf_y", twodads::output_t::o_strmf_y}
};

const std::map<std::string, twodads::field_t> slab_config_js::fname_map
{
    {"theta", twodads::field_t::f_theta},
    {"theta_x", twodads::field_t::f_theta_x},
    {"theta_y", twodads::field_t::f_theta_y},
    {"omega", twodads::field_t::f_omega},
    {"omega_x", twodads::field_t::f_omega_x},
    {"omega_y", twodads::field_t::f_omega_y},
    {"tau", twodads::field_t::f_tau},
    {"tau_x", twodads::field_t::f_tau_x},
    {"tau_y", twodads::field_t::f_tau_y},
    {"strmf", twodads::field_t::f_strmf},
    {"strmf_x", twodads::field_t::f_strmf_x},
    {"strmf_y", twodads::field_t::f_strmf_y},
    {"tmp", twodads::field_t::f_tmp},
    {"tmp_x", twodads::field_t::f_tmp_x},
    {"tmp_y", twodads::field_t::f_tmp_y},
    {"theta_rhs", twodads::field_t::f_theta_rhs},
    {"tau_rhs", twodads::field_t::f_tau_rhs},
    {"omega_rhs", twodads::field_t::f_omega_rhs}
};


const std::map<std::string, twodads::dyn_field_t> slab_config_js :: dyn_fname_map
{
    {"theta", twodads::dyn_field_t::f_theta},
    {"omega", twodads::dyn_field_t::f_omega},
    {"tau", twodads::dyn_field_t::f_tau}
};


const std::map<std::string, twodads::diagnostic_t> slab_config_js::diagnostic_map 
{
    {"com_theta", twodads::diagnostic_t::diag_com_theta},
    {"com_tau", twodads::diagnostic_t::diag_com_tau},
    {"max_theta", twodads::diagnostic_t::diag_max_theta},
    {"max_tau", twodads::diagnostic_t::diag_max_tau}
    //{"max_omega", twodads::diagnostic_t::diag_max_omega},
    //{"max_strmf", twodads::diagnostic_t::diag_max_strmf},
    //{"energy", twodads::diagnostic_t::diag_energy},
    //{"energy_ns", twodads::diagnostic_t::diag_energy_ns},
    //{"energy_local", twodads::diagnostic_t::diag_energy_local},
    //{"probes", twodads::diagnostic_t::diag_probes},
    //{"consistency", twodads::diagnostic_t::diag_consistency},
	//{"memory", twodads::diagnostic_t::diag_mem}
};

const std::map<std::string, twodads::init_fun_t> slab_config_js::init_func_map 
{
    {"constant", twodads::init_fun_t::init_constant}, 
    {"gaussian", twodads::init_fun_t::init_gaussian},
    {"lamb_dipole", twodads::init_fun_t::init_lamb_dipole},
    {"mode", twodads::init_fun_t::init_mode},
    {"sine", twodads::init_fun_t::init_sine},
    {"turbulent_bath", twodads::init_fun_t::init_turbulent_bath}    
};
    
const std::map<std::string, twodads::rhs_t> slab_config_js::rhs_func_map 
{
    {"rhs_theta_ns", twodads::rhs_t::rhs_theta_ns},
    {"rhs_theta_lin", twodads::rhs_t::rhs_theta_lin},
    {"rhs_theta_log", twodads::rhs_t::rhs_theta_log},
    {"rhs_theta_hw", twodads::rhs_t::rhs_theta_hw},
    {"rhs_theta_full", twodads::rhs_t::rhs_theta_full}, 
    {"rhs_theta_hwmod", twodads::rhs_t::rhs_theta_hwmod},
    {"rhs_theta_sheath_nlin", twodads::rhs_t::rhs_theta_sheath_nlin},
    {"rhs_theta_null", twodads::rhs_t::rhs_theta_null},
    {"rhs_tas_sheath_nlin", twodads::rhs_t::rhs_tau_sheath_nlin},
    {"rhs_tau_null", twodads::rhs_t::rhs_tau_null},
    {"rhs_omega_ns", twodads::rhs_t::rhs_omega_ns},
    {"rhs_omega_hw", twodads::rhs_t::rhs_omega_hw},
    {"rhs_omega_hwmod", twodads::rhs_t::rhs_omega_hwmod},
    {"rhs_omega_hwzf", twodads::rhs_t::rhs_omega_hwzf},
    {"rhs_omega_ic", twodads::rhs_t::rhs_omega_ic},
    {"rhs_omega_sheath_nlin", twodads::rhs_t::rhs_omega_sheath_nlin}, 
    {"rhs_omega_null", twodads::rhs_t::rhs_omega_null}
};


const std::map<std::string, twodads::bc_t> slab_config_js :: bc_map
{
    {"dirichlet", twodads::bc_t::bc_dirichlet},
    {"neumann", twodads::bc_t::bc_neumann},
    {"periodic", twodads::bc_t::bc_periodic}
};

const std::map<std::string, twodads::grid_t> slab_config_js :: grid_map
{
    {"vertex", twodads::grid_t::vertex_centered},
    {"cell", twodads::grid_t::cell_centered}
};

slab_config_js :: slab_config_js(std::string fname) :
	    log_theta{false},
        log_tau{false}
	    //do_dealiasing{false},
        //do_randomize_modes{false},
        //particle_tracking{false},
        //nprobes{0}
{
    try
    {
        boost::property_tree::read_json(fname, pt);
    }
    catch(std::exception const& e)
    {
        std::cerr << "Could not initialize configuration: " << e.what() << std::endl;
    }
}

twodads::rhs_t slab_config_js :: get_rhs_t(const twodads::dyn_field_t fname) const
{
    auto it = std::find_if(dyn_fname_map.begin(), dyn_fname_map.end(), finder<twodads::dyn_field_t>(fname));
    return(map_safe_select(pt.get<std::string>(std::string("2dads.model.rhs_") + std::get<0>(*it)), 
                           rhs_func_map));
}

twodads::init_fun_t slab_config_js :: get_init_func_t(const twodads::dyn_field_t fname) const
{
    auto it = std::find_if(dyn_fname_map.begin(), dyn_fname_map.end(), finder<twodads::dyn_field_t>(fname));
    return(map_safe_select(pt.get<std::string>(std::string("2dads.initial.init_func_") + std::get<0>(*it)), 
                           init_func_map));
}


std::vector<twodads::real_t> slab_config_js :: get_initc(const twodads::dyn_field_t fname) const
{
    auto it = std::find_if(dyn_fname_map.begin(), dyn_fname_map.end(), finder<twodads::dyn_field_t>(fname));
    std::string nodename = "2dads.initial.initc_" + std::get<0>(*it);

    std::vector<twodads::real_t> res;
    for (auto cell : pt.get_child(nodename))
    {
        res.push_back(cell.second.get_value<twodads::real_t>() );
    }

    return(res);
}


std::vector<twodads::diagnostic_t> slab_config_js :: get_diagnostics() const
{
// Merge conflic with master branch, 2017-05-28
//<<<<<<< HEAD
//	// Test for empty output line
//	//if (get_output().size() == 0) 
//	//	throw config_error("No output variables specified. If no full output is desired, write output = None\n");
//	// Test x domain boundary
//	if (xleft >= xright) 
//		throw config_error("Left domain boundary must be smaller than right domain boundary\n");
//	// Test y domain boundary
//	if (ylow >= yup)
//		throw config_error("Lower domain boundary must be smaller than upper domain boundary\n");
//	// Test initialization routine
//	if (get_init_function_theta() == twodads::init_fun_t::init_NA) 
//		throw config_error("Invalid initialization routine specified for theta in init file\n");
//	if (get_init_function_omega() == twodads::init_fun_t::init_NA) 
//		throw config_error("Invalid initialization routine specified for omega in init file\n");
//	if (get_init_function_tau() == twodads::init_fun_t::init_NA) 
//		throw config_error("Invalid initialization routine specified for tau in init file\n");
//	// Test if logarithmic theta has correct RHS
//	if (get_theta_rhs_type() == twodads::rhs_t::theta_rhs_log && log_theta == false) 
//		throw config_error("Selecting rhs_theta_log as the RHS function must have log_theta = true\n");
//	//if (get_theta_rhs_type() == twodads::rhs_t::theta_rhs_full && log_theta == false) 
//	//	throw config_error("Selecting rhs_theta_full as the RHS function must have log_theta = true\n");
//	if (get_theta_rhs_type() == twodads::rhs_t::theta_rhs_lin && log_theta == true) 
//		throw config_error("Selecting rhs_theta as the RHS function must have log_theta = false\n");
//    if (get_omega_rhs_type() == twodads::rhs_t::omega_rhs_sheath_nlin && (!get_log_theta() || !get_log_tau()))
//        throw config_error("Selecting omega_rhs_sheath_nlin requires logarithmic density and temperature\n");
//	// If radial probe diagnostics is specified we need the number of radial probes
//	if (nprobes == 0 ) 
//		throw config_error("Radial probe diagnostics requested but no number of radial probes specified\n");
//            
//	return (0);
//=======
    std::vector<twodads::diagnostic_t> res;
    for(auto it : pt.get_child("2dads.diagnostics.routines"))
        res.push_back(map_safe_select(it.second.data(), diagnostic_map));

    return(res); 
//>>>>>>> bc
}


std::vector<twodads::output_t> slab_config_js :: get_output() const
{
    std::vector<twodads::output_t> res;
    for(auto it: pt.get_child("2dads.output.fields"))
        res.push_back(map_safe_select(it.second.data(), output_map));

    return(res);
}


std::vector<twodads::real_t> slab_config_js :: get_model_params(const twodads::dyn_field_t fname) const
{
    auto it = std::find_if(dyn_fname_map.begin(), dyn_fname_map.end(), finder<twodads::dyn_field_t>(fname));
    std::string node_name = "2dads.model.parameters_" + std::get<0>(*it);

    std::vector<twodads::real_t> res;
    for(auto cell : pt.get_child(node_name))
        res.push_back(cell.second.get_value<twodads::real_t>());

    return(res);
}


twodads::dft_t slab_config_js :: get_dft_t() const
{
    switch(get_grid_type())
    {
        case twodads::grid_t::vertex_centered:
            return twodads::dft_t::dft_2d;
        case twodads::grid_t::cell_centered:
            return twodads::dft_t::dft_1d;
    }
}

twodads::bvals_t<twodads::real_t> slab_config_js :: get_bvals(const twodads::field_t fname) const
{
    // Emulate map reverse look-up:
    // See // http://stackoverflow.com/questions/5749073/reverse-map-lookup
    auto it = std::find_if(fname_map.begin(), fname_map.end(), finder<twodads::field_t>(fname));

    // Build and create a boundary value structure
    twodads::bvals_t<twodads::real_t> bvals(
        map_safe_select(pt.get<std::string>(std::string("2dads.geometry.") + std::get<0>(*it) + std::string("_bc_left")), bc_map),
        map_safe_select(pt.get<std::string>(std::string("2dads.geometry.") + std::get<0>(*it) + std::string("_bc_right")), bc_map),
        pt.get<twodads::real_t>(std::string("2dads.geometry.") + std::get<0>(*it) + std::string("_bval_left")),
        pt.get<twodads::real_t>(std::string("2dads.geometry.") + std::get<0>(*it)+ std::string("_bval_right"))
    );
    
    return(bvals);
};


twodads::stiff_params_t slab_config_js :: get_tint_params(const twodads::dyn_field_t fname) const
{
    auto it = std::find_if(dyn_fname_map.begin(), dyn_fname_map.end(), finder<twodads::dyn_field_t>(fname));

    // Get the diffusion parameter, the first entry in the parameters_fname vector
    auto mparams = pt.get_child(std::string("2dads.model.parameters_" + std::get<0>(*it)));
    std::vector<twodads::real_t> res;
    for(auto cell : mparams)
        res.push_back(cell.second.get_value<twodads::real_t>());

    twodads::stiff_params_t sp(
        get_deltat(), get_Lx(), get_Ly(),
        res[0],
        pt.get<twodads::real_t>("2dads.integrator.hypervisc"), 
        get_nx(), get_my21(), get_tlevs()
    );
    return(sp);
}


bool slab_config_js :: check_consistency() const
{
    // When using periodic boundary conditions in the x-direction a cell-centered grid
    // may not be used. Use a cell-centered grid here

    if(get_grid_type() == twodads::grid_t::cell_centered &&
       (get_bvals(twodads::field_t::f_theta).get_bc_left() == twodads::bc_t::bc_periodic 
        ||
        get_bvals(twodads::field_t::f_theta).get_bc_right() == twodads::bc_t::bc_periodic
        ||
        get_bvals(twodads::field_t::f_omega).get_bc_left() == twodads::bc_t::bc_periodic
        ||
        get_bvals(twodads::field_t::f_omega).get_bc_right() == twodads::bc_t::bc_periodic
        ||
        get_bvals(twodads::field_t::f_strmf).get_bc_left() == twodads::bc_t::bc_periodic
        ||
        get_bvals(twodads::field_t::f_strmf).get_bc_right() == twodads::bc_t::bc_periodic
        ||
        get_bvals(twodads::field_t::f_tau).get_bc_left() == twodads::bc_t::bc_periodic
        ||
        get_bvals(twodads::field_t::f_tau).get_bc_left() == twodads::bc_t::bc_periodic))
    {
        throw config_error(std::string("Periodic boundary conditions in the x-direction are not allowed when using a cell-centered grid. Use a vertex centered grid"));
    }

    if(get_grid_type() == twodads::grid_t::vertex_centered &&
        (get_bvals(twodads::field_t::f_theta).get_bc_left() != twodads::bc_t::bc_periodic
         ||
         get_bvals(twodads::field_t::f_theta).get_bc_right() != twodads::bc_t::bc_periodic
         ||
         get_bvals(twodads::field_t::f_omega).get_bc_left() != twodads::bc_t::bc_periodic
         ||
         get_bvals(twodads::field_t::f_omega).get_bc_right() != twodads::bc_t::bc_periodic
         ||
         get_bvals(twodads::field_t::f_strmf).get_bc_left() != twodads::bc_t::bc_periodic
         ||
         get_bvals(twodads::field_t::f_strmf).get_bc_right() != twodads::bc_t::bc_periodic
         ||
         get_bvals(twodads::field_t::f_tau).get_bc_left() != twodads::bc_t::bc_periodic
         ||
         get_bvals(twodads::field_t::f_tau).get_bc_right() != twodads::bc_t::bc_periodic))
    {
        throw config_error(std::string("Periodic boundary conditions in the x-direction are required when using a vertex-centered\n"));
    }


    assert(get_my() % 4 == 0);
    assert(get_nx() % 4 == 0);

    return(true);
}

// End of file slab_config.cpp
