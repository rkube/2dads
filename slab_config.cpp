/*
 *  config.cpp
 *  2dads-oo
 *
 *  Created by Ralph Kube on 09.12.10.
 *  Copyright 2010 __MyCompanyName__. All rights reserved.
 *
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/filesystem.hpp>
#include "slab_config.h"
#include "error.h"
#include "2dads_types.h"

using namespace std;
namespace po = boost::program_options;
namespace fs = boost::filesystem;

// Split string params at whitespaces and put values in vector values
void slab_config :: split_at_whitespace(const string params_str, vector<double>* result)
{

	// Split string params into substrings which are separated by whitespace characters
	vector<string> split_vector;
	boost::split(split_vector, params_str, boost::is_any_of(" \t"), boost::token_compress_on);
	
	// Iterate over split_vector, cast items to vector and push back in result
	for(vector<string>::const_iterator it = split_vector.begin(); it != split_vector.end(); it++){
		try{
			result -> push_back(boost::lexical_cast<double> (*it) );
		} 
		catch ( boost::bad_lexical_cast& ) {
			cerr << "Could not cast model parameter " << (*it) << " to double\n";
		}
	}		
}


const map<string, twodads::output_t> slab_config::output_map 
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
    {"strmf_y", twodads::output_t::o_strmf_y},
    {"theta_rhs", twodads::output_t::o_theta_rhs},
    {"omega_rhs", twodads::output_t::o_omega_rhs}
};

const map<string, twodads::diagnostic_t> slab_config::diagnostic_map 
{
    {"blobs", twodads::diagnostic_t::diag_blobs},
    {"com_theta", twodads::diagnostic_t::diag_com_theta},
    {"com_tau", twodads::diagnostic_t::diag_com_tau},
    {"max_theta", twodads::diagnostic_t::diag_max_theta},
    {"max_tau", twodads::diagnostic_t::diag_max_tau},
    {"max_omega", twodads::diagnostic_t::diag_max_omega},
    {"max_strmf", twodads::diagnostic_t::diag_max_strmf},
    {"energy", twodads::diagnostic_t::diag_energy},
    {"energy_ns", twodads::diagnostic_t::diag_energy_ns},
    {"energy_local", twodads::diagnostic_t::diag_energy_local},
    {"probes", twodads::diagnostic_t::diag_probes},
    {"consistency", twodads::diagnostic_t::diag_consistency},
	{"memory", twodads::diagnostic_t::diag_mem}
};

const map<string, twodads::init_fun_t> slab_config::init_func_map 
{
    {"gaussian", twodads::init_fun_t::init_gaussian},
    {"constant", twodads::init_fun_t::init_constant}, 
    {"sine", twodads::init_fun_t::init_sine},
    {"mode", twodads::init_fun_t::init_mode},
    {"turbulent_bath", twodads::init_fun_t::init_turbulent_bath},
    {"lamb_dipole", twodads::init_fun_t::init_lamb_dipole}
};
    
const map<string, twodads::rhs_t> slab_config::rhs_func_map 
{
    {"theta_rhs_ns", twodads::rhs_t::theta_rhs_ns},
    {"theta_rhs_lin", twodads::rhs_t::theta_rhs_lin},
    {"theta_rhs_log", twodads::rhs_t::theta_rhs_log},
    {"theta_rhs_hw", twodads::rhs_t::theta_rhs_hw},
    {"theta_rhs_full", twodads::rhs_t::theta_rhs_full}, 
    {"theta_rhs_hwmod", twodads::rhs_t::theta_rhs_hwmod},
    {"theta_rhs_sheath_nlin", twodads::rhs_t::theta_rhs_sheath_nlin},
    {"theta_rhs_null", twodads::rhs_t::theta_rhs_null},
    {"tau_rhs_sheath_nlin", twodads::rhs_t::tau_rhs_sheath_nlin},
    {"tau_rhs_null", twodads::rhs_t::tau_rhs_null},
    {"omega_rhs_ns", twodads::rhs_t::omega_rhs_ns},
    {"omega_rhs_hw", twodads::rhs_t::omega_rhs_hw},
    {"omega_rhs_hwmod", twodads::rhs_t::omega_rhs_hwmod},
    {"omega_rhs_hwzf", twodads::rhs_t::omega_rhs_hwzf},
    {"omega_rhs_ic", twodads::rhs_t::omega_rhs_ic},
    {"omega_rhs_sheath_nlin", twodads::rhs_t::omega_rhs_sheath_nlin}, 
    {"omega_rhs_null", twodads::rhs_t::omega_rhs_null}
};


// Parse input from input.ini
//config :: config( string filename, int foo ) :
slab_config :: slab_config(string cfg_file) :
	runnr(0), xleft(0.0), xright(0.0), ylow(0.0), yup(0.0),
	My(0), Nx(0), tlevs(0), deltat(0), tend(0), tdiag(0.0),
	tout(0.0), log_theta(false), log_tau(false), do_dealiasing(false), particle_tracking(0), 
	nprobes(0), chunksize(0)
{
	po::options_description cf_opts("Allowed options");
	po::variables_map vm;

    //fs::path dir(basedir);
    //fs::path file("input.ini");
    fs::path dir(string("."));
    fs::path file(cfg_file);
    fs::path full_path = dir / file;
    cout << "\tfs::path = " << full_path << endl;

	// Temporary strings that need to be parsed
	string model_params_str;
	string initc_theta_str;
	string initc_tau_str;
	string initc_omega_str;
	string diagnostics_str;
	string output_str;
    string theta_rhs_str;
    string tau_rhs_str;
    string omega_rhs_str;
    string init_function_str;
    string strmf_solver_str;    

    // Add recognized options to cf_opts
	try{
	cf_opts.add_options()
		("runnr", po::value<unsigned int>(&runnr), "run number")
		("xleft", po::value<double>(&xleft), "left x boundary")
		("xright", po::value<double>(&xright), "right x boundary")
		("ylow", po::value<double>(&ylow), "lower y boundary")
		("yup", po::value<double>(&yup), "upper y boundary")
		("Nx", po::value<unsigned int>(&Nx), "Resolution in x direction")
		("My", po::value<unsigned int>(&My), "Resolution in y direction")
		("scheme", po::value<string>(&scheme), "Time integration scheme")
		("tlevs", po::value<unsigned int>(&tlevs), "Precision of time integration scheme")
		("deltat", po::value<double>(&deltat), "Time step length")
		("tend", po::value<double>(&tend), "End of simulation")
		("tdiag", po::value<double>(&tdiag), "Distance between diagnostic output")
		("tout", po::value<double>(&tout), "Distance between full output")
        ("nthreads", po::value<unsigned int>(&nthreads), "Number of threads for diagnostic arrays")
		("model_params", po::value<string> (&model_params_str) , "Model parameters")
		("theta_rhs", po::value<string> (&theta_rhs_str), "Right hand side function of theta")
		("tau_rhs", po::value<string> (&tau_rhs_str), "Right hand side function of tau")
		("omega_rhs", po::value<string> (&omega_rhs_str), "Right hand side function of omega")
		("diagnostics", po::value<string> (&diagnostics_str), "List of diagnostic functions to use")
		("output", po::value<string> (&output_str), "List of variables for output")
		("init_function_theta", po::value<string> (&init_function_theta_str), "Initialization function for theta")
		("init_function_omega", po::value<string> (&init_function_omega_str), "Initialization function for omega")
		("init_function_tau", po::value<string> (&init_function_tau_str), "Initialization function for tau")
		("initial_conditions_theta", po::value<string> (&initc_theta_str) , "Initial conditions for theta")
		("initial_conditions_omega", po::value<string> (&initc_omega_str) , "Initial conditions for omega")
		("initial_conditions_tau", po::value<string> (&initc_tau_str) , "Initial conditions for tau")
		("chunksize", po::value<int> (&chunksize), "Chunksize")
		("log_theta", po::value<bool> (&log_theta), "Use logarithmic particle density")
		("log_tau", po::value<bool> (&log_tau), "Use logarithmic temperature")
        ("strmf_solver", po::value<string> (&strmf_solver_str), "Stream function solver")
        ("do_particle_tracking", po::value<int> (&particle_tracking), "Track particles")
        ("dealias", po::value<bool> (&do_dealiasing), "Set ambiguous frequencies to zero")
        ("nprobes", po::value<unsigned int> (&nprobes), "Number of radial probes");
        //("shift_modes", po::value<string> (&shift_modes_str), "Modes whose phase is shifted randomly");
	}
	catch(exception& e)
    {
		cout << e.what() << "\n";
	}
   
    cout << "init_function_str = " << init_function_str << endl; 
	// Parse configuration file
	ifstream cf_stream(full_path.string());
	if (!cf_stream )
    {
		cerr << "slab_config::slab_config(string): Could not open config file: " << full_path.string() << "\n";
        exit(1);
	}
    else 
    {
        po::store( parse_config_file(cf_stream, cf_opts ), vm);
		po::notify(vm);
	}
	
	// Split model_params and initial_condition strings at whitespaces, 
	// convert substrings into floats and store in approproate vectors
    split_at_whitespace(model_params_str, &model_params);
    split_at_whitespace(initc_theta_str, &initc_theta);
    split_at_whitespace(initc_tau_str, &initc_tau);
    split_at_whitespace(initc_omega_str, &initc_omega);

    vector<string> diagnostics_split_vec;
	// Split up diagnostics_str at whitespaces and store substrings in diagnostics
	if (vm.count("diagnostics"))
		boost::split(diagnostics_split_vec, diagnostics_str, boost::is_any_of(" \t"), boost::token_compress_on);
	
    for(auto str : diagnostics_split_vec)
    {
        try
        {
            diagnostics.push_back(diagnostic_map.at(str));
        }
        catch (const std::out_of_range & oor)
        {
            stringstream err_msg;
            err_msg << "Invalid diagnostic: " << str << endl;
            err_msg << "Valid strings are: " << endl;
            for(auto const &it : diagnostic_map)
                err_msg << it.first << endl;
            throw out_of_range(err_msg.str());
        }
    }

	// Split output_str at whitespaces and store substrings in output
    vector<string> output_split_vec;
	if (vm.count("output"))
    {
		boost::split(output_split_vec, output_str, boost::is_any_of(" \t"), boost::token_compress_on);
        // Use .at() for map access. Throws out of range errors, when 
        // keys are requested that are not initialized
        for(auto str: output_split_vec)
        {
            if (str.compare("None") == 0)
            {
                cout << "Not writing output\n";
                output.clear();
                break;
            }
            try
            {
                output.push_back(output_map.at(str));
            }
            catch (const std::out_of_range & oor )
            {
                stringstream err_msg;
                err_msg << "Invalid output:" << str << endl;
                err_msg << "Valid strings are: " << endl;
                for(auto const &it : output_map)
                    err_msg << it.first << endl;
                throw out_of_range(err_msg.str());
            }
        }
    }
    else
    {
        cout << "Not writing output\n";
    }
    // Parse string shift_modes, string is in the form shift_modes = (kx1 ky1), (kx2 ky2), (kx3 ky3) ... (kxn kyn)
    /*if ( vm.count("shift_modes") ){
        cout << "Reading tuples from " << shift_modes_str << "\n";	
        vector<string> split_vec;
        boost::split(split_vec, shift_modes_str, boost::is_any_of(",;"), boost::token_compress_on);
        mode mode_in;

        for ( vector<string>::iterator it = split_vec.begin(); it != split_vec.end(); it++) {
            istringstream tuple_in( (*it));
            tuple_in >> mode_in;
            cout << "Read mode: " << mode_in << "\n";
            mode_list.push_back(mode_in);
            do_randomize_modes = true;
        }
    }*/

    init_function_theta = map_safe_select(init_function_theta_str, init_func_map);
    init_function_omega = map_safe_select(init_function_omega_str, init_func_map);
    init_function_tau = map_safe_select(init_function_tau_str, init_func_map);

    theta_rhs = map_safe_select(theta_rhs_str, rhs_func_map);
    omega_rhs = map_safe_select(omega_rhs_str, rhs_func_map);
    tau_rhs = map_safe_select(tau_rhs_str, rhs_func_map);
}


int slab_config :: consistency() 
{
	// Test for empty output line
	//if (get_output().size() == 0) 
	//	throw config_error("No output variables specified. If no full output is desired, write output = None\n");
	// Test x domain boundary
	if (xleft >= xright) 
		throw config_error("Left domain boundary must be smaller than right domain boundary\n");
	// Test y domain boundary
	if (ylow >= yup)
		throw config_error("Lower domain boundary must be smaller than upper domain boundary\n");
	// Test initialization routine
	if (get_init_function_theta() == twodads::init_fun_t::init_NA) 
		throw config_error("Invalid initialization routine specified for theta in init file\n");
	if (get_init_function_omega() == twodads::init_fun_t::init_NA) 
		throw config_error("Invalid initialization routine specified for omega in init file\n");
	if (get_init_function_tau() == twodads::init_fun_t::init_NA) 
		throw config_error("Invalid initialization routine specified for tau in init file\n");
	// Test if logarithmic theta has correct RHS
	if (get_theta_rhs_type() == twodads::rhs_t::theta_rhs_log && log_theta == false) 
		throw config_error("Selecting rhs_theta_log as the RHS function must have log_theta = true\n");
	//if (get_theta_rhs_type() == twodads::rhs_t::theta_rhs_full && log_theta == false) 
	//	throw config_error("Selecting rhs_theta_full as the RHS function must have log_theta = true\n");
	if (get_theta_rhs_type() == twodads::rhs_t::theta_rhs_lin && log_theta == true) 
		throw config_error("Selecting rhs_theta as the RHS function must have log_theta = false\n");
    if (get_omega_rhs_type() == twodads::rhs_t::omega_rhs_sheath_nlin && (!get_log_theta() || !get_log_tau()))
        throw config_error("Selecting omega_rhs_sheath_nlin requires logarithmic density and temperature\n");
	// If radial probe diagnostics is specified we need the number of radial probes
	if (nprobes == 0 ) 
		throw config_error("Radial probe diagnostics requested but no number of radial probes specified\n");
            
	return (0);
}


// Helper function for reverse lookup in ..._map members.
// Used by print_config()
template <typename T>
string str_from_map(const T findval, map<string, T> my_map)
{
    typedef typename map<string, T>::value_type pair;
    auto it = find_if(my_map.begin(), my_map.end(), 
            [findval](const pair & p){ return p.second == findval;}
            );
    return (*it).first;
}


void slab_config :: print_config() const 
{
    cout << "01: runnr = " << get_runnr() << endl;
    cout << "02 xleft = " << get_xleft() << endl;
    cout << "03 xright = " << get_xright() << endl;
    cout << "04 ylow = " << get_ylow() << endl;
    cout << "05 yup = " << get_yup() << endl;
    cout << "06 Nx = " << get_nx() << endl;
    cout << "07 My = " << get_my() << endl;
    cout << "08 scheme = .... defaults to ss3, not parsed" << endl; 
    cout << "09 tlevs = ... defaults to 4, not parsed" << endl;
    cout << "10 deltat = " << get_deltat() << endl;
    cout << "11 tend = " << get_tend() << endl;
    cout << "12 tdiag = " << get_tdiag() << endl;
    cout << "13 tout = " << get_tout() << endl;
    cout << "14 log_theta = " << get_log_theta() << endl;
    cout << "15 log_tau = " << get_log_theta() << endl;
    cout << "15 do_particle_tracking = " << do_particle_tracking() << endl;
    cout << "16 nprobes = " << get_nprobe() << endl;
    cout << "17 nthreads = " << get_nthreads() << endl;

    // Use reverse map lookup technique to map twodads::... types to their keys in
    // maps defined in the top of this file:
    // http://stackoverflow.com/questions/5749073/reverse-map-lookup
    
    //typedef map<string, twodads::rhs_t> Map;
    //typedef Map::value_type pair;
    //twodads::rhs_t findval = get_theta_rhs_type();
    //auto it = find_if(rhs_func_map.begin(), rhs_func_map.end(), 
    //        [findval](const pair & p){ return p.second == findval;}
    //        );
    cout << "18 theta_rhs = " << str_from_map<twodads::rhs_t>(get_theta_rhs_type(), rhs_func_map) << endl;
    cout << "19 omega_rhs = " << str_from_map<twodads::rhs_t>(get_omega_rhs_type(), rhs_func_map) << endl;
    cout << "20 strmf_solver = .... defaults, not parsed" << endl;
    cout << "21 init_function_theta = " << str_from_map<twodads::init_fun_t>(get_init_function_theta(), init_func_map) << endl;
    cout << "   initial_conditions: ";
    for(auto it : get_initc_theta())
        cout << it << ", ";
    cout << endl;

    cout << "22 init_function_omega = " << str_from_map<twodads::init_fun_t>(get_init_function_omega(), init_func_map) << endl;
    cout << "   initial_conditions: ";
    for(auto it : get_initc_omega())
        cout << it << ", ";
    cout << endl;

    cout << "23 init_function_tau = " << str_from_map<twodads::init_fun_t>(get_init_function_tau(), init_func_map) << endl;
    cout << "   initial_conditions: ";
    for(auto it : get_initc_tau())
        cout << it << ", ";
    cout << endl;

    cout << "24 model_params = ";
    for(auto it : get_model_params())
        cout << it << ", ";
    cout << endl;

    cout << "25 output = ";
    for(auto it : get_output())
        cout << str_from_map<twodads::output_t>(it, output_map) << "\t";
    cout << endl;
    
    cout << "26 diagnostics = ";
    for(auto it : get_diagnostics())
        cout << str_from_map<twodads::diagnostic_t>(it, diagnostic_map) << "\t";
    cout << endl;
}

// End of file slab_config.cpp
