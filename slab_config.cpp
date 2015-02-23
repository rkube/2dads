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
    {"strmf", twodads::output_t::o_strmf},
    {"strmf_x", twodads::output_t::o_strmf_x},
    {"strmf_y", twodads::output_t::o_strmf_y},
    {"theta_rhs", twodads::output_t::o_theta_rhs},
    {"omega_rhs", twodads::output_t::o_omega_rhs}
};

const map<string, twodads::diagnostic_t> slab_config::diagnostic_map 
{
    {"blobs", twodads::diagnostic_t::diag_blobs},
    {"energy", twodads::diagnostic_t::diag_energy},
    {"probes", twodads::diagnostic_t::diag_probes},
	{"memory", twodads::diagnostic_t::diag_mem}
};

const map<string, twodads::init_fun_t> slab_config::init_func_map 
{
    {"theta_gaussian", twodads::init_fun_t::init_theta_gaussian},
    {"both_gaussian", twodads::init_fun_t::init_both_gaussian},
    {"theta_sine", twodads::init_fun_t::init_theta_sine},
    {"omega_sine", twodads::init_fun_t::init_omega_sine},
    {"both_sine", twodads::init_fun_t::init_both_sine},
    {"init_test", twodads::init_fun_t::init_test},
    {"theta_mode", twodads::init_fun_t::init_theta_mode},
    {"omega_mode", twodads::init_fun_t::init_omega_mode},
    {"both_mode", twodads::init_fun_t::init_both_mode},
    {"turbulent_bath", twodads::init_fun_t::init_turbulent_bath}
};

const map<string, twodads::rhs_t> slab_config::rhs_func_map 
{
    {"theta_rhs_ns", twodads::rhs_t::theta_rhs_ns},
    {"theta_rhs_lin", twodads::rhs_t::theta_rhs_lin},
    {"theta_rhs_log", twodads::rhs_t::theta_rhs_log},
    {"theta_rhs_hw", twodads::rhs_t::theta_rhs_hw},
    {"theta_rhs_hwmod", twodads::rhs_t::theta_rhs_hwmod},
    {"theta_rhs_null", twodads::rhs_t::theta_rhs_null},
    {"omega_rhs_ns", twodads::rhs_t::omega_rhs_ns},
    {"omega_rhs_hw", twodads::rhs_t::omega_rhs_hw},
    {"omega_rhs_hwmod", twodads::rhs_t::omega_rhs_hwmod},
    {"omega_rhs_hwzf", twodads::rhs_t::omega_rhs_hwzf},
    {"omega_rhs_ic", twodads::rhs_t::omega_rhs_ic},
    {"omega_rhs_null", twodads::rhs_t::omega_rhs_null}
};

// Parse input from input.ini
//config :: config( string filename, int foo ) :
slab_config :: slab_config(string basedir) :
	runnr(0), xleft(0.0), xright(0.0), ylow(0.0), yup(0.0),
	My(0), Nx(0), tlevs(0), deltat(0), tend(0), tdiag(0.0),
	tout(0.0), log_theta(false), do_dealiasing(false), particle_tracking(0), 
	nprobes(0), nthreads(0), chunksize(0)
{
	po::options_description cf_opts("Allowed options");
	po::variables_map vm;

    cout << "slab_config::slab_config(basedir) \n";
    cout << "\tbasedir = " << basedir << endl;

    fs::path dir(basedir);
    fs::path file("input.ini");
    fs::path full_path = dir / file;
    cout << "\tfs::path = " << full_path << endl;

	// Temporary strings that need to be parsed
	string model_params_str;
	string initc_str;
	string diagnostics_str;
	string output_str;
    string theta_rhs_str;
    string omega_rhs_str;
    //string init_function_str;
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
		("model_params", po::value<string> (&model_params_str) , "Model parameters")
		("theta_rhs", po::value<string> (&theta_rhs_str), "Right hand side function of theta")
		("omega_rhs", po::value<string> (&omega_rhs_str), "Right hand side function of omega")
		("diagnostics", po::value<string> (&diagnostics_str), "List of diagnostic functions to use")
		("output", po::value<string> (&output_str), "List of variables for output")
		("init_function", po::value<string> (&init_function_str), "Initialization function")
		("initial_conditions", po::value<string> (&initc_str) , "Initial conditions")
		("nthreads", po::value<int> (&nthreads), "Number of threads")
		("chunksize", po::value<int> (&chunksize), "Chunksize")
		("log_theta", po::value<bool> (&log_theta), "Use logarithmic particle density")
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
	
    if(vm.count("nthreads"))
        nthr = nthreads;    
	
	// Split model_params and initial_condition strings at whitespaces, 
	// convert substrings into floats and store in approproate vectors
	if (vm.count("model_params"))
		split_at_whitespace(model_params_str, &model_params);

	if (vm.count("initial_conditions"))
		split_at_whitespace(initc_str, &initc);

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
		boost::split(output_split_vec, output_str, boost::is_any_of(" \t"), boost::token_compress_on);

    for(auto str: output_split_vec)
    {
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

    // Assign theta_rhs, omega_rhs, and init_function by parsing the string 
    try 
    {
        init_function = init_func_map.at(init_function_str); 
    }
    catch (const std::out_of_range & oor)
    {
        stringstream err_msg;
        err_msg << "Invalid initial_function" << init_function_str << endl;
        err_msg << "Valid strings are: " << endl;
        for (auto const &it : init_func_map)
            err_msg << it.first << endl;
        throw out_of_range(err_msg.str());
    }


    // Set RHS type of theta equation
    try
    {
        theta_rhs = rhs_func_map.at(theta_rhs_str);
    }
    catch (const std::out_of_range & oor)
    {
        stringstream err_msg;
        err_msg << "Invalid RHS for theta: " << theta_rhs_str << endl;
        err_msg << "Valid strings are: " << endl;
        for (auto const &it : rhs_func_map)
            err_msg << it.first << endl;
        throw out_of_range(err_msg.str());
    }

    try
    {
        omega_rhs = rhs_func_map.at(omega_rhs_str);
    }
    catch (const std::out_of_range & oor)
    {
        stringstream err_msg;
        err_msg << "Invalid RHS for omega: " << omega_rhs_str << endl;
        err_msg << "Valid strings are: " << endl;
        for (auto const &it : rhs_func_map)
            err_msg << it.first << endl;
        throw out_of_range(err_msg.str());
    }
}


int slab_config :: consistency() 
{
	// Test for empty output line
	if (get_output().size() == 0) 
		throw config_error("No output variables specified. If no full output is desired, write output = None\n");
	// Test x domain boundary
	if (xleft >= xright) 
		throw config_error("Left domain boundary must be smaller than right domain boundary\n");
	// Test y domain boundary
	if (ylow >= yup)
		throw config_error("Lower domain boundary must be smaller than upper domain boundary\n");
	// Test initialization routine
	if (get_init_function() == twodads::init_fun_t::init_NA) 
		throw config_error("Invalid initialization routine specified in init file\n");
	// Test if logarithmic theta has correct RHS
	if (get_theta_rhs_type() == twodads::rhs_t::theta_rhs_log && log_theta == false) 
		throw config_error("Selecting rhs_theta_log as the RHS function must have log_theta = true\n");
	if (get_theta_rhs_type() == twodads::rhs_t::theta_rhs_lin && log_theta == true) 
		throw config_error("Selecting rhs_theta as the RHS function must have log_theta = false\n");
	// If radial probe diagnostics is specified we need the number of radial probes
	if (nprobes == 0 ) 
		throw config_error("Radial probe diagnostics requested but no number of radial probes specified\n");
    if(Nx % nthreads != 0)
        throw config_error("Nx must be a multiple of nthreads\n");
            
	return (0);
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
    cout << "15 do_particle_tracking = " << do_particle_tracking() << endl;
    cout << "16 nprobes = " << get_nprobe() << endl;

    // Use reverse map lookup technique to map twodads::... types to their keys in
    // maps defined in the top of this file:
    // http://stackoverflow.com/questions/5749073/reverse-map-lookup
    
    //namespace
    //{
        //typedef map<string, twodads::rhs_t> Map;
        //typedef Map::key_type key;
        //typedef Map::value_type pair;
        //typedef Map::mapped_type value;

        //twodads::rhs_t findval = get_theta_rhs_type();
        for (auto const &it : rhs_func_map)
        {
            cout << it.first << endl;
        }
//        auto it = find_if(rhs_func_map.begin(), rhs_func_map.end(), 
//                [findval](const pair & p){
//                    return p.second == findval;
//                });
//        if (it == rhs_func_map.end())
//           cerr << "theta_rhs not found" << endl; 
//        else 
//            cout << "17 theta_rhs = " << (*it).first << endl;
    //}
//    cout << "18 omega_rhs = "
//    cout << "19 strmf_solver = "
//    cout << "20 init_function = "
//    cout << "21 initial_conditions = "
//    cout << "22 model_params = "
//    cout << "23 output = "
// `  cout << "24 diagnostics = "
//    cout << "25 nthreads = "
}


// End of file slab_config.cpp
