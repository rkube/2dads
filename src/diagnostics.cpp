/*
 *  diagnostics.cpp
 *  Contains diagnostic routines for turbulence simulation 
 */


#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <cmath>
#include <vector>
#include "diagnostics.h"
#include "utility.h"

using namespace std;

// Constructor

diag_com_t :: diag_com_t() : 
    C(std::make_tuple(0.0, 0.0, 0.0)), 
    C_old(std::make_tuple(0.0, 0.0, 0.0))
{
}


std::tuple<twodads::real_t, twodads::real_t> diag_com_t::get_vcom() const
{
    const twodads::real_t dt = {(std::get<2>(get_com()) - std::get<2>(get_com_old()))};
    const twodads::real_t Vx{(std::get<0>(get_com()) - std::get<0>(get_com_old())) / dt};
    const twodads::real_t Vy{(std::get<1>(get_com()) - std::get<1>(get_com_old())) / dt};

    return(std::make_tuple(Vx, Vy));
};


void diag_com_t::update_com(const cuda_array_bc_nogp<twodads::real_t, allocator_host>& vec, const size_t tidx, const twodads::real_t time)
{
        address_t<twodads::real_t>* addr{vec.get_address_ptr()};
        twodads::real_t* data_ptr = vec.get_tlev_ptr(tidx);

        twodads::real_t sum{0.0};
        twodads::real_t sum_x{0.0};
        twodads::real_t sum_y{0.0};
        twodads::real_t x{0.0};
        twodads::real_t y{0.0};
        twodads::real_t current_val{0.0};

        // Backup old center-of-mass coordinates
        C_old = C;
        for(size_t n = 0; n < vec.get_geom().get_nx(); n++)
        {
            x = vec.get_geom().x_left + (n + vec.get_geom().cellshift) * vec.get_geom().delta_x;
            for(size_t m = 0; m < vec.get_geom().get_my(); m++)
            {
                y = vec.get_geom().y_lo + (m + vec.get_geom().cellshift) * vec.get_geom().delta_y;
                current_val = addr -> get_elem(data_ptr, n, m);

                sum += current_val;
                sum_x += current_val * x;
                sum_y += current_val * y;
            }
        }

        sum_x /= sum;
        sum_y /= sum;

        // Update center-of-mass coordinates
        C = std::make_tuple(sum_x, sum_y, time);
}


// Maps header strings to each diagnostic type defined in 2dads_types.h
const std::map<twodads::diagnostic_t, std::string> diagnostic_t :: header_str_map 
{
    {twodads::diagnostic_t::diag_com_theta, std::string("#1: time\t# 2: int_x\t#3: int_y\t#4: COMvx\t#5: COMvy\n")},
    {twodads::diagnostic_t::diag_max_theta, std::string("#1: time\t# 2: max\t#3: max_x\t4: max_y\t#5: max_vx\t#6: max_vy\n")},
    {twodads::diagnostic_t::diag_com_tau, std::string("#1: time\t#2: int_x\t#3: int_y\t#4: COMvx\t#5: COMvy\n")},
    {twodads::diagnostic_t::diag_max_tau, std::string("#1: time\t#2: max\t#3: max_x\t#4: max_y\t#5: max_vx\t#6: max_vy\n")},
    {twodads::diagnostic_t::diag_probes, std::string("#1: time\t#2: theta\t#3: oemga\t#4: strmf\t#5: strmf_y\n")}
};

// Maps filename to each diagnostic type defined in 2dads_types.h
const std::map<twodads::diagnostic_t, std::string> diagnostic_t :: filename_str_map
{
    {twodads::diagnostic_t::diag_com_theta, std::string("com_theta.dat")},
    {twodads::diagnostic_t::diag_com_tau, std::string("com_tau.dat")},
    {twodads::diagnostic_t::diag_max_theta, std::string("max_theta.dat")},
    {twodads::diagnostic_t::diag_max_tau, std::string("max_tau.dat")}
};


diagnostic_t :: diagnostic_t(const slab_config_js& config) :
    slab_layout(config.get_geom()),
    // Map member functions to each diagnostic type defined in 2dads_types.h
    diag_func_map{
        std::map<twodads::diagnostic_t, dfun_ptr_t>
        {
            {twodads::diagnostic_t::diag_com_theta, &diagnostic_t::diag_com_theta},
            {twodads::diagnostic_t::diag_max_theta, &diagnostic_t::diag_max_theta},
            {twodads::diagnostic_t::diag_com_tau, &diagnostic_t::diag_com_tau},
            {twodads::diagnostic_t::diag_max_tau, &diagnostic_t::diag_max_tau},
            {twodads::diagnostic_t::diag_probes, &diagnostic_t::diag_probes}
        }
    },
    theta_ptr{nullptr}, theta_x_ptr{nullptr}, theta_y_ptr{nullptr},
    omega_ptr{nullptr}, omega_x_ptr{nullptr}, omega_y_ptr{nullptr},
    tau_ptr{nullptr}, tau_x_ptr{nullptr}, tau_y_ptr{nullptr},
    strmf_ptr{nullptr}, strmf_x_ptr{nullptr}, strmf_y_ptr{nullptr},
    // Map field names to data pointers to simulation data
    data_ptr_map{
        std::map<twodads::field_t, const arr_real**>
        {
            {twodads::field_t::f_theta, &theta_ptr},
            {twodads::field_t::f_theta_x, &theta_x_ptr},
            {twodads::field_t::f_theta_y, &theta_y_ptr},
            {twodads::field_t::f_omega, &omega_ptr},
            {twodads::field_t::f_omega_x, &omega_x_ptr},
            {twodads::field_t::f_omega_y, &omega_y_ptr},
            {twodads::field_t::f_tau, &tau_ptr},
            {twodads::field_t::f_tau_x, &tau_x_ptr},
            {twodads::field_t::f_tau_y, &tau_y_ptr},
            {twodads::field_t::f_strmf, &strmf_ptr},
            {twodads::field_t::f_strmf_x, &strmf_x_ptr},
            {twodads::field_t::f_strmf_y, &strmf_y_ptr},
        }
    }
{
    stringstream err_msg;
    ofstream out_file;
	
    // Initialize datafiles for each diagnostic routine in the config file
    for(auto it : config.get_diagnostics())
    {
        if(it == twodads::diagnostic_t::diag_probes)
            continue;
        out_file.exceptions(ofstream::badbit);
        try
        {
            out_file.open(filename_str_map.at(it), ios::trunc);
            if(!out_file)
            {
                err_msg << "Diagnostics: Could not open file" << filename_str_map.at(it) << "." << std::endl;
                throw new diagnostics_error(err_msg.str());
            }
        }
        catch (diagnostics_error err)
        {
            std :: cerr << err.what();
            std :: exit(1);
        } 
        out_file << header_str_map.at(it);
        out_file.close();
    }
}


//**********************************************************************************
//*                           Log file                                             *      
//**********************************************************************************


void diagnostic_t :: write_logfile() 
{
	ofstream logfile;
	stringstream filename;
	logfile.open(std::string("2dads.log"), ios::trunc);
    
    switch(slab_layout.grid)
    {
        case twodads::grid_t::cell_centered:
            logfile << "Grid: cell centered\n";
            break;
        case twodads::grid_t::vertex_centered:
            logfile << "Grid: vertex centered\n";
            break;
    }

	logfile << "Resolution: " << slab_layout.My << "x" << slab_layout.Nx  << "\n";
    logfile << "x_left: " << slab_layout.x_left << "\n";
    logfile << "delta_x" << slab_layout.delta_x << "\n";
    logfile << "y_lo: " << slab_layout.y_lo << "\n";
    logfile << "delta_y: " << slab_layout.delta_y << "\n";

	logfile.close();
}



void diagnostic_t :: init_field_ptr(twodads::field_t fname, arr_real* src_ptr)
{
    // Not really sure what is going on with the const_casting but it all works out.
    const arr_real** dptr{const_cast<const arr_real**>(data_ptr_map.at(fname))};
    *dptr = src_ptr;
}


//**********************************************************************************
//*                                 Diagnostic routines                            *
//**********************************************************************************


// Dispatch function
// Iterates over requested diagnostics from the config object
// and calls the associated member functions
void diagnostic_t::write_diagnostics(const twodads::real_t time, const slab_config_js& config)
{
    diagnostic_t::dfun_ptr_t dfun_ptr{nullptr};

    for(auto dname : config.get_diagnostics())
    {
        // Map-lookup of the diagnostic routine
        dfun_ptr = diag_func_map.at(dname);
        // Call diagnostics for current time
        (this->*dfun_ptr)(time);
    }

}


void diagnostic_t::diag_com(const twodads::field_t fieldname, diag_com_t& com, const std::string filename, const twodads::real_t time) 
{  
    std::ofstream out_file;
    const arr_real* const* arr_ptr2{data_ptr_map.at(fieldname)};

    //Update center-of-mass 
    com.update_com(**arr_ptr2, 1, time);

	// Write to out_file file
	out_file.open(filename, std::ios::app);	
	if ( out_file.is_open() )
    {
		out_file << time << "\t";
		out_file << setw(12) << std::get<0>(com.get_com()) << "\t";
        out_file << setw(12) << std::get<1>(com.get_com()) << "\t";
        out_file << setw(12) << std::get<0>(com.get_vcom()) << "\t";
		out_file << setw(12) << std::get<1>(com.get_vcom()) << std::endl;
		out_file.close();
	}
}


void diagnostic_t::diag_max(const twodads::field_t fname, const std::string filename, const twodads::real_t time) const
{  
    std::ofstream out_file;

    //static twodads::real_t time_old{0.0};
    //static twodads::real_t max_x_old{0.0};
    //static twodads::real_t max_y_old{0.0};

    //const arr_real* const * arr_ptr2{data_ptr_map.at(fname)};
    
    // Compute COM of density field
    //std::tuple<twodads::real_t, twodads::real_t, twodads::real_t> res_max = utility :: com(**arr_ptr2, 1);
    //const twodads::real_t max_val{std::get<0>(res_max)};
    //const twodads::real_t max_x{slab_layout.x_left + (std::get<1>(res_max) - slab_layout.cellshift) * slab_layout.delta_x};
    //const twodads::real_t max_y{slab_layout.y_lo + (std::get<2>(res_max) - slab_layout.cellshift) * slab_layout.delta_y};
    //// Update com velocities
    //const twodads::real_t max_vx{(max_x - max_x_old) / (time - time_old)};
    //const twodads::real_t max_vy{(max_y - max_y_old) / (time - time_old)};
    
    // Store for next call
    //time_old = time;
    //max_x_old = max_x;
    //max_y_old = max_y;

	// Write to out_file file
	out_file.open(filename, std::ios::app);	
	if ( out_file.is_open() )
    {
		out_file << time << "\n";
		//out_file << setw(12) << max_val << "\t" << setw(12) << max_x << "\t" << setw(12) << max_y << "\t";
		//out_file << setw(12) << max_vx << "\t" << setw(12) << max_vy << "\n" << std::endl;
		out_file.close();
	}
}


void diagnostic_t::diag_probes(const twodads::real_t time)
{
    std::ofstream out_file;;
    std::stringstream filename;	
    constexpr size_t n_probes{8};
    //const size_t delta_n{int(slab_layout.Nx) / n_probes};

    // probe_type = [field type, probe values, time index]
    using probe_type = std::tuple<twodads::field_t, size_t, std::vector<twodads::real_t>>;

    // Vector of probe_types. No probe values are not initialized
    std::vector<probe_type> probe_data{
        {twodads::field_t::f_theta, 1, std::vector<twodads::real_t>{}},
        {twodads::field_t::f_omega, 1, std::vector<twodads::real_t>{}},
        {twodads::field_t::f_strmf, 0, std::vector<twodads::real_t>{}},
        {twodads::field_t::f_strmf_y, 0, std::vector<twodads::real_t>{}},
    };

    //std::array<twodads::real_t, 4> probe_vals{{0.0, 0.0, 0.0, 0.0}};
    size_t tlev{0};
    address_t<twodads::real_t>* address_ptr{nullptr};
    
    // Spacing of probes
    size_t delta_n{0};
    size_t probe_my{0};

    // Make the iterator a reference since we are updating the vector in probe_data :)
    for(auto& it : probe_data)
    {
        // Get pointer on data array and address object
        const arr_real* const* arr_ptr2{data_ptr_map.at(std::get<0>(it))};
        address_ptr = (*arr_ptr2) -> get_address_ptr();

        // Get the time index
        tlev = std::get<1>(it);

        // Update probe spacing
        delta_n = (*arr_ptr2) -> get_nx() / n_probes;
        probe_my = (*arr_ptr2) -> get_my() / 2;

        // Iterate over the probe location (in index space) and put values in the respective probe vector
        for(size_t n_pr = 0; n_pr < n_probes; n_pr++)
        {
            std::get<2>(it).push_back((*address_ptr).get_elem((*arr_ptr2) -> get_tlev_ptr(tlev), n_pr * delta_n, probe_my));
        }

        //std::cout << "In first loop" << std::endl;
        for(auto vec_it : std::get<2>(it))
        {
            std::cout << vec_it << "\t";
        }
        std::cout << std::endl;
    }

    /*
    std::cout << "In second loop" << std::endl;
    for(auto it : probe_data)
    {
        for(auto vec_it : std::get<2>(it))
        {
            std::cout << vec_it << "\t";
        }
        std::cout << std::endl;
    }
    */

    std::cout << "Writing" << std::endl;
    // Write out this for n probes
    // #1: time    #2: n    #3: omega   #4: phi #5: phi_y
    for (size_t npr = 0; npr < n_probes; npr++ )
    {
        filename << "probe" << setw(3) << setfill('0') << npr << ".dat";
        out_file.open(filename.str().data(), ofstream::app );
        if ( out_file.is_open() )
        {
            out_file << time << "\t";                                  // time
            for(auto it : probe_data)
            {
                out_file << setw(12) << std::get<2>(it)[npr];
            }
            //out_file << setw(12) << theta        (np, mp) << "\t";     // n
            out_file << std::endl;
            out_file.close();
        }
        filename.str(std::string());   // Delete current string member variable
    }	
}




// void diagnostics::diag_energy(const twodads::real_t time)
// {
//     ofstream out_file;
//     const double Lx = slab_layout.delta_y * double(slab_layout.My);
//     const double Ly = slab_layout.delta_x * double(slab_layout.Nx);
//     const double invdA = 1. / (Lx * Ly);

//     diag_array<double> strmf_tilde(move(strmf.tilde()));
//     diag_array<double> strmf_bar(move(strmf.bar()));
//     diag_array<double> strmf_x_tilde(move(strmf_x.tilde()));
//     diag_array<double> omega_tilde(move(omega.tilde()));
//     diag_array<double> omega_bar(move(omega.bar()));
//     diag_array<double> theta_tilde(move(theta.tilde()));
//     diag_array<double> theta_bar(move(theta.bar())); 

//     const double E{0.5 * invdA * ((theta_tilde * theta_tilde) + (strmf_x * strmf_x) + (strmf_y * strmf_y)).get_mean()};
//     const double K{((strmf_x_tilde * strmf_x_tilde) + (strmf_y * strmf_y)).get_mean()};
//     const double T{((strmf_x_tilde * strmf_y.tilde()) * omega_bar).get_mean()};
//     const double U{(strmf_x.bar() * strmf_x.bar()).get_mean()};
//     const double W{(omega_bar * omega_bar).get_mean()};

//     const double D1{0.5 * (theta * theta).get_mean()};
//     const double D2{0.5 * ((strmf_x * strmf_x) + (strmf_y * strmf_y)).get_mean()};
//     const double D3{0.5 * (omega * omega).get_mean()};
//     const double D4{-1.0 * (theta_tilde * strmf_y.tilde()).get_mean()};
//     const double D5{ ((theta_bar - strmf_bar) *  (theta_bar - strmf_bar)).get_mean()};
//     const double D6{ ( (theta_tilde - strmf_tilde) * (theta_tilde - strmf_tilde)).get_mean()};
//     const double D7{ (strmf_bar * (strmf_bar - theta_bar)).get_mean()};
//     const double D8{ (strmf_tilde * (strmf_tilde - theta_tilde)).get_mean()};
//     const double D9{0.0};
//     const double D10{ (theta_x * theta_x + theta_y * theta_y).get_mean()};
//     const double D11{ (strmf_x * omega_x + strmf_y * omega_y).get_mean()};
//     const double D12{ ((theta_x.d2_dx2(Lx) * theta_x.d2_dx2(Lx)) + (theta_y.d2_dy2(Ly) * theta_y.d2_dy2(Ly))).get_mean()};
//     const double D13{ ((strmf_x.d2_dx2(Lx) * omega_x.d2_dx2(Lx)) + (strmf_y.d2_dy2(Ly) * omega_y.d2_dy2(Ly))).get_mean()};

// 	out_file.open("energy.dat", ios::app);
// 	if (out_file.is_open()) 
//     {
// 		out_file << time << "\t";
// 		out_file << setw(12) << E << "\t" << setw(12) << K << "\t" << setw(12) << T << "\t" << setw(12) << U << "\t" << setw(12) << W << "\t";
// 		out_file << setw(12) << D1 << "\t" << setw(12) << D2 << "\t" << setw(12) << D3 << "\t" << setw(12) << D4 << "\t" << setw(12) << D5 << "\t";
// 		out_file << setw(12) << D6 << "\t" << setw(12) << D7 << "\t" << setw(12) << D8 << "\t" << setw(12) << D9 << "\t" << setw(12) << D10 << "\t";
// 		out_file << setw(12) << D11 << "\t" << setw(12) << D12 << "\t" << setw(12) << D13 << "\n"; 
// 		out_file.close();
// 	}
// }



/// Write poloidal profile of phi at X_com of a blob
//void diagnostics::strmf_max(slab const * myslab, double const t) {
//
//    double x;
//
//    const int Nx = myslab -> get_nx();
//    const int My = myslab -> get_my();
//    
//    double theta_int = 0.0;
//    double theta_int_x = 0.0;
//	unsigned int idx_xcom;
//    const double delta_x = myslab -> slab_config -> get_deltax();
//    ofstream out_file;
//
//    slab_array theta( myslab -> get_theta() );
//
//    // Compute COM position and index where blob is at X_com 
//    for ( int n = 0; n < Nx; n++ ){
//        x = myslab -> slab_config -> get_xleft() + double(n) * delta_x;
//        for ( int m = 0; m < My; m++ ){
//            //y = myslab -> slab_config -> get_ylow() + double(m) * delta_y;
//    
//            theta_int += theta(n,m);
//            theta_int_x += theta(n,m) * x;
//       }
//    }
//    theta_int_x /= theta_int;
//
//    // Compute index where X=X_com
//    if ( (theta_int_x - myslab -> slab_config -> get_xleft()) / delta_x < 0 ){
//        cerr << "diagnostics::strmf_max : theta_int_x = " << theta_int_x;
//        cerr << "\tx_left = " <<   myslab -> slab_config -> get_xleft() << "\tdelta_x = " << delta_x << "\n";
//    }
//    idx_xcom = (int) (floor((theta_int_x - myslab -> slab_config -> get_xleft()) / delta_x));
//    out_file.open("strmf_max.dat", ios::app);
//    if ( out_file.is_open() ){
//        out_file << t << " " << theta_int_x << " " << idx_xcom << " ";
//        for ( int m = 0; m < My; m++ ){
//            if ( int(idx_xcom) < myslab -> get_nx() ){
//                out_file << setw(12) << myslab -> get_strmf(idx_xcom, m) << " ";
//            } else {
//                out_file << setw(12) <<  -999.9 << " ";
//           }
//        }
//        out_file << "\n";
//    }
//
//
//}



// Write particle positions
// Correct, my_slab is not used in this function. It is in the parameter list to satisft the interface 
// for diagnostic functions
//void diagnostics::particles_full(tracker const* my_tracker, slab const * my_slab, double const time){
//	ofstream out_file;
//    out_file.open("particles.dat", ios::app);
//    if ( out_file.is_open() ) {
//        out_file << time << "\t";
//        for ( vector<coordinate>::const_iterator it = my_tracker ->get_positions().begin(); it != my_tracker ->get_positions().end(); it++) {
//            out_file << it ->get_x() << "," << it -> get_y() << "\t";    
//        }
//        out_file << "\n";
//        out_file.close();
//    }
//}



diagnostic_t :: ~diagnostic_t() 
{
    theta_ptr = nullptr;
    theta_x_ptr = nullptr;
    theta_y_ptr = nullptr;
    omega_ptr = nullptr;
    omega_x_ptr = nullptr;
    omega_y_ptr = nullptr;
    tau_ptr = nullptr;
    tau_x_ptr = nullptr;
    tau_y_ptr = nullptr;
    strmf_ptr = nullptr;
    strmf_x_ptr = nullptr;
    strmf_y_ptr = nullptr;
}

// End of file diagnostics.cpp
