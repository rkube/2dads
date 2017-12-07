/*
 *  diagnostics.cpp
 *  Contains diagnostic routines for turbulence simulation 
 */


#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <cmath>
#include "diagnostics.h"
#include "utility.h"

using namespace std;

// Maps header strings to each diagnostic type defined in 2dads_types.h
const std::map<twodads::diagnostic_t, std::string> diagnostic_t :: header_str_map {
    {twodads::diagnostic_t::diag_com_theta, std::string("# 1: time \t2: theta_int\t3: theta_int_x\t4: theta_int_y\t# 5: COMvx\t6: COMvy\n")},
    {twodads::diagnostic_t::diag_com_tau, std::string("# 1: time \t2: tau_int\t3: tau_int_x\t4: tau_int_y\t# 5: COMvx\t6: COMvy\n")},
    {twodads::diagnostic_t::diag_max_theta, std::string("# 1: time \t2: theta_max\t3: max_x\t4: max_y\t# 5: max_vx\t6: max_vy\n")},
    {twodads::diagnostic_t::diag_max_tau, std::string("# 1: time \t2: tau_max\t3: max_x\t4: max_y\t# 5: max_vx\t6: max_vy\n")}
};

// Maps filename to each diagnostic type defined in 2dads_types.h
const std::map<twodads::diagnostic_t, std::string> diagnostic_t :: filename_str_map{
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
            {twodads::diagnostic_t::diag_max_tau, &diagnostic_t::diag_max_tau}
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
    //t_probe(config.get_tdiag()),
    //n_probes(config.get_nprobe()),
{
    stringstream err_msg;
    ofstream out_file;
	
    // Initialize datafiles for each diagnostic routine in the config file
    for(auto it : config.get_diagnostics())
    {
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


/*
 ************** Logfile **************************************************
 */


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


void diagnostic_t::diag_com(const twodads::field_t fieldname, const std::string filename, const twodads::real_t time) const
{  
    std::ofstream out_file;

    static twodads::real_t time_old{0.0};
    static twodads::real_t com_x_old{0.0};
    static twodads::real_t com_y_old{0.0};

    const arr_real* const* arr_ptr2{data_ptr_map.at(fieldname)};
    
    // Compute COM of density field
    std::tuple<twodads::real_t, twodads::real_t, twodads::real_t> res_com = utility :: com(**arr_ptr2, 1);
    const twodads::real_t int_val{std::get<0>(res_com)};
    const twodads::real_t com_x{std::get<1>(res_com)};
    const twodads::real_t com_y{std::get<2>(res_com)};

    // Update com velocities
    const twodads::real_t com_vx{(com_x - com_x_old) / (time - time_old)};
    const twodads::real_t com_vy{(com_y - com_y_old) / (time - time_old)};
    time_old = time;
    com_x_old = com_x;
    com_y_old = com_y;

	// Write to output file
	out_file.open(filename, std::ios::app);	
	if ( out_file.is_open() )
    {
		out_file << time << "\t";
		out_file << setw(12) << int_val << "\t" << setw(12) << com_x << "\t" << setw(12) << com_y << "\t";
		out_file << setw(12) << com_vx << "\t" << setw(12) << com_vy << "\n" << std::endl;
		out_file.close();
	}
}


void diagnostic_t::diag_max(const twodads::field_t fname, const std::string filename, const twodads::real_t time) const
{  
    std::ofstream out_file;

    static twodads::real_t time_old{0.0};
    static twodads::real_t max_x_old{0.0};
    static twodads::real_t max_y_old{0.0};

    const arr_real* const * arr_ptr2{data_ptr_map.at(fname)};
    
    // Compute COM of density field
    std::tuple<twodads::real_t, twodads::real_t, twodads::real_t> res_max = utility :: com(**arr_ptr2, 1);
    const twodads::real_t max_val{std::get<0>(res_max)};
    const twodads::real_t max_x{slab_layout.x_left + (std::get<1>(res_max) - slab_layout.cellshift) * slab_layout.delta_x};
    const twodads::real_t max_y{slab_layout.y_lo + (std::get<2>(res_max) - slab_layout.cellshift) * slab_layout.delta_y};
    // Update com velocities
    const twodads::real_t max_vx{(max_x - max_x_old) / (time - time_old)};
    const twodads::real_t max_vy{(max_y - max_y_old) / (time - time_old)};
    
    // Store for next call
    time_old = time;
    max_x_old = max_x;
    max_y_old = max_y;

	// Write to output file
	out_file.open(filename, std::ios::app);	
	if ( out_file.is_open() )
    {
		out_file << time << "\t";
		out_file << setw(12) << max_val << "\t" << setw(12) << max_x << "\t" << setw(12) << max_y << "\t";
		out_file << setw(12) << max_vx << "\t" << setw(12) << max_vy << "\n" << std::endl;
		out_file.close();
	}
}
// void diagnostics::diag_energy(const twodads::real_t time)
// {
//     ofstream output;
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

// 	output.open("energy.dat", ios::app);
// 	if (output.is_open()) 
//     {
// 		output << time << "\t";
// 		output << setw(12) << E << "\t" << setw(12) << K << "\t" << setw(12) << T << "\t" << setw(12) << U << "\t" << setw(12) << W << "\t";
// 		output << setw(12) << D1 << "\t" << setw(12) << D2 << "\t" << setw(12) << D3 << "\t" << setw(12) << D4 << "\t" << setw(12) << D5 << "\t";
// 		output << setw(12) << D6 << "\t" << setw(12) << D7 << "\t" << setw(12) << D8 << "\t" << setw(12) << D9 << "\t" << setw(12) << D10 << "\t";
// 		output << setw(12) << D11 << "\t" << setw(12) << D12 << "\t" << setw(12) << D13 << "\n"; 
// 		output.close();
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
//    ofstream output;
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
//    output.open("strmf_max.dat", ios::app);
//    if ( output.is_open() ){
//        output << t << " " << theta_int_x << " " << idx_xcom << " ";
//        for ( int m = 0; m < My; m++ ){
//            if ( int(idx_xcom) < myslab -> get_nx() ){
//                output << setw(12) << myslab -> get_strmf(idx_xcom, m) << " ";
//            } else {
//                output << setw(12) <<  -999.9 << " ";
//           }
//        }
//        output << "\n";
//    }
//
//
//}



// Write particle positions
// Correct, my_slab is not used in this function. It is in the parameter list to satisft the interface 
// for diagnostic functions
//void diagnostics::particles_full(tracker const* my_tracker, slab const * my_slab, double const time){
//	ofstream output;
//    output.open("particles.dat", ios::app);
//    if ( output.is_open() ) {
//        output << time << "\t";
//        for ( vector<coordinate>::const_iterator it = my_tracker ->get_positions().begin(); it != my_tracker ->get_positions().end(); it++) {
//            output << it ->get_x() << "," << it -> get_y() << "\t";    
//        }
//        output << "\n";
//        output.close();
//    }
//}

// void diagnostics::diag_probes(const twodads::real_t time)
// {
// 	ofstream output;
//     stringstream filename;	
// 	const int delta_n {int(slab_layout.Nx) / int(n_probes)};
//     const int delta_m {int(slab_layout.My) / int(n_probes)};
//     int np;
//     int mp;

//     diag_array<double> theta_tilde(theta.tilde());
//     diag_array<double> strmf_tilde(strmf.tilde());
//     diag_array<double> omega_tilde(omega.tilde());
//     diag_array<double> strmf_y_tilde(strmf_y.tilde());
//     diag_array<double> strmf_x_tilde(strmf_x.tilde());

//     // Write out this for n probes
//     // #1: time    #2: n_tilde    #3: n    #4: phi    #5: phi_tilde    #6: Omega    #7: Omega_tilde    #8: v_x    #9: v_y    #10: v_y_tilde    #11: Gamma_r
//     for (int n = 0; n < (int) n_probes; n++ )
//     {
//         np = n * delta_n;
//         for(int m = 0; m < (int) n_probes; m++)
//         {
//             mp = m * delta_m;
//             filename << "probe" << setw(3) << setfill('0') << n * n_probes + m << ".dat";
//             output.open(filename.str().data(), ofstream::app );
//             if ( output.is_open() ){
//                 output << time << "\t";                                  // time
//                 output << setw(12) << theta_tilde  (np, mp) << "\t";     // n_tilde
//                 output << setw(12) << theta        (np, mp) << "\t";     // n
//                 output << setw(12) << strmf        (np, mp) << "\t";     // phi
//                 output << setw(12) << strmf_tilde  (np, mp) << "\t";     // phi_tilde
//                 output << setw(12) << omega        (np, mp) << "\t";     // Omega
//                 output << setw(12) << omega_tilde  (np, mp) << "\t";     // Omega_tilde
//                 output << setw(12) << strmf_y_tilde(np, mp) << "\t";     // -v_x
//                 output << setw(12) << strmf_x      (np, mp) << "\t";     // v_y
//                 output << setw(12) << strmf_x_tilde(np, mp) << "\t";     // v_y_tilde
//                 output << "\n";
//                 output.close();
//             }
//             filename.str(string());   // Delete current string member variable
//         }	
//     }
// }

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
