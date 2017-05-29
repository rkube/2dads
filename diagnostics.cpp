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


diagnostic_t :: diagnostic_t(const slab_config_js& config) :
    slab_layout(config.get_geom()),
    theta_ptr{nullptr}, theta_x_ptr{nullptr}, theta_y_ptr{nullptr},
    omega_ptr{nullptr}, omega_x_ptr{nullptr}, omega_y_ptr{nullptr},
    tau_ptr{nullptr}, tau_x_ptr{nullptr}, tau_y_ptr{nullptr},
    strmf_ptr{nullptr}, strmf_x_ptr{nullptr}, strmf_y_ptr{nullptr}
    //t_probe(config.get_tdiag()),
    //n_probes(config.get_nprobe()),
    //use_log_theta(config.get_log_theta()), 
    //theta_bg(config.get_initc(0)),
    //init_flag_blobs(false)
    //init_flag_energy(false),
    //init_flag_particles(false),
    //init_flag_tprobe(false),
    //init_flag_oprobe(false)
{
    stringstream filename;
    string header;
	
    for(auto it: config.get_diagnostics())
    {
        filename.str(string());
        switch(it)
        {
            case twodads::diagnostic_t::diag_blobs:
                std::cout << "Initializing blobs" << std::endl;
                header = string("# 1: time \t2: theta_max\t3: theta_max_x\t4: theta_max_y\t# 5: strmf_max\t6: strmf_max_x\t7: strmf_max_y\t# 8: theta_int\t9: theta_int_x\t10: theta_int_y\t# 11: COMvx\t12: COMvy\n");
                init_diagnostic_output(std::string("blobs.dat"), header);
                break;
		
            //case twodads::diagnostic_t::diag_probes:
            //
            //    header = string("#1: time\t#2: n_tilde\t#3: n\t#4: phi\t#5: phi_tilde\t#6: Omega\t#7: Omega_tilde\t#8: v_x\t#9: v_y\t#10: v_y_tilde\t#11: Gamma_r\n");
            //    for (unsigned int n = 0; n < n_probes * n_probes; n++ ){
            //        filename << "probe" << setw(3) << setfill('0') << n << ".dat";
            //        init_diagnostic_output(filename.str(), header, init_flag_tprobe);
            //        init_flag_tprobe = false;
            //        filename.str(string());
            //    }
            //    init_flag_tprobe = true;
            //    break;
            //case twodads::diagnostic_t::diag_energy:
            //    cout << "diag_energy\n";
            //    header = string("#01: time\t#02: E\t#03: K\t#04: T\t#05: U\t#06: W\t#07: D1\t#08: D2\t#09: D3\t:#10: D4\t#11: D5\t#12:D6\t#13: D7\t#14: D8\t#15: D9\t#16: D10\t#17: D11\t#18: D12\t#19: D13\n");
            //    init_diagnostic_output(string("energy.dat"), header, init_flag_energy);
            //    break;
            //
            //case twodads::diagnostic_t::diag_mem:
            //	cout << "diag_memory\n";
            //	break;
        default:
                string err_msg("diagnostic_t::diagnostic_t: unknown diagnostic function requested.");
                throw name_error(err_msg);
                break;
		}
	}
}


const diagnostic_t::arr_real** diagnostic_t :: get_dptr_by_name(const twodads::field_t fname)
{
    const arr_real** res{nullptr};
    switch(fname)
    {
        case twodads::field_t::f_theta:
            res = &theta_ptr;
            break;
        case twodads::field_t::f_theta_x:
            res = &theta_x_ptr;
            break;
        case twodads::field_t::f_theta_y:
            res = &theta_y_ptr;
            break;
        case twodads::field_t::f_omega:
            res = &omega_ptr;
            break;
        case twodads::field_t::f_omega_x:
            res = &omega_x_ptr;
            break;
        case twodads::field_t::f_omega_y:
            res = &omega_y_ptr;
            break;
        case twodads::field_t::f_tau:
            res = &tau_ptr;
            break;
        case twodads::field_t::f_tau_x:
            res = &tau_x_ptr;
            break;
        case twodads::field_t::f_tau_y:
            res = &tau_y_ptr;
            break;
        case twodads::field_t::f_strmf:
            res = &strmf_ptr;
            break;
        case twodads::field_t::f_strmf_x:
            res = &strmf_x_ptr;
            break;
        case twodads::field_t::f_strmf_y:
            res = &strmf_y_ptr;
            break;
        default:
            std::cerr << "Requested pointer does not exist";
            break;
    }

    return(res);
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

/*
 *************************** Initialization for diagnostic routines ************
 */ 


void diagnostic_t :: init_diagnostic_output(string filename, string header)
{
    cout << "Initializing output file " << filename << "\n";
    stringstream err_msg;
    ofstream output;
    output.exceptions(ofstream::badbit);
    output.open(filename.data(), ios::trunc);
    if ( !output ) 
    {
        err_msg << "Error: " << filename << " already initialized." << std::endl;
        throw new diagnostics_error(err_msg.str());
    }
    output << header;
    output.close();
}


void diagnostic_t :: init_field_ptr(twodads::field_t fname, arr_real* src_ptr)
{
    //std :: cout << "init_field_ptr: before: " << get_dptr_by_name(fname) << ", src_ptr at " << src_ptr << std::endl;
    const arr_real** dptr{get_dptr_by_name(fname)};
    *dptr = src_ptr;
    //std :: cout << "init_field_ptr: after: " << get_dptr_by_name(fname) << std::endl;
    //std :: cout << "theta_ptr = " << theta_ptr << std::endl;
}


/*
 *************************** Diagnostic routines *******************************
 */ 


void diagnostic_t::write_diagnostics(const twodads::real_t time, const slab_config_js& config)
{
    for(auto d_name : config.get_diagnostics())
    {
        switch(d_name)
        {
            case twodads::diagnostic_t::diag_blobs:
                std::cout << "Diagnosing blobs" << std::endl;
                diag_blobs(time);
                break;

            case twodads::diagnostic_t::diag_energy:
                std::cout << "Diagnosing energy" << std::endl;
                //diag_energy(time);
                break;

            case twodads::diagnostic_t::diag_probes:
                std::cout << "Diagnosing probes" << std::endl;
                //diag_probes(time);
                break;
        }
    }
}


void diagnostic_t::diag_blobs(const twodads::real_t time)
{  
    std::cout << "diagnostics_t::diag_blobs" << std::endl;
	double theta_max_val{-1.0};
	double theta_max_x{-1.0};
	double theta_max_y{-1.0};
	double theta_int_val{-1.0};
	double theta_com_x{-1.0};
	double theta_com_y{-1.0};
	double strmf_max_val{-1.0};
	double strmf_max_x{0.0};
	double strmf_max_y{0.0};
	double com_vx{0.0}, com_vy{0.0};

    static double time_old{0.0};
    static double theta_com_x_old{0.0};
    static double theta_com_y_old{0.0};

    const arr_real** theta_ptr2{get_dptr_by_name(twodads::field_t::f_theta)};
    const arr_real** strmf_ptr2{get_dptr_by_name(twodads::field_t::f_strmf)};

    std::tuple<twodads::real_t, twodads::real_t, twodads::real_t> res_com = utility :: com(**theta_ptr2, 1);
    theta_int_val = std::get<0>(res_com);
    theta_com_x = std::get<1>(res_com);
    theta_com_y = std::get<2>(res_com);

    // Update radial blob velocity
    com_vx = (theta_com_x - theta_com_x_old) / (time - time_old);
    time_old = time;
    theta_com_x_old = theta_com_x;
    theta_com_y_old = theta_com_y;

    std::tuple<twodads::real_t, size_t, size_t> res_max = utility :: max_idx(**theta_ptr2, 1);
    theta_max_val = std::get<0>(res_max);
    theta_max_x = slab_layout.x_left + (twodads::real_t(std::get<1>(res_max)) - slab_layout.cellshift) * slab_layout.delta_x;
    theta_max_y = slab_layout.y_lo + (twodads::real_t(std::get<2>(res_max)) - slab_layout.cellshift) * slab_layout.delta_y;

    res_max = utility :: max_idx(**strmf_ptr2, 0);
    strmf_max_val = std::get<0>(res_max);
    strmf_max_x = slab_layout.x_left + (twodads::real_t(std::get<1>(res_max)) - slab_layout.cellshift) * slab_layout.delta_x;
    strmf_max_y = slab_layout.y_lo + (twodads::real_t(std::get<2>(res_max)) - slab_layout.cellshift) * slab_layout.delta_y;

	std::ofstream output;
	
	// Write to output file
	output.open("blobs.dat", std::ios::app);	
	if ( output.is_open() )
    {
		output << time << "\t";
		output << setw(12) << theta_max_val << "\t" << setw(12) << theta_max_x << "\t" << setw(12) << theta_max_y << "\t";
		output << setw(12) << strmf_max_val << "\t" << setw(12) << strmf_max_x << "\t" << setw(12) << strmf_max_y << "\t";
		output << setw(12) << theta_int_val << "\t" << setw(12) << theta_com_x << "\t" << setw(12) << theta_com_y << "\t";
		output << setw(12) << com_vx << "\t" << setw(12) << com_vy << "\n" << std::endl;
		output.close();
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

// End of file diagnostics.cpp