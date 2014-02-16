///
/// diagnostics.cpp
///

/// Contains diagnostic routines for turbulence simulation


#include "include/diagnostics.h"
#include <string>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <cmath>

using namespace std;


diagnostics :: diagnostics(slab_config const config) :
    slab_layout{config.get_nx(), config.get_my(), config.get_runnr(), config.get_xleft(),
                config.get_deltax(), config.get_ylow(), config.get_deltay(),
                config.get_deltat()},
    theta(config.get_nx(), config.get_my()), theta_x(config.get_nx(), config.get_my()), theta_y(config.get_nx(), config.get_my()),
    omega(config.get_nx(), config.get_my()), omega_x(config.get_nx(), config.get_my()), omega_y(config.get_nx(), config.get_my()),
    strmf(config.get_nx(), config.get_my()), strmf_x(config.get_nx(), config.get_my()), strmf_y(config.get_nx(), config.get_my()),
    theta_rhs(config.get_nx(), config.get_my()), omega_rhs(config.get_nx(), config.get_my()),
    time(0.0),
    old_com_x(0.0),
    old_com_y(0.0),
    old_wxx(0.0),
    old_wyy(0.0),
    t_probe(config.get_tdiag()),
    n_probes(config.get_nprobe()),
    use_log_theta(config.get_log_theta()), 
    theta_bg(config.get_initc(0)),
    init_flag_blobs(false),
    init_flag_kinetic(false),
    init_flag_thermal(false),
    init_flag_flow(false),
    init_flag_particles(false),
    init_flag_tprobe(false),
    init_flag_oprobe(false)
{

	#ifdef DEBUG
		cout << "Initializing diagnostic output\n";
		cout << "Diagnostic routines specified: ";
	# endif
	
    stringstream filename;
    string header;
	
    for(auto it: config.get_diagnostics())
    {
        filename.str(string());
        switch(it)
        {
            case twodads::diagnostic_t::diag_blobs:
                header = string("# 1: time \t2: theta_max\t3: theta_max_x\t4: theta_max_y\n# 5: strmf_max\t6: strmf_max_x\t7: strmf_max_y\n# 8: theta_int\t9: theta_int_x\t10: theta_int_y\n# 11: COMvx\t12: COMvy\t13: Wxx\t14: Wyy\t15: Dxx\t16: Dyy\n");
                init_diagnostic_output(string("blobs.dat"), header, init_flag_blobs);
                break;
		
            case twodads::diagnostic_t::diag_energy:
        
                header = string("# 1: time\t 2: C\t3: E\t4: W\t5: S\t6: V\t7: O\t8: T\t9: K\t 10: U\t11: CFL\n");
                init_diagnostic_output(string("kinetic.dat"), header, init_flag_kinetic);

                header = string("# 1: time\t 2: H\t3: P\t4: B\t5: F\t6: Q\t7: D\t8: A\t9: G\t 10: L\t11: R\n");
                init_diagnostic_output(string("thermal.dat"), header, init_flag_thermal);

                header = string("#1: time\t 2: D1\t3: D2\t4: D3\t5: D4\t6: D5\t7: D6\t8: D7\t9: D8\t10: D9\n");
                init_diagnostic_output(string("flows.dat"), header, init_flag_flow);
        
                break;
		
            case twodads::diagnostic_t::diag_probes:
        
                header = string("#1: time\t#2: n_tilde\t#3: n\t#4: phi\t#5: phi_tilde\t#6: Omega\t#7: Omega_tilde\t#8: v_x\t#9: v_y\t#10: v_y_tilde\t#11: Gamma_r\n");
                for ( unsigned int n = 0; n < n_probes; n++ ){
                    filename << "probe" << setw(2) << setfill('0') << n << ".dat";
                    init_diagnostic_output(filename.str(), header, init_flag_tprobe);
                    init_flag_tprobe = false;
                    filename.str(string());
                }
                break;
        /*
        else if ( (*it).compare(string("strmf_max")) == 0 ) 
        {
            cout << "Writing poloidal profile for phi at Xcom\n";
            dfun_ptr diag_strmf( boost::bind(&diagnostics::strmf_max, this, _1, _2) );
            d_funs_vec.push_back(diag_strmf);
        }
        */
            default:
                string err_msg("diagnostics::diagnostics: unknown diagnostic function requested.");
                throw name_error(err_msg);
                break;
		}
	}
	#ifdef DEBUG
		cout << "\n";
	#endif
}


diagnostics::~diagnostics()
{
}



/*
 ************** Logfile stuff **************************************************
 */


void diagnostics::write_logfile() {
	ofstream logfile;
	stringstream filename;
	filename << "log." << setw(3) << setfill('0') << slab_layout.runnr;
	logfile.open((filename.str()).data(), ios::trunc);
	
	logfile << "Slab type: spectral\n";
	logfile << "Resolution: " << slab_layout.My << "x" << slab_layout.Nx  << "\n";
	
	logfile.close();
}

/*
 *************************** Initialization for diagnostic routines ************
 */ 


void diagnostics :: init_diagnostic_output(string filename, string header, bool& init_flag){
    cout << "Initializing output file " << filename << "\n";
    if ( init_flag == false ) {
        ofstream output;
        output.exceptions(ofstream::badbit);
        output.open(filename.data(), ios::trunc);
        if ( !output ) {
            throw new std::exception;
        }
        output << header;
        output.close();
        init_flag = true;
    } 
}

	
void diagnostics :: update_arrays(slab_cuda& slab)
{
    theta.update(slab.theta);
    theta_x.update(slab.theta_x);
    theta_y.update(slab.theta_y);
    omega.update(slab.omega);
    omega_x.update(slab.omega_x);
    omega_y.update(slab.omega_y);
    strmf.update(slab.strmf);
    strmf_x.update(slab.strmf_x);
    strmf_y.update(slab.strmf_y);
    theta_rhs.update(slab.theta_rhs);
    omega_rhs.update(slab.omega_rhs);
}

/*
 *************************** Diagnostic routines *******************************
 */ 


void diagnostics::diag_rhs(const twodads::real_t time)
{
    cout << "diag_rhs:\n";
    cout << "theta_rhs.max = " << theta_rhs.get_max() << "\t";
    cout << "theta_rhs.min = " << theta_rhs.get_min() << "\t";
    cout << "theta_rhs.mean = " << theta_rhs.get_mean() << "\t";
    cout << "omega_rhs.max = " << omega_rhs.get_max() << "\t";
    cout << "omega_rhs.min = " << omega_rhs.get_min() << "\t";
    cout << "omega_rhs.mean = " << omega_rhs.get_mean() << "\n";
}


void diagnostics::write_diagnostics(const twodads::real_t time, const slab_config& config)
{
    for(auto d_name : config.get_diagnostics())
    {
        switch(d_name)
        {
            case twodads::diagnostic_t::diag_blobs:
                diag_blobs(time);
                break;

            case twodads::diagnostic_t::diag_energy:
                diag_energy(time);
                break;

            case twodads::diagnostic_t::diag_probes:
                diag_probes(time);
                break;
            case twodads::diagnostic_t::diag_rhs:
                diag_rhs(time);
        }
    }
}


void diagnostics::diag_blobs(const twodads::real_t time)
{  

	double theta_max = -1.0;
	double theta_max_x = 0.0;
	double theta_max_y = 0.0;
	double theta_int = 0.0;
	double theta_int_x = 0.0;
	double theta_int_y = 0.0;
	double strmf_max = -1.0;
	double strmf_max_x = 0.0;
	double strmf_max_y = 0.0;
	double wxx = 0.0, wyy = 0.0, dxx = 0.0, dyy = 0.0;
	double com_vx = 0.0, com_vy = 0.0;
	double x,y;
	
	ofstream output;
	// Copy theta array from slab, without ghost points
	//slab_array theta( myslab -> get_theta() );
    if ( use_log_theta ){
        for (int n = 0; n < int(slab_layout.Nx); n++){
            for (int m = 0; m < int(slab_layout.My); m++){
                theta(n,m) = exp( theta(n,m) ) - theta_bg;
            }
        }
    }
	
    theta_max_x = slab_layout.x_left;
    theta_max_y = slab_layout.y_lo;
    strmf_max_x = slab_layout.x_left;
    strmf_max_y = slab_layout.y_lo;

	// Compute maxima for theta and strmf, integrated particle density,
	// center of mass coordinates and relative dispersion tensor components
	for(int n = 0; n < int(slab_layout.Nx); n++){
		x = slab_layout.x_left + double(n) * slab_layout.delta_x;
		for(int m = 0; m < int(slab_layout.My); m++){
			y = slab_layout.y_lo + double(m) * slab_layout.delta_y;
			theta_int += theta(n,m);
			theta_int_x += theta(n,m) * x;
			theta_int_y += theta(n,m) * y;
 
			if ( fabs(theta(n,m)) >= theta_max) {
				theta_max = fabs(theta(n,m));
				theta_max_x = x;
				theta_max_y = y;
			}
			if ( fabs ( strmf(n,m) ) >= strmf_max ) {
				strmf_max = fabs( strmf(n,m) );
				strmf_max_x = x;
				strmf_max_y = y;
			}
		}
	}
	
	theta_int_x /= theta_int;
	theta_int_y /= theta_int;
	for(int n = 0; n < int(slab_layout.Nx); n++){
		x = slab_layout.x_left + double(n) * slab_layout.delta_x;
		for(int m = 0; m < int(slab_layout.My); m++){
			y = slab_layout.y_lo + double(m) * slab_layout.delta_y;
			wxx = theta(n,m) * ( x - theta_int_x ) * (x - theta_int_x);
			wyy = theta(n,m) * ( y - theta_int_y ) * (y - theta_int_y);
		}
	}
	wxx /= theta_int;
	wyy /= theta_int;
	
	// Compute center of mass velocities and diffusivities
	com_vx = (theta_int_x - old_com_x) / t_probe;
	com_vy = (theta_int_y - old_com_y) / t_probe;
	dxx = 0.5*(wxx - old_wxx) / t_probe;
	dyy = 0.5*(wyy - old_wyy) / t_probe;
	
	// Update center of mass coordinates and dispersion tensor elements
	old_com_x = theta_int_x;
	old_com_y = theta_int_y;
	old_wxx = wxx;
	old_wyy = wyy;

	// normalize the integral of the thermodynamic field
	theta_int *= (slab_layout.delta_x * slab_layout.delta_y);

	// Write to output file
	output.open("blobs.dat", ios::app);	
	if ( output.is_open() ){
		output << time << "\t";
		output << setw(12) << theta_max << "\t" << setw(12) << theta_max_x << "\t" << setw(12) << theta_max_y << "\t";
		output << setw(12) << strmf_max << "\t" << setw(12) << strmf_max_x << "\t" << setw(12) << strmf_max_y << "\t";
		output << setw(12) << theta_int << "\t" << setw(12) << theta_int_x << "\t" << setw(12) << theta_int_y << "\t";
		output << setw(12) << com_vx << "\t" << setw(12) << com_vy << "\t" << setw(12) << wxx << "\t" << setw(12) << wyy << "\t";
		output << setw(12) << dxx << "\t" << setw(12) << dyy << "\n";
		output.close();
	}
}


///@brief Compute energy integrals for various turbulence models
///@param time Time of output
///@detailed flows.dat: t D1 D2 D3 D4 D5 D6 D7 D8 D9 D10 D11 D12 D13
///@detailed $D_{1} = \frac{1}{2A} \int \mathrm{d}A n^2$
///@detailed $D_{2} = \frac{1}{2A} \int \mathrm{d}A \left\nabla_\perp \phi\right)^2$
///@detailed $D_{3} = \frac{1}{2A} \int \mathrm{d}A \Omega^2$
///@detailed $D_{4} = -\frac{1}{A} \int \mathrm{d}A \widetilde{n}\widetilde{\phi_y}$
///@detailed $D_{5} = \frac{1}{A} \int \mathrm{d}A \left(\bar{n} - \bar{\phi}\right)^2$
///@detailed $D_{6} = \frac{1}{A} \int \mathrm{d}A \left(\widetilde{n} - \widetilde{\phi}\right)^2$
///@detailed $D_{7} = \frac{1}{A} \int \mathrm{d}A \bar{\phi} \left( \bar{\phi} - \bar{n} \right)$
///@detailed $D_{8] = \frac{1}{A} \int \mathrm{d}A \widetilde{\phi} \left( \widetilde{\phi} - \widetilde{n}\right)$
///@detailed $D_{9} = \frac{1}{A} \int \mathrm{d}A 0$
///@detailed $D_{10} = \frac{1}{A} \int \mathrm{d}A \nabla_\perp^2 n$
///@detailed $D_{11} = \frac{1}{A} \int \mathrm{d}A \nabla_\perp \phi \nabla_\perp \Omega$
///@detailed $D_{12} = \frac{1}{A} \int \mathrm{d}A n_{xxx}^2 + n_{yyy}^2$
///@detailed $D_{13} = \frac{1}{A} \int \mathrm{d}A \phi_{xxx} \Omega_{xxx} + \phi_{yyy} \Omega_{yyy}$

void diagnostics::diag_energy(const twodads::real_t time)
{
	ofstream output;

    // Energy transfer integrals for interchange model
    double max1 = 0.0, max2 = 0.0, cfl = 0.0;

    double Ly = slab_layout.delta_y * double(slab_layout.My);
    double Lx = slab_layout.delta_x * double(slab_layout.Nx);


    // Multiply by 0.25 to get the integral:
    // get_mean returns the average, i.e. U.get_man() = sum_{n,m} u(n,m) / (Nx*My)
    // Domain is [-Lx:Lx] x [-Ly:Ly], thus
    // 1/A int_A u(x) dA = 1 / (4*Lx*Ly) sum_{n,m} u(x_n, y_m) delta_x delta_y
    // where delta_x = Lx/Nx and delta_y = Ly/My

    const double A{0.25 * (theta_x * (theta * strmf_y).bar()).get_mean()};
    const double B{0.125 * (theta.bar() * theta.bar()).get_mean()};
    const double C{0.25 * omega.get_mean()};
    const double D{0.125 * (theta_x.bar() * theta_x.bar()).get_mean()}; 
    const double E{0.5 * (omega.tilde() * omega.tilde()).get_mean()};
    const double F{-0.25 * (theta * strmf_y).get_mean()};
    const double H{0.25 * theta.get_mean()};
    const double G{0.0}; // ???
    const double K{0.125 * (strmf_x.tilde()*strmf_x.tilde() + strmf_y*strmf_y).get_mean()};
    const double O{0.25 * omega_x.bar().get_mean()};
    const double P{0.25 * (theta.tilde() * theta.tilde()).get_mean()};
    const double Q{0.25 * ((theta_x.tilde() * theta_x.tilde()) + (theta_y * theta_y)).get_mean()};
    const double S{-0.125 * (omega * strmf_y).get_mean()};
    const double T{0.25 * ((strmf_x * strmf_y) * omega.bar()).get_mean()};
    const double U{0.125 * (strmf_x.bar() * strmf_x.bar()).get_mean()};
    const double W{0.125 * (omega.bar() * omega.bar()).get_mean()};
    const double V{0.125 * (omega_x.tilde()*omega_x.tilde() + omega_y * omega_y).get_mean()};

    const double D1{0.125 * (theta * theta).get_mean()};
    const double D2{0.125 * ((strmf_x * strmf_x) + (strmf_y * strmf_y)).get_mean()};
    const double D3{0.125 * (omega * omega).get_mean()};
    const double D4{-0.25 * (theta.tilde() * strmf_y.tilde()).get_mean()};
    const double D5{0.25 * ((theta.bar() - strmf.bar()) *  (theta.bar() - strmf.bar())).get_mean()};
    const double D6{0.25 * ( (theta.tilde() - strmf.tilde()) *  (theta.tilde() - strmf.tilde()) ).get_mean()};
    const double D7{0.25 * (strmf.bar() * (strmf.bar() - theta.bar())).get_mean()};
    const double D8{0.25 * (strmf.tilde() * (strmf.tilde() - theta.tilde())).get_mean()};
    const double D9{0.0};
    const double D10{0.25 * (theta_x * theta_x + theta_y * theta_y).get_mean()};
    const double D11{0.25 * (strmf_x * omega_x + strmf_y * omega_y).get_mean()};
    const double D12{0.25 * ((theta_x.d2_dx2(Lx) * theta_x.d2_dx2(Lx)) + (theta_y.d2_dy2(Ly) * theta_y.d2_dy2(Ly))).get_mean()};
    const double D13{0.25 * ((strmf_x.d2_dx2(Lx) * omega_x.d2_dx2(Lx)) + (strmf_y.d2_dy2(Ly) * omega_y.d2_dy2(Ly))).get_mean()};

    // Compute the Reynolds stress
    //slab_array v(*theta);
    //for ( int n = 0; n < Nx; n++ ){
    //    for ( int m = 0; m < My; m++ ){
    //        v(n,m) = sqrt( (*strmf_x)(n,m) * (*strmf_x)(n,m) + (*strmf_y)(n,m) * (*strmf_y)(n,m) ); 
    //    }
    //}

    //slab_array rs2 = (strmf_y -> tilde()) * (strmf_x -> tilde());
    //fftw_array rs2_hat(1, Nx, My); 
    //// Derive w.r.t x
    //plan_fw = fftw_plan_dft_r2c_2d( Nx, My, rs2.get_array(), (fftw_complex*) rs2_hat.get_array(), FFTW_ESTIMATE);
    //plan_bw = fftw_plan_dft_c2r_2d( Nx, My, (fftw_complex*) rs2_hat.get_array(), rs2.get_array(), FFTW_ESTIMATE);
    //fftw_execute(plan_fw);
    //rs2_hat.d_dx( Lx );
    //fftw_execute(plan_bw);
    //rs2.normalize();

    //fftw_destroy_plan( plan_fw );
    //fftw_destroy_plan( plan_bw );

    //D9 = (v * rs2).get_mean();
    //D9 = -123.45;

	// Compute the CFL number
	max1 = max(strmf_x.get_max(), -1.0 * strmf_x.get_min());
	max2 = max(strmf_y.get_max(), -1.0 * strmf_y.get_min());
	cfl = max(max1, max2) * slab_layout.delta_t / slab_layout.delta_x;
	// Open file object
	output.open("kinetic.dat", ios::app);
	if ( output.is_open() ) {
		output << time << "\t";
		output << setw(12) << C << "\t" << setw(12) << E << "\t" << setw(12) << W << "\t" << setw(12) << S << "\t" << setw(12) << V << "\t";
        output << setw(12) << O << "\t" << setw(12) << T << "\t" << setw(12) << K << "\t" << setw(12) << U << "\t" << setw(12) << cfl << "\n";
		output.close();
	}

    output.open("thermal.dat", ios::app);
    if ( output.is_open() ) {
        output.width(12);
        output << time << "\t";
        output << setw(12) << H << "\t" << setw(12) << P << "\t" << setw(12) << B << "\t" << setw(12) << F << "\t" << setw(12) << Q << "\t";
        output << setw(12) << D << "\t" << setw(12) << A << "\t" << setw(12) << G << "\n";
        output.close();
    }

    output.open("flows.dat", ios::app);
    if ( output.is_open() ) {
        output << time << "\t";
        output << setw(12) << D1 << "\t" << setw(12) << D2 << "\t" << setw(12) << D3 << "\t" << setw(12) << D4 << "\t" << setw(12) << D5 << "\t";
        output << setw(12) << D6 << "\t" << setw(12) << D7 << "\t" << setw(12) << D8 << "\t" << setw(12) << D9 << "\t" << setw(12) << D10 << "\t";
        output << setw(12) << D11 << "\t" << setw(12) << D12 << "\t" << setw(12) << D13 << "\n";
        output.close();
    }

	
}

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


void diagnostics::diag_probes(const twodads::real_t time)
{
	ofstream output;
    stringstream filename;	
	const unsigned int delta_n = slab_layout.Nx / (1 + n_probes);
    const unsigned int delta_m = slab_layout.My / (1 + n_probes);
    unsigned npr2 = n_probes * n_probes;

    // Put probes in a square
    int np[npr2];
    int mp[npr2];

    for (uint i = 0; i < n_probes; i++ )
    {
        for(uint j = 0; j < n_probes; j++)
        {
            np[i * n_probes + j] = i * delta_n;
            mp[i * n_probes + j] = j * delta_m;
        }
    }


    // Write out this for n probes
    // #1: time    #2: n_tilde    #3: n    #4: phi    #5: phi_tilde    #6: Omega    #7: Omega_tilde    #8: v_x    #9: v_y    #10: v_y_tilde    #11: Gamma_r

    for (unsigned int n = 0; n < npr2 ; n++ ){
        filename << "probe" << setw(2) << setfill('0') << n << ".dat";
        output.open(filename.str().data(), ofstream::app );
        if ( output.is_open() ){
            output << time << "\t";                                                           // time
            output << setw(12) << (theta.tilde()  )( np[n], mp[n] ) << "\t";     // n_tilde
            output << setw(12) << (theta          )( np[n], mp[n] ) << "\t";     // n
            output << setw(12) << (strmf          )( np[n], mp[n] ) << "\t";     // phi
            output << setw(12) << (strmf.tilde()  )( np[n], mp[n] ) << "\t";     // phi_tilde
            output << setw(12) << (omega          )( np[n], mp[n] ) << "\t";     // Omega
            output << setw(12) << (omega.tilde()  )( np[n], mp[n] ) << "\t";     // Omega_tilde
            output << setw(12) << (strmf_y.tilde())( np[n], mp[n] ) << "\t";     // v_x
            output << setw(12) << (strmf_x        )( np[n], mp[n] ) << "\t";     // v_y
            output << setw(12) << (strmf_x.tilde())( np[n], mp[n] ) << "\t";     // v_y_tilde
            output << setw(12) << (theta*strmf_y  )( np[n], mp[n] ) << "\t";     // Gamma_r
            output << "\n";
            output.close();
        }
        filename.str(string());   // Delete current string member variable
	}	
}

// End of file diagnostics.cpp
