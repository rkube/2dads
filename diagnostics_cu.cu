/*
 *
 * Implmentation of diagnostics class using cuda_darray
 *
 */


#include <fstream>
#include "diagnostics_cu.h"


const map <twodads::diagnostic_t, std::string> dfile_fname{
	{twodads::diagnostic_t::diag_blobs, "cu_blobs.dat"},
	{twodads::diagnostic_t::diag_probes, "cu_probes.dat"},
	{twodads::diagnostic_t::diag_energy, "cu_energy.dat"},
	{twodads::diagnostic_t::diag_mem, "cu_memory.day"}
};

const map <twodads::diagnostic_t, std::string> dfile_header{
	{twodads::diagnostic_t::diag_blobs,  "#01: time\t\t#02: theta_max\t#03: theta_max_x\t#04: theta_max_y\n#05: strmf_max\t#06: strmf_max_x\t#07: strmf_max_y\n#08: theta_int\t#09: theta_int_x\t#10: theta_int_y\n#11: COMvx\t#12: COMvy\t#13: Wxx\t#14: Wyy\t#15: Dxx\t#16: Dyy\n"},
	{twodads::diagnostic_t::diag_energy, "#01: time\t#02: E\t#03: K\t#04: T\t#05: U\t#06: W\t#07: D1\t#08: D2\t#09: D3\t:#10: D4\t#11: D5\t#12:D6\t#13: D7\t#14: D8\t#15: D9\t#16: D10\t#17: D11\t#18: D12\t#19: D13\n"},
	{twodads::diagnostic_t::diag_probes, "#01: time\t#02: n_tilde\t#03: n\t#04: phi\t#05: phi_tilde\t#06: Omega\t#07: Omega_tilde\t#08: v_x\t#09: v_y\t#10: v_y_tilde\t#11: Gamma_r\n"},
	{twodads::diagnostic_t::diag_mem, "none"}
};

//map <twodads::diagnostic_t, diagnostics_cu::diag_func_ptr> diagnostics_cu :: get_diag_func_by_name = "diagnostics_cu :: create_diag_func_map();


diagnostics_cu :: diagnostics_cu(const slab_config& config) :
    slab_layout{config.get_nx(), config.get_my(), config.get_runnr(), config.get_xleft(),
                config.get_deltax(), config.get_ylow(), config.get_deltay(),
                config.get_deltat()},
    theta(config.get_my(), config.get_nx()),
    theta_x(config.get_my(), config.get_nx()),
    theta_y(config.get_my(), config.get_nx()),
    omega(config.get_my(), config.get_nx()),
    omega_x(config.get_my(), config.get_nx()),
    omega_y(config.get_my(), config.get_nx()),
    strmf(config.get_my(), config.get_nx()),
    strmf_x(config.get_my(), config.get_nx()),
    strmf_y(config.get_my(), config.get_nx()),
    theta_rhs(config.get_my(), config.get_nx()),
    omega_rhs(config.get_my(), config.get_nx()),
    x_vec(new twodads::real_t[config.get_nx()]),
    y_vec(new twodads::real_t[config.get_my()]),
    x_arr(config.get_my(), config.get_nx()),
    y_arr(config.get_my(), config.get_nx()),
    time(0.0),
    dt_diag(config.get_tdiag()),
    n_probes(config.get_nprobe()),
    use_log_theta(config.get_log_theta()),
    theta_bg(config.get_initc(0)),
    get_darr_by_name{{twodads::field_t::f_theta,     &theta},
                	 {twodads::field_t::f_theta_x,   &theta_x},
                	 {twodads::field_t::f_theta_y,   &theta_y},
                	 {twodads::field_t::f_omega,     &omega},
                	 {twodads::field_t::f_omega_x,   &omega_x},
                	 {twodads::field_t::f_omega_y,   &omega_y},
                	 {twodads::field_t::f_strmf,     &strmf},
                	 {twodads::field_t::f_strmf_x,   &strmf_x},
                	 {twodads::field_t::f_strmf_y,   &strmf_y},
                	 {twodads::field_t::f_theta_rhs, &theta_rhs},
                	 {twodads::field_t::f_omega_rhs, &omega_rhs}
                },
    get_dfunc_by_name{{twodads::diagnostic_t::diag_energy,  &diagnostics_cu::diag_energy},
                      {twodads::diagnostic_t::diag_blobs,  &diagnostics_cu::diag_blobs},
                      {twodads::diagnostic_t::diag_probes,  &diagnostics_cu::diag_probes},
                      {twodads::diagnostic_t::diag_mem,  &diagnostics_cu::diag_mem}}
{
    stringstream filename;
    string header;

    // Create x_arr and y_arr
	// Create cuda arrays with x-values along the columns
	for(unsigned int n = 0; n < slab_layout.Nx; n++)
		x_vec[n] = slab_layout.x_left + slab_layout.delta_x * (cuda::real_t) n;
	x_arr.upcast_col(x_vec, slab_layout.Nx);

	for(unsigned int m = 0; m < slab_layout.My; m++)
		y_vec[m] = slab_layout.y_lo + slab_layout.delta_y * (cuda::real_t) m;
	y_arr.upcast_row(y_vec, slab_layout.My);

    for(auto it: config.get_diagnostics())
    {
        filename.str(string());
        init_diagnostic_output(dfile_fname.at(it), dfile_header.at(it));
    }
}


diagnostics_cu :: ~diagnostics_cu()
{
	if(x_vec != nullptr)
		delete [] x_vec;
	if(y_vec != nullptr)
		delete [] y_vec;
}


void diagnostics_cu :: write_diagnostics(const twodads::real_t time, const slab_config& config)
{
	diag_func_ptr dfun_ptr{nullptr};
    for(auto diag_name : config.get_diagnostics())
    {
    	dfun_ptr = get_dfunc_by_name.at(diag_name);
    	// Syntax for calling pointer to member functions:
    	// http://www.parashift.com/c++-faq/macro-for-ptr-to-memfn.html
    	((*this).*(dfun_ptr))(time);
    }
}


void diagnostics_cu::write_logfile()
{
	ofstream logfile;
	stringstream filename;
	filename << "cu_log." << setw(3) << setfill('0') << slab_layout.runnr;
	logfile.open((filename.str()).data(), ios::trunc);

	logfile << "Slab type: spectral\n";
	logfile << "Resolution: " << slab_layout.My << "x" << slab_layout.Nx  << "\n";
    logfile << "x_left: " << slab_layout.x_left << "\n";
    logfile << "delta_x" << slab_layout.delta_x << "\n";
    logfile << "y_lo: " << slab_layout.y_lo << "\n";
    logfile << "delta_y: " << slab_layout.delta_y << "\n";

	logfile.close();
}


void diagnostics_cu :: init_diagnostic_output(string filename, string header)
{
    cout << "Initializing output file " << filename << "\n";
	ofstream output;
	output.exceptions(ofstream::badbit);
	output.open(filename.data(), ios::trunc);
	if ( !output )
	{
		throw new std::exception;
	}
	output << header;
	output.close();
}


void diagnostics_cu :: update_arrays(const slab_cuda& slab)
{
	slab.get_data_device(twodads::field_t::f_theta,   theta.get_array_d()  , slab_layout.My, slab_layout.Nx);
	slab.get_data_device(twodads::field_t::f_theta_x, theta_x.get_array_d(), slab_layout.My, slab_layout.Nx);
	slab.get_data_device(twodads::field_t::f_theta_y, theta_y.get_array_d(), slab_layout.My, slab_layout.Nx);
	slab.get_data_device(twodads::field_t::f_omega  , omega.get_array_d()  , slab_layout.My, slab_layout.Nx);
	slab.get_data_device(twodads::field_t::f_omega_x, omega_x.get_array_d(), slab_layout.My, slab_layout.Nx);
	slab.get_data_device(twodads::field_t::f_omega_y, omega_y.get_array_d(), slab_layout.My, slab_layout.Nx);
	slab.get_data_device(twodads::field_t::f_strmf  , strmf.get_array_d(),   slab_layout.My, slab_layout.Nx);
	slab.get_data_device(twodads::field_t::f_strmf_x, strmf_x.get_array_d(), slab_layout.My, slab_layout.Nx);
	slab.get_data_device(twodads::field_t::f_strmf_y, strmf_y.get_array_d(), slab_layout.My, slab_layout.Nx);
}


void diagnostics_cu :: diag_mem(const twodads::real_t time)
{
	size_t free_bytes;
	size_t total_bytes;

	gpuErrchk(cudaMemGetInfo(&free_bytes, &total_bytes));

	cerr << "GPU memory usage:   ";
	cerr << "used: " << (total_bytes - free_bytes) / 1048576 << " MB\t";
	cerr << "free: " << free_bytes / 1048576 << " MB\t";
	cerr << "total:" << total_bytes / 1048675 << " MB\n";
}

void diagnostics_cu :: diag_probes(const twodads::real_t time)
{
	ofstream output;
    stringstream filename;
	const int delta_n {int(slab_layout.Nx) / int(n_probes)};
    const int delta_m {int(slab_layout.My) / int(n_probes)};
    int np;
    int mp;

    cuda_darray<cuda::real_t> theta_tilde(theta.tilde());
    cuda_darray<cuda::real_t> strmf_tilde(strmf.tilde());
    cuda_darray<cuda::real_t> omega_tilde(omega.tilde());
    cuda_darray<cuda::real_t> strmf_y_tilde(strmf_y.tilde());
    cuda_darray<cuda::real_t> strmf_x_tilde(strmf_x.tilde());

    // Write out this for n probes
    // #1: time    #2: n_tilde    #3: n    #4: phi    #5: phi_tilde    #6: Omega    #7: Omega_tilde    #8: v_x    #9: v_y    #10: v_y_tilde    #11: Gamma_r
    for (int n = 0; n < (int) n_probes; n++ )
    {
        np = n * delta_n;
        for(int m = 0; m < (int) n_probes; m++)
        {
            mp = m * delta_m;
            filename << "probe" << setw(3) << setfill('0') << n * n_probes + m << ".dat";
            output.open(filename.str().data(), ofstream::app );
            if ( output.is_open() ){
                output << time << "\t";                                  // time
                output << setw(12) << theta_tilde  (np, mp) << "\t";     // n_tilde
                output << setw(12) << theta        (np, mp) << "\t";     // n
                output << setw(12) << strmf        (np, mp) << "\t";     // phi
                output << setw(12) << strmf_tilde  (np, mp) << "\t";     // phi_tilde
                output << setw(12) << omega        (np, mp) << "\t";     // Omega
                output << setw(12) << omega_tilde  (np, mp) << "\t";     // Omega_tilde
                output << setw(12) << strmf_y_tilde(np, mp) << "\t";     // -v_x
                output << setw(12) << strmf_x      (np, mp) << "\t";     // v_y
                output << setw(12) << strmf_x_tilde(np, mp) << "\t";     // v_y_tilde
                output << "\n";
                output.close();
            }
            filename.str(string());   // Delete current string member variable
        }
    }
}

void diagnostics_cu :: diag_blobs(const twodads::real_t time)
{
	// Maximum value of theta and strmf, including x- and y- position
	twodads::real_t theta_max{-1.0};
	twodads::real_t theta_max_x{slab_layout.x_left};
	twodads::real_t theta_max_y{slab_layout.y_lo};
	twodads::real_t strmf_max{0.0};
	twodads::real_t strmf_max_x{slab_layout.x_left};
	twodads::real_t strmf_max_y{slab_layout.y_lo};
	twodads::real_t theta_int{0.0};

	// Mass center and dispersion tensor elements
	static twodads::real_t theta_com_x{0.0};
	static twodads::real_t theta_com_y{0.0};
	static twodads::real_t wxx{0.0};
	static twodads::real_t wyy{0.0};

	// Mass center velocities and dispersion velocity
	twodads::real_t com_vx{0.0}, com_vy{0.0};
	twodads::real_t dxx{0.0};
	twodads::real_t dyy{0.0};

	// Save old values of com position and dispersion
	twodads::real_t old_com_x{theta_com_x};
	twodads::real_t old_com_y{theta_com_y};
	twodads::real_t old_wxx{wxx};
	twodads::real_t old_wyy{wyy};

	static const twodads::real_t dA{slab_layout.delta_x * slab_layout.delta_y};
	ofstream output;

	if(use_log_theta)
		theta.remove_bg(theta_bg);

//	double m_ts = 0.0;
//	double m_tmax = -1000.0;
//	double m_tmin = 1000.0;
//	double m_x = 0.0;
//	double m_y = 0.0;
//	double m_tx = 0.0;
//	double m_ty = 0.0;
//	double m_wxx = 0.0;
//	double m_wyy = 0.0;
//	for(uint m = 0; m < slab_layout.My; m++)
//	{
//		m_y = slab_layout.y_lo + (double) m * slab_layout.delta_y;
//		for(uint n = 0; n < slab_layout.Nx; n++)
//		{
//			m_x = slab_layout.x_left + (double) n * slab_layout.delta_x;
//			m_ts += theta(m, n);
//			m_tx += theta(m, n) * m_x;
//			m_ty += theta(m, n) * m_y;
//			m_tmax = max(m_tmax, theta(m, n));
//			m_tmin = min(m_tmin, theta(m, n));
//		}
//	}
//
//
//	m_tx /= m_ts;
//	m_ty /= m_ts;
//
//	m_wxx = 0.0;
//	m_wyy = 0.0;
//	for(uint m = 0; m < slab_layout.My; m++)
//	{
//		m_y = slab_layout.y_lo + (double) m * slab_layout.delta_y;
//		for(uint n = 0; n < slab_layout.Nx; n++)
//		{
//			m_x = slab_layout.x_left + (double) n * slab_layout.delta_x;
//			m_wxx = theta(m, n) * (m_x - m_tx) * (m_x - m_tx);
//			m_wyy = theta(m, n) * (m_y - m_ty) * (m_y - m_ty);
//		}
//	}
//	m_wxx /= m_ts;
//	m_wyy /= m_ts;
//
//	cout << "manually: sum(theta) = " << m_ts << endl;
//	cout << "          max(theta) = " << m_tmax << endl;
//	cout << "          min(theta) = " << m_tmin << endl;
//	cout << "          int(theta) = " << m_ts * dA << endl;
//	cout << "          COM_X      = " << m_tx << endl;
//	cout << "          COM_Y      = " << m_ty << endl;
//	cout << "          W_XX       = " << m_wxx << endl;
//	cout << "          W_YY       = " << m_wyy << endl;


	theta_max = theta.get_max();
	strmf_max = strmf.get_max();

	// Compute mass-center position
	theta_int = theta.get_sum();
	theta_com_x = (theta * x_arr).get_sum();
	theta_com_y = (theta * y_arr).get_sum();
	theta_com_x /= theta_int;
	theta_com_y /= theta_int;
	theta_int *= dA;

//	cout << "theta.get_sum() = " << theta.get_sum() << endl;
//	cout << "theta.get_max() = " << theta.get_max() << endl;
//	cout << "theta.get_min() = " << theta.get_min() << endl;
//	cout << "theta_int       = " << theta_int << endl;
//	cout << "theta_com_x     = " << theta_com_x << endl;
//	cout << "theta_com_y     = " << theta_com_y << endl;



//	// Compute dispersion
	for(unsigned int n = 0; n < slab_layout.Nx; n++)
		x_vec[n] = (x_vec[n] - theta_com_x) * (x_vec[n] - theta_com_x);
	for(unsigned int m = 0; m < slab_layout.My; m++)
		y_vec[m] = (y_vec[m] - theta_com_y) * (y_vec[m] - theta_com_y);
	x_arr.upcast_col(x_vec, slab_layout.Nx);
	y_arr.upcast_row(y_vec, slab_layout.My);

	wxx = (theta * x_arr).get_sum() / (theta_int / dA);
	wyy = (theta * y_arr).get_sum() / (theta_int / dA);

//	cout << "wxx            = " << wxx << endl;
//	cout << "wyy            = " << wyy << endl;

	for(unsigned int n = 0; n < slab_layout.Nx; n++)
		x_vec[n] = slab_layout.x_left + (double) n * slab_layout.delta_x;

	for(unsigned int m = 0; m < slab_layout.My; m++)
		y_vec[m] = slab_layout.y_lo + (double) m * slab_layout.delta_y;
	x_arr.upcast_col(x_vec, slab_layout.Nx);
	y_arr.upcast_row(y_vec, slab_layout.My);

	if(use_log_theta)
		theta.add_bg(theta_bg);

	// Update center-of-mass velocities with forward finite-difference scheme
	com_vx = (theta_com_x - old_com_x) / dt_diag;
	com_vy = (theta_com_y - old_com_y) / dt_diag;
	// Update dispersion velocities with forward finite-difference scheme
	dxx = (wxx - old_wxx) / dt_diag;
	dyy = (wyy - old_wyy) / dt_diag;


	cout << "===================================diag_blobs: time = " << time << endl;
	output.open("cu_blobs.dat", ios::app);
	if(output.is_open())
	{
		output << time << "\t";
		output << setw(12) << theta_max << "\t";
		output << setw(12) << theta_max_x << "\t";
		output << setw(12) << theta_max_y << "\t";
		output << setw(12) << strmf_max << "\t";
		output << setw(12) << strmf_max_x << "\t";
		output << setw(12) << strmf_max_y << "\t";
		output << setw(12) << theta_int << "\t";
		output << setw(12) << theta_com_x << "\t";
		output << setw(12) << theta_com_y << "\t";
		output << setw(12) << com_vx << "\t";
		output << setw(12) << com_vy << "\t";
		output << setw(12) << wxx << "\t";
		output << setw(12) << wyy << "\t";
		output << setw(12) << dxx << "\t";
		output << setw(12) << dyy << "\n";
		output.close();
	}
}

void diagnostics_cu :: diag_energy(const twodads::real_t time)
{
	ofstream output;
	//static const twodads::real_t Lx{slab_layout.delta_y * twodads::real_t(slab_layout.My)};
	//static const twodads::real_t Ly{slab_layout.delta_x * twodads::real_t(slab_layout.Nx)};
	static const twodads::real_t dA{slab_layout.delta_x * slab_layout.delta_y};

	cuda_darray<twodads::real_t> strmf_tilde(std::move(strmf.tilde()));
	cuda_darray<twodads::real_t> theta_tilde(std::move(theta.tilde()));
	cuda_darray<twodads::real_t> omega_tilde(std::move(omega.tilde()));
	cuda_darray<twodads::real_t> strmf_x_tilde(std::move(strmf_x.tilde()));
	cuda_darray<twodads::real_t> strmf_y_tilde(std::move(strmf_y.tilde()));

	// Total energy, thermal and kinetic
	const twodads::real_t E{0.5 * dA * ((theta_tilde * theta_tilde) + (strmf_x * strmf_x) + (strmf_y * strmf_y)).get_sum()};
	// Kinetic energy
	const twodads::real_t K{0.5 * dA * ((strmf_x * strmf_x) + (strmf_y * strmf_y)).get_sum()};
	// Generalized enstrophy
	const twodads::real_t U{0.5 * dA * ((theta_tilde - omega_tilde) * (theta_tilde - omega_tilde)).get_sum()};
	// T
	const twodads::real_t T{0.5 * dA * (strmf_x_tilde * strmf_y_tilde).get_sum()};
	// W
	cout << "diag_energy: " << time << setw(12) << E << "\n";

	output.open("cu_energy.dat", ios::app);
	if(output.is_open())
	{
		output << time << "\t";
		output << setw(12) << E << "\t";
		output << setw(12) << K << "\t";
		output << setw(12) << T << "\t";
		output << setw(12) << U << "\t";
		output << "\n";
		output.close();
	}
}



