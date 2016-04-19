/*
 *
 * Implmentation of diagnostics class using cuda_darray
 *
 */


#include <fstream>
#include "diagnostics_cu.h"

const map <twodads::diagnostic_t, std::string> dfile_fname{
	{twodads::diagnostic_t::diag_blobs, "cu_blobs.dat"},
    {twodads::diagnostic_t::diag_com_tau, "cu_com_tau.dat"},
    {twodads::diagnostic_t::diag_com_theta, "cu_com_theta.dat"},
    {twodads::diagnostic_t::diag_max_theta, "cu_max_theta.dat"},
    {twodads::diagnostic_t::diag_max_tau, "cu_max_tau.dat"},
    {twodads::diagnostic_t::diag_max_omega, "cu_max_omega.dat"},
    {twodads::diagnostic_t::diag_max_strmf, "cu_max_strmf.dat"},
	{twodads::diagnostic_t::diag_probes, "cu_probes.dat"},
	{twodads::diagnostic_t::diag_energy, "cu_energy.dat"},
	{twodads::diagnostic_t::diag_energy_ns, "cu_energy_ns.dat"},
    {twodads::diagnostic_t::diag_consistency, "cu_consistency.dat"},
	{twodads::diagnostic_t::diag_mem, "cu_memory.dat"}
};

// Use fields with width 18
const map <twodads::diagnostic_t, std::string> dfile_header{
	{twodads::diagnostic_t::diag_blobs,     "#01: time         #02: theta_max   #03: theta_max_x #04: theta_max_y  #05: strmf_max    #06: strmf_max_x  #07: strmf_max_y  #08: theta_int   #09: theta_int_x   #10: theta_int_y  #11: COMvx        #12: COMvy       #13: Wxx          #14: Wyy          #15: Dxx          #16: Dyy\n"},
    {twodads::diagnostic_t::diag_energy,    "#01: time         #02: E           #03: K           #04: U            #05: T            #06: D            #07: A            #08: D_omega     #09: D_theta       #10: Gamma_tilde  #11: J_tilde      #12: CFL           \n"},
	{twodads::diagnostic_t::diag_energy_ns, "#01: time         #02: V           #03: O           #04: E            #05: D1           #06: D2           #07: D4           #08CFL\n"},
	{twodads::diagnostic_t::diag_probes,    "#01: time         #02: n_tilde     #03: n           #04: phi          #05: phi_tilde    #06: Omega        #07: Omega_tilde  #08: v_x         #09: v_y           #10: v_y_tilde    #11: Gamma_r\n"},
    {twodads::diagnostic_t::diag_com_theta, "#01: time         #02: X_com       #03: Y_com       #04: VX_com       #05: VY_com       #06: Wxx          #07: Wyy          #08: Dxx         #09: Dyy\n"},
    {twodads::diagnostic_t::diag_com_tau,   "#01: time         #02: X_com       #03: Y_com       #04: VX_com       #05: VY_com       #06: Wxx          #07: Wyy          #08: Dxx         #09: Dyy\n"},
    {twodads::diagnostic_t::diag_max_theta, "#01: time         #02: max         #03: x(max)      #04: y(max)       #05: min          #06: x(min)       #07: y(min)\n"},
    {twodads::diagnostic_t::diag_max_tau,   "#01: time         #02: max         #03: x(max)      #04: y(max)       #05: min          #06: x(min)       #07: y(min)\n"},
    {twodads::diagnostic_t::diag_max_omega, "#01: time         #02: max         #03: x(max)      #04: y(max)       #05: min          #06: x(min)       #07: y(min)\n"},
    {twodads::diagnostic_t::diag_max_strmf, "#01: time         #02: max         #03: x(max)      #04: y(max)       #05: min          #06: x(min)       #07: y(min)\n"},
    {twodads::diagnostic_t::diag_consistency, "none"},
	{twodads::diagnostic_t::diag_mem, "none"}
};

/****************************************
 * Update get_dfunc_by_name in the constructor when including another diagnostic routine
 */


diagnostics_cu :: diagnostics_cu(const slab_config& cfg) :
    config(cfg), 
    slab_layout{config.get_xleft(), config.get_deltax(), config.get_ylow(), config.get_deltay(),
                config.get_deltat(), config.get_my(), config.get_nx()},
    theta(config.get_my(), config.get_nx()),
    theta_x(config.get_my(), config.get_nx()),
    theta_y(config.get_my(), config.get_nx()),
    tau(config.get_my(), config.get_nx()),
    tau_x(config.get_my(), config.get_nx()),
    tau_y(config.get_my(), config.get_nx()),
    omega(config.get_my(), config.get_nx()),
    omega_x(config.get_my(), config.get_nx()),
    omega_y(config.get_my(), config.get_nx()),
    strmf(config.get_my(), config.get_nx()),
    strmf_x(config.get_my(), config.get_nx()),
    strmf_y(config.get_my(), config.get_nx()),
    theta_xx(config.get_my(), config.get_nx()),
    theta_yy(config.get_my(), config.get_nx()),
    omega_xx(config.get_my(), config.get_nx()),
    omega_yy(config.get_my(), config.get_nx()),
    theta_rhs(config.get_my(), config.get_nx()),
    tau_rhs(config.get_my(), config.get_nx()),
    omega_rhs(config.get_my(), config.get_nx()),
    tmp_array(1, config.get_my(), config.get_nx() / 2 + 1),
    x_vec(new twodads::real_t[config.get_nx()]),
    y_vec(new twodads::real_t[config.get_my()]),
    x_arr(config.get_my(), config.get_nx()),
    y_arr(config.get_my(), config.get_nx()),
    der(slab_layout),
    time(0.0),
    dt_diag(config.get_tdiag()),
    n_probes(config.get_nprobe()),
    use_log_theta(config.get_log_theta()),
    theta_bg(config.get_initc_theta(0)),
    get_darr_by_name{{twodads::field_t::f_theta,     &theta},
                	 {twodads::field_t::f_theta_x,   &theta_x},
                	 {twodads::field_t::f_theta_y,   &theta_y},
                	 {twodads::field_t::f_tau,       &tau},
                	 {twodads::field_t::f_tau_x,     &tau_x},
                	 {twodads::field_t::f_tau_y,     &tau_y},
                	 {twodads::field_t::f_omega,     &omega},
                	 {twodads::field_t::f_omega_x,   &omega_x},
                	 {twodads::field_t::f_omega_y,   &omega_y},
                	 {twodads::field_t::f_strmf,     &strmf},
                	 {twodads::field_t::f_strmf_x,   &strmf_x},
                	 {twodads::field_t::f_strmf_y,   &strmf_y},
                	 {twodads::field_t::f_theta_rhs, &theta_rhs},
                	 {twodads::field_t::f_tau_rhs,   &tau_rhs},
                	 {twodads::field_t::f_omega_rhs, &omega_rhs}
                },
    get_dfunc_by_name{{twodads::diagnostic_t::diag_energy,  &diagnostics_cu::diag_energy},
                      {twodads::diagnostic_t::diag_energy_ns,  &diagnostics_cu::diag_energy_ns},
                      {twodads::diagnostic_t::diag_com_theta,  &diagnostics_cu::diag_com_theta},
                      {twodads::diagnostic_t::diag_com_tau,  &diagnostics_cu::diag_com_tau},
                      {twodads::diagnostic_t::diag_max_theta,  &diagnostics_cu::diag_max_theta},
                      {twodads::diagnostic_t::diag_max_tau,  &diagnostics_cu::diag_max_tau},
                      {twodads::diagnostic_t::diag_max_omega,  &diagnostics_cu::diag_max_omega},
                      {twodads::diagnostic_t::diag_max_strmf,  &diagnostics_cu::diag_max_strmf},
                      {twodads::diagnostic_t::diag_probes,  &diagnostics_cu::diag_probes},
                      {twodads::diagnostic_t::diag_consistency,  &diagnostics_cu::diag_consistency},
                      {twodads::diagnostic_t::diag_mem,  &diagnostics_cu::diag_mem}}
{
    stringstream filename;
    string header;

    // Create x_arr and y_arr
	// Create cuda arrays with x-values along the columns
	for(unsigned int n = 0; n < slab_layout.Nx; n++)
		x_vec[n] = slab_layout.x_left + slab_layout.delta_x * (cuda::real_t) n;
        //x_vec[n] = 2.0;
	x_arr.upcast_row(x_vec, slab_layout.Nx);

	for(unsigned int m = 0; m < slab_layout.My; m++)
		y_vec[m] = slab_layout.y_lo + slab_layout.delta_y * (cuda::real_t) m;
        //y_vec[m] = 2.0;
	y_arr.upcast_col(y_vec, slab_layout.My);

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
	logfile.open("run.log", ios::trunc);

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
    if(config.get_log_theta())
    {
        theta.op_apply_t<d_op0_expassign<cuda::real_t> >(0);
        theta_x.op_array_t<d_op1_mulassign<cuda::real_t> >(theta, 0);
        theta_y.op_array_t<d_op1_mulassign<cuda::real_t> >(theta, 0);
    }
	slab.get_data_device(twodads::field_t::f_tau,   tau.get_array_d()  , slab_layout.My, slab_layout.Nx);
	slab.get_data_device(twodads::field_t::f_tau_x, tau_x.get_array_d(), slab_layout.My, slab_layout.Nx);
	slab.get_data_device(twodads::field_t::f_tau_y, tau_y.get_array_d(), slab_layout.My, slab_layout.Nx);

    if(config.get_log_tau())
    {
        tau.op_apply_t<d_op0_expassign<cuda::real_t> >(0);
        tau_x.op_array_t<d_op1_mulassign<cuda::real_t> >(tau, 0);
        tau_y.op_array_t<d_op1_mulassign<cuda::real_t> >(tau, 0);
    }

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
                output << setw(20) << std::fixed << std::setprecision(16) <<  theta_tilde  (np, mp) << "\t";     // n_tilde
                output << setw(20) << std::fixed << std::setprecision(16) <<  theta        (np, mp) << "\t";     // n
                output << setw(20) << std::fixed << std::setprecision(16) <<  strmf        (np, mp) << "\t";     // phi
                output << setw(20) << std::fixed << std::setprecision(16) <<  strmf_tilde  (np, mp) << "\t";     // phi_tilde
                output << setw(20) << std::fixed << std::setprecision(16) << omega        (np, mp) << "\t";     // Omega
                output << setw(20) << std::fixed << std::setprecision(16) << omega_tilde  (np, mp) << "\t";     // Omega_tilde
                output << setw(20) << std::fixed << std::setprecision(16) << strmf_y_tilde(np, mp) << "\t";     // -v_x
                output << setw(20) << std::fixed << std::setprecision(16) << strmf_x      (np, mp) << "\t";     // v_y
                output << setw(20) << std::fixed << std::setprecision(16) << strmf_x_tilde(np, mp) << "\t";     // v_y_tilde
                output << "\n";
                output.close();
            }
            filename.str(string());   // Delete current string member variable
        }
    }
}

void diagnostics_cu :: diag_consistency(const twodads::real_t time)
{
    cuda::real_t l2norm;
    static cuda_darray<cuda::real_t> invdel2_omega(config.get_my(), config.get_nx());
    static cuda_darray<cuda::real_t> del2phi(config.get_my(), config.get_nx());
    static cuda_darray<CuCmplx<cuda::real_t> >tmp_hat(config.get_my(), config.get_nx() / 2 + 1);
    // By construction, phi should not have a mean
    assert(strmf.get_mean() < 1e-10);

    ///////////////////////////////////////////// Reconstruct phi from current omega in real space
    /* Test of phi = del^-2 omega */
    der.inv_laplace(omega, invdel2_omega, 0);
    l2norm = sqrt(((invdel2_omega - strmf) *(invdel2_omega - strmf)).get_mean());
    cout << "|del^{-2}(omega) - phi|_2 = " << l2norm << "\t";
    assert(l2norm < 1e-8);

    /////////////////////////////////////////////// Reconstruct omega from current phi in real space
    der.d_laplace(strmf, del2phi, 0);
    del2phi += omega.get_mean();
    l2norm = sqrt(((del2phi - omega) * (del2phi - omega)).get_mean());
    cout << "|del^2(phi) - omega|_2 = " << l2norm << "\t";
    assert(l2norm < 1e-8);

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // Test accuracy of DFT with omega
    der.dft_r2c(omega.get_array_d(), tmp_hat.get_array_d());
    der.dft_c2r(tmp_hat.get_array_d(), del2phi.get_array_d());
    del2phi.normalize();
    l2norm = sqrt(((del2phi - omega) * (del2phi - omega)).get_mean());
    cout << "|iFFT(FFT(omega)) - omega|_2 = " << l2norm << "\t";
    assert(l2norm < 1e-10);

    der.dft_r2c(strmf.get_array_d(), tmp_hat.get_array_d());
    der.dft_c2r(tmp_hat.get_array_d(), invdel2_omega.get_array_d());
    invdel2_omega.normalize();
    l2norm = sqrt(((invdel2_omega- strmf) * (invdel2_omega - strmf)).get_mean());
    cout << "|iFFT(FFT(strmf)) - strmf|_2 = " << l2norm << endl;
    assert(l2norm < 1e-10);

}

void diagnostics_cu :: diag_com_theta(const twodads::real_t time)
{
    cout << "diag_com_theta " << endl;
    static com_t old_com{0.0, 0.0, 0.0, 0.0};
    diag_com(twodads::field_t::f_theta, old_com, time);
}

void diagnostics_cu :: diag_com_tau(const twodads::real_t time)
{
    cout << "diag_com_tau " << endl;
    static com_t old_com{0.0, 0.0, 0.0, 0.0};
    diag_com(twodads::field_t::f_tau, old_com, time);
}

void diagnostics_cu :: diag_com(const twodads::field_t fname, com_t& old_com, const twodads::real_t time)
{
    static const std::map<twodads::field_t, std::string> get_com_fname_by_name{{twodads::field_t::f_theta, "cu_com_theta.dat"},
                                                                               {twodads::field_t::f_tau, "cu_com_tau.dat"}};
    twodads::real_t X_c{0.0};
    twodads::real_t Y_c{0.0};
    twodads::real_t W_xx{0.0};
    twodads::real_t W_yy{0.0};
    // Mass center velocities and dispersion velocity
    twodads::real_t com_vx{0.0}, com_vy{0.0};
    twodads::real_t D_xx{0.0}, D_yy{0.0};

    // The array we operate on
    cuda_darray<twodads::real_t>* arr_ptr = get_darr_by_name.at(fname);
    twodads::real_t array_int{0.0};
    // Surface area element
    ofstream output;

    twodads::real_t bg_val = 0.25 * ((*arr_ptr)(0,0) + 
                                     (*arr_ptr)(0, config.get_nx() - 1) + 
                                     (*arr_ptr)(config.get_my() - 1, 0) +
                                     (*arr_ptr)(config.get_my() - 1, config.get_nx() -1 ));


    // Remove background before computing center of mass. Only when using logarithmic
    // formulation of thermodynamic fields
    if(fname == twodads::field_t::f_theta) 
        if(config.get_log_theta())
        {
            cout << "subtracting bg_val = " << bg_val << endl;
            (*arr_ptr).remove_bg(bg_val);
        }
    else if (fname == twodads::field_t::f_tau)
        if(config.get_log_tau())
            (*arr_ptr).remove_bg(bg_val);

    array_int = (*arr_ptr).get_sum();
	X_c = ((*arr_ptr) * x_arr).get_sum() / array_int;
	Y_c = ((*arr_ptr) * y_arr).get_sum() / array_int;

    cout << "sum = " << (*arr_ptr).get_sum() << ", arr * x.sum() = " << ((*arr_ptr) * x_arr).get_sum() << ", X_c = " << X_c << endl;

	// Compute dispersion
	//for(unsigned int n = 0; n < slab_layout.Nx; n++)
	//	x_vec[n] = (x_vec[n] - X_c) * (x_vec[n] - X_c);
	//for(unsigned int m = 0; m < slab_layout.My; m++)
	//	y_vec[m] = (y_vec[m] - Y_c) * (y_vec[m] - Y_c);
	//x_arr.upcast_col(x_vec, slab_layout.Nx);
	//y_arr.upcast_row(y_vec, slab_layout.My);

	//W_xx = ((*arr_ptr) * x_arr).get_sum() / array_int;
	//W_yy = ((*arr_ptr) * y_arr).get_sum() / array_int;

	//for(unsigned int n = 0; n < slab_layout.Nx; n++)
	//	x_vec[n] = slab_layout.x_left + (double) n * slab_layout.delta_x;

	//for(unsigned int m = 0; m < slab_layout.My; m++)
	//	y_vec[m] = slab_layout.y_lo + (double) m * slab_layout.delta_y;
	//x_arr.upcast_col(x_vec, slab_layout.Nx);
	//y_arr.upcast_row(y_vec, slab_layout.My);

    if(fname == twodads::field_t::f_theta) 
        if(config.get_log_theta())
            (*arr_ptr).add_bg(bg_val);
    else if (fname == twodads::field_t::f_tau)
        if(config.get_log_tau())
            (*arr_ptr).add_bg(bg_val);

    // Update center of mass velocities with forward finite difference scheme
    com_vx = (X_c - old_com.X_c) / dt_diag;
    com_vy = (Y_c - old_com.Y_c) / dt_diag;
    old_com.X_c = X_c;
    old_com.Y_c = Y_c;
    // Update dispersion velocities with forward finite difference scheme
    D_xx = (W_xx - old_com.W_xx) / dt_diag;
    D_yy = (W_yy - old_com.W_yy) / dt_diag;
    old_com.W_xx = W_xx;
    old_com.W_yy = W_yy;

    output.open(get_com_fname_by_name.at(fname), ios::app);
    if(output.is_open())
    {
        output << time << "\t";
        output << setw(20) << std::fixed << std::setprecision(16) << X_c << "\t";
        output << setw(20) << std::fixed << std::setprecision(16) << Y_c << "\t";
        output << setw(20) << std::fixed << std::setprecision(16) << com_vx << "\t";
        output << setw(20) << std::fixed << std::setprecision(16) << com_vy << "\t";
        output << setw(20) << std::fixed << std::setprecision(16) << W_xx << "\t";
        output << setw(20) << std::fixed << std::setprecision(16) << W_yy << "\t";
        output << setw(20) << std::fixed << std::setprecision(16) << D_xx << "\t";
        output << setw(20) << std::fixed << std::setprecision(16) << D_yy << "\n";
        output.close();
    }
}


void diagnostics_cu :: diag_max(const twodads::field_t fname, const twodads::real_t time)
{
    static const std::map<twodads::field_t, std::string> get_max_fname_by_name{{twodads::field_t::f_theta, "cu_max_theta.dat"},
                                                                               {twodads::field_t::f_tau, "cu_max_tau.dat"},
                                                                               {twodads::field_t::f_omega, "cu_max_omega.dat"},
                                                                               {twodads::field_t::f_strmf, "cu_max_strmf.dat"}};
    cuda_darray<twodads::real_t>* arr_ptr = get_darr_by_name.at(fname);
    twodads::real_t max_val{arr_ptr -> get_max()};
    twodads::real_t min_val{arr_ptr -> get_min()};

    twodads::real_t max_x{0.0}, max_y{0.0};
    twodads::real_t min_x{0.0}, min_y{0.0};

    ofstream output;

    output.open(get_max_fname_by_name.at(fname), ios::app);
    if(output.is_open())
    {
        output << time << "\t";
        output << setw(20) << std::fixed << std::setprecision(16) << max_val << "\t";
        output << setw(20) << std::fixed << std::setprecision(16) << max_x << "\t";
        output << setw(20) << std::fixed << std::setprecision(16) << max_y << "\t";
        output << setw(20) << std::fixed << std::setprecision(16) << min_val << "\t";
        output << setw(20) << std::fixed << std::setprecision(16) << min_x << "\t";
        output << setw(20) << std::fixed << std::setprecision(16) << min_y << "\n";
    }
}


/*
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

	theta_max = theta.get_max();
	strmf_max = strmf.get_max();

	// Compute mass-center position
	theta_int = theta.get_sum();
	theta_com_x = (theta * x_arr).get_sum() / theta_int;
	theta_com_y = (theta * y_arr).get_sum() / theta_int;
	//theta_com_x /= theta_int;
	//theta_com_y /= theta_int;
	theta_int *= dA;

	// Compute dispersion
	for(unsigned int n = 0; n < slab_layout.Nx; n++)
		x_vec[n] = (x_vec[n] - theta_com_x) * (x_vec[n] - theta_com_x);
	for(unsigned int m = 0; m < slab_layout.My; m++)
		y_vec[m] = (y_vec[m] - theta_com_y) * (y_vec[m] - theta_com_y);
	x_arr.upcast_col(x_vec, slab_layout.Nx);
	y_arr.upcast_row(y_vec, slab_layout.My);

	wxx = (theta * x_arr).get_sum() / (theta_int / dA);
	wyy = (theta * y_arr).get_sum() / (theta_int / dA);

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
		output << setw(20) << std::fixed << std::setprecision(16) << theta_max << "\t";
		output << setw(20) << std::fixed << std::setprecision(16) << theta_max_x << "\t";
		output << setw(20) << std::fixed << std::setprecision(16) << theta_max_y << "\t";
		output << setw(20) << std::fixed << std::setprecision(16) << strmf_max << "\t";
		output << setw(20) << std::fixed << std::setprecision(16) << strmf_max_x << "\t";
		output << setw(20) << std::fixed << std::setprecision(16) << strmf_max_y << "\t";
		output << setw(20) << std::fixed << std::setprecision(16) << theta_int << "\t";
		output << setw(20) << std::fixed << std::setprecision(16) << theta_com_x << "\t";
		output << setw(20) << std::fixed << std::setprecision(16) << theta_com_y << "\t";
		output << setw(20) << std::fixed << std::setprecision(16) << com_vx << "\t";
		output << setw(20) << std::fixed << std::setprecision(16) << com_vy << "\t";
		output << setw(20) << std::fixed << std::setprecision(16) << wxx << "\t";
		output << setw(20) << std::fixed << std::setprecision(16) << wyy << "\t";
		output << setw(20) << std::fixed << std::setprecision(16) << dxx << "\t";
		output << setw(20) << std::fixed << std::setprecision(16) << dyy << "\n";
		output.close();
	}
}
*/

void diagnostics_cu :: diag_energy_ns(const twodads::real_t time)
{
    ofstream output;
    static const twodads::real_t dA{slab_layout.delta_x * slab_layout.delta_y};

    // Integrated vorticity
    const twodads::real_t V{dA * omega.get_sum()};
    // Total Enstrophy
    const twodads::real_t O{0.5 * dA * (omega * omega).get_sum()};
    // Kinetic energy
    const twodads::real_t E{0.5 * dA * (strmf_x * strmf_x + strmf_y * strmf_y).get_sum()};

    const twodads::real_t cfl_x{strmf_x.get_absmax() * slab_layout.delta_t / slab_layout.delta_x};
    const twodads::real_t cfl_y{strmf_y.get_absmax() * slab_layout.delta_t / slab_layout.delta_y};

    
    // Loss due to diffusion
    //cuda_darray<cuda::real_t> omega_x2(config.get_my(), config.get_nx());
    //cuda_darray<cuda::real_t> omega_y2(config.get_my(), config.get_nx());
    //der.d_dx2_dy2(omega, omega_x2, omega_y2);

    // \int (grad Omega)^2 dA
    const twodads::real_t D1{dA * (omega_x * omega_x + omega_y * omega_y).get_sum()};

    // Energy conservation, poisson bracket
    const twodads::real_t D2{dA * (strmf * (strmf_x * omega_y - strmf_y * omega_x)).get_sum()};
    // Enstrophy conservation, poisson bracket
    const twodads::real_t D3{dA * (omega * (strmf_x * omega_y - strmf_y * omega_x)).get_sum()};

    output.open("cu_energy_ns.dat", ios::app);
    if(output.is_open())
    {
         output << time << "\t";
         output << setw(20) << std::fixed << std::setprecision(16) << V << "\t";
         output << setw(20) << std::fixed << std::setprecision(16) << O << "\t";
         output << setw(20) << std::fixed << std::setprecision(16) << E << "\t";
         output << setw(20) << std::fixed << std::setprecision(16) << D1 << "\t";
         output << setw(20) << std::fixed << std::setprecision(16) << D2 << "\t";
         output << setw(20) << std::fixed << std::setprecision(16) << D3 << "\t";
         output << setw(20) << std::fixed << std::setprecision(16) << cfl_x + cfl_y << "\n";
    } 

}

void diagnostics_cu :: diag_energy(const twodads::real_t time)
{
	ofstream output;
	static const twodads::real_t dA{slab_layout.delta_x * slab_layout.delta_y};

	cuda_darray<twodads::real_t> strmf_tilde(std::move(strmf.tilde()));
	cuda_darray<twodads::real_t> theta_tilde(std::move(theta.tilde()));
	cuda_darray<twodads::real_t> omega_tilde(std::move(omega.tilde()));
	cuda_darray<twodads::real_t> strmf_x_tilde(std::move(strmf_x.tilde()));
	cuda_darray<twodads::real_t> strmf_y_tilde(std::move(strmf_y.tilde()));

    cuda_darray<twodads::real_t> diag_strmf_x(strmf_x);
    diag_strmf_x.op_apply_t<d_op0_absassign<cuda::real_t> >(0);
    cuda_darray<twodads::real_t> diag_strmf_y(strmf_y);
    diag_strmf_y.op_apply_t<d_op0_absassign<cuda::real_t> >(0);


    der.d_dx2_dy2(omega, omega_xx, omega_yy);
    der.d_dx2_dy2(theta, theta_xx, theta_yy);


	// Total energy, thermal and kinetic
	const twodads::real_t E{0.5 * dA * ((theta * theta) + (strmf_x * strmf_x) + (strmf_y * strmf_y)).get_sum()};
	// Kinetic energy
	const twodads::real_t K{0.5 * dA * ((strmf_x * strmf_x) + (strmf_y * strmf_y)).get_sum()};
    // Enstrophy
    const twodads::real_t U{0.5 * dA * (omega * omega).get_sum()};
	// Densities
	const twodads::real_t T{dA * ((theta_x * theta_x) + (theta_y * theta_y)).get_sum()};
    // Density
    const twodads::real_t D{0.5 * dA * (theta * theta).get_sum()};
    // A = phi(phi - n)
    const twodads::real_t A{dA * (strmf * (strmf - theta)).get_sum()};
    // B = n(phi - n)
    // D_omega = \int dA phi (omega_xx + omega_yy)
    const twodads::real_t D_omega{dA * (strmf * (omega_xx + omega_yy)).get_sum()};
    // D_n = \int dA theta (theta_xx + theta_yy)
    const twodads::real_t D_theta {dA * (theta * (theta_xx + theta_yy)).get_sum()};
    // Gamma_tilde = - \int dA theta_tilde * strmf_y
    const twodads::real_t Gamma_tilde {-1.0 * dA * (theta_tilde * strmf_y).get_sum()};
    // J_tilde = alpha * \int dA (theta_tilde - strmf_tilde) * (theta_tilde - strmf_tilde)
    const twodads::real_t J_tilde{dA * ((theta_tilde - strmf_tilde) * (theta_tilde - strmf_tilde)).get_sum()};
	// CFL
    const twodads::real_t CFL{diag_strmf_x.get_absmax() * slab_layout.delta_t / slab_layout.delta_x + 
                              diag_strmf_y.get_absmax() * slab_layout.delta_t / slab_layout.delta_y};

	output.open("cu_energy.dat", ios::app);
	if(output.is_open())
	{
		output << time << "\t";
		output << setw(20) << std::fixed << std::setprecision(16) << E << "\t";
		output << setw(20) << std::fixed << std::setprecision(16) << K << "\t";
		output << setw(20) << std::fixed << std::setprecision(16) << U << "\t";
		output << setw(20) << std::fixed << std::setprecision(16) << T << "\t";
        output << setw(20) << std::fixed << std::setprecision(16) << D << "\t";
        output << setw(20) << std::fixed << std::setprecision(16) << A << "\t";
        output << setw(20) << std::fixed << std::setprecision(16) << D_omega << "\t";
        output << setw(20) << std::fixed << std::setprecision(16) << D_theta << "\t";
        output << setw(20) << std::fixed << std::setprecision(16) << Gamma_tilde << "\t";
        output << setw(20) << std::fixed << std::setprecision(16) << J_tilde << "\t";

        output << setw(20) << std::fixed << std::setprecision(16) << CFL;
		output << "\n";
		output.close();
	}
}

