/*
 * Implementation of slab_bc
 */

#include "slab_bc.h"

using namespace std;

map<twodads::rhs_t, slab_bc::rhs_func_ptr> slab_bc :: rhs_func_map = slab_bc::create_rhs_func_map();

slab_bc :: slab_bc(const slab_config_js& _conf) :
    conf(_conf),
    output(_conf),
    diagnostic(_conf),
#ifdef DEVICE
    myfft{new cufft_object_t<twodads::real_t>(get_config().get_geom(), get_config().get_dft_t())},
#endif //DEVICE
#ifdef HOST
    myfft{new fftw_object_t<twodads::real_t>(get_config().get_geom(), get_config().get_dft_t())},
#endif //HOST
    tint_theta{nullptr},
    tint_omega{nullptr},
    tint_tau{nullptr},
    theta(get_config().get_geom(),     get_config().get_bvals(twodads::field_t::f_theta), get_config().get_tlevs()), 
    theta_x(get_config().get_geom(),   get_config().get_bvals(twodads::field_t::f_theta), 1), 
    theta_y(get_config().get_geom(),   get_config().get_bvals(twodads::field_t::f_theta), 1),
    omega(get_config().get_geom(),     get_config().get_bvals(twodads::field_t::f_omega), get_config().get_tlevs()), 
    omega_x(get_config().get_geom(),   get_config().get_bvals(twodads::field_t::f_omega), 1), 
    omega_y(get_config().get_geom(),   get_config().get_bvals(twodads::field_t::f_omega), 1),
    tau(get_config().get_geom(),       get_config().get_bvals(twodads::field_t::f_tau), get_config().get_tlevs()),   
    tau_x(get_config().get_geom(),     get_config().get_bvals(twodads::field_t::f_tau), 1),   
    tau_y(get_config().get_geom(),     get_config().get_bvals(twodads::field_t::f_tau), 1),
    strmf(get_config().get_geom(),     get_config().get_bvals(twodads::field_t::f_strmf), 1), 
    strmf_x(get_config().get_geom(),   get_config().get_bvals(twodads::field_t::f_strmf), 1), 
    strmf_y(get_config().get_geom(),   get_config().get_bvals(twodads::field_t::f_strmf), 1),
    theta_rhs(get_config().get_geom(), get_config().get_bvals(twodads::field_t::f_theta), get_config().get_tlevs() - 1),
    omega_rhs(get_config().get_geom(), get_config().get_bvals(twodads::field_t::f_omega), get_config().get_tlevs() - 1),
    tau_rhs(get_config().get_geom(),   get_config().get_bvals(twodads::field_t::f_tau), get_config().get_tlevs() - 1),
    get_field_by_name{ {twodads::field_t::f_theta,     &theta},
                       {twodads::field_t::f_theta_x,   &theta_x},
                       {twodads::field_t::f_theta_y,   &theta_y},
                       {twodads::field_t::f_omega,     &omega},
                       {twodads::field_t::f_omega_x,   &omega_x},
                       {twodads::field_t::f_omega_y,   &omega_y},
                       {twodads::field_t::f_tau,       &tau},
                       {twodads::field_t::f_tau_x,     &tau_x},
                       {twodads::field_t::f_tau_y,     &tau_y},
                       {twodads::field_t::f_strmf,     &strmf},
                       {twodads::field_t::f_strmf_x,   &strmf_x},
                       {twodads::field_t::f_strmf_y,   &strmf_y},
                       {twodads::field_t::f_theta_rhs, &theta_rhs},
                       {twodads::field_t::f_omega_rhs, &omega_rhs},
                       {twodads::field_t::f_tau_rhs, &tau_rhs}},
    get_dfield_by_name{ {twodads::dyn_field_t::f_theta, &theta},
                        {twodads::dyn_field_t::f_omega, &omega},
                        {twodads::dyn_field_t::f_tau,   &tau}},
    get_output_by_name{ {twodads::output_t::o_theta,    &theta},
                        {twodads::output_t::o_theta_x,  &theta_x},
                        {twodads::output_t::o_theta_y,  &theta_y},
                        {twodads::output_t::o_omega,    &omega},
                        {twodads::output_t::o_omega_x,  &omega_x},
                        {twodads::output_t::o_omega_y,  &omega_y},
                        {twodads::output_t::o_tau,      &tau},
                        {twodads::output_t::o_tau_x,    &tau_x},
                        {twodads::output_t::o_tau_y,    &tau_y},
                        {twodads::output_t::o_strmf,    &strmf},
                        {twodads::output_t::o_strmf_x,  &strmf_x},
                        {twodads::output_t::o_strmf_y,  &strmf_y}},
    theta_rhs_func{rhs_func_map.at(get_config().get_rhs_t(twodads::dyn_field_t::f_theta))},
    omega_rhs_func{rhs_func_map.at(get_config().get_rhs_t(twodads::dyn_field_t::f_omega))},
    tau_rhs_func{rhs_func_map.at(get_config().get_rhs_t(twodads::dyn_field_t::f_tau))}
{
    puts(__PRETTY_FUNCTION__);
    switch(get_config().get_grid_type())
    {
        case twodads::grid_t::vertex_centered:
#ifdef HOST
            my_derivs = new deriv_spectral_t<value_t, allocator_host>(get_config().get_geom());
            tint_theta = new integrator_karniadakis_bs_t<value_t, allocator_host>(get_config().get_geom(), get_config().get_bvals(twodads::field_t::f_theta), get_config().get_tint_params(twodads::dyn_field_t::f_theta));
            tint_omega = new integrator_karniadakis_bs_t<value_t, allocator_host>(get_config().get_geom(), get_config().get_bvals(twodads::field_t::f_omega), get_config().get_tint_params(twodads::dyn_field_t::f_omega));
            tint_tau = new integrator_karniadakis_bs_t<value_t, allocator_host>(get_config().get_geom(), get_config().get_bvals(twodads::field_t::f_tau), get_config().get_tint_params(twodads::dyn_field_t::f_tau));
#endif //HOST
#ifdef DEVICE
            my_derivs = new deriv_spectral_t<value_t, allocator_device>(get_config().get_geom());
            tint_theta = new integrator_karniadakis_bs_t<value_t, allocator_devicet>(get_config().get_geom(), get_config().get_bvals(twodads::field_t::f_theta), get_config().get_tint_params(twodads::dyn_field_t::f_theta));
            tint_omega = new integrator_karniadakis_bs_t<value_t, allocator_device>(get_config().get_geom(), get_config().get_bvals(twodads::field_t::f_omega), get_config().get_tint_params(twodads::dyn_field_t::f_omega));
            tint_tau = new integrator_karniadakis_bs_t<value_t, allocator_device>(get_config().get_geom(), get_config().get_bvals(twodads::field_t::f_tau), get_config().get_tint_params(twodads::dyn_field_t::f_tau));
#endif //DEVICE
            break;

        case twodads::grid_t::cell_centered:
#ifdef HOST
            my_derivs = new deriv_fd_t<value_t, allocator_host>(get_config().get_geom());
            tint_theta = new integrator_karniadakis_fd_t<value_t, allocator_host>(get_config().get_geom(), get_config().get_bvals(twodads::field_t::f_theta), get_config().get_tint_params(twodads::dyn_field_t::f_theta));
            tint_omega = new integrator_karniadakis_fd_t<value_t, allocator_host>(get_config().get_geom(), get_config().get_bvals(twodads::field_t::f_omega), get_config().get_tint_params(twodads::dyn_field_t::f_omega));
            tint_tau = new integrator_karniadakis_fd_t<value_t, allocator_host>(get_config().get_geom(), get_config().get_bvals(twodads::field_t::f_tau), get_config().get_tint_params(twodads::dyn_field_t::f_tau));
#endif //HOST
#ifdef DEVICE
            my_derivs = new deriv_fd_t<value_t, allocator_device>(get_config().get_geom());
            tint_theta = new integrator_karniadakis_fd_t<value_t, allocator_device>(get_config().get_geom(), get_config().get_bvals(twodads::field_t::f_theta), get_config().get_tint_params(twodads::dyn_field_t::f_theta));
            tint_omega = new integrator_karniadakis_fd_t<value_t, allocator_device>(get_config().get_geom(), get_config().get_bvals(twodads::field_t::f_omega), get_config().get_tint_params(twodads::dyn_field_t::f_omega));
            tint_tau = new integrator_karniadakis_fd_t<value_t, allocator_device>(get_config().get_geom(), get_config().get_bvals(twodads::field_t::f_tau), get_config().get_tint_params(twodads::dyn_field_t::f_tau));
#endif //DEVICE
            break;
    }
 
    try
    {
        conf.check_consistency();    
    }
    catch (config_error e)
    {
        std::cerr << "Error in slab configuration: " << e.what() << std::endl;
    }

    // Set data pointers of diagnostic data member
    // This routine should be constant as to forbid overwriting of the simulation data
    diagnostic.init_field_ptr(twodads::field_t::f_theta, &theta);
    diagnostic.init_field_ptr(twodads::field_t::f_theta_x, &theta_x);
    diagnostic.init_field_ptr(twodads::field_t::f_theta_y, &theta_y);
    diagnostic.init_field_ptr(twodads::field_t::f_omega, &omega);
    diagnostic.init_field_ptr(twodads::field_t::f_omega_x, &omega_x);
    diagnostic.init_field_ptr(twodads::field_t::f_omega_y, &omega_y);
    diagnostic.init_field_ptr(twodads::field_t::f_tau, &tau);
    diagnostic.init_field_ptr(twodads::field_t::f_tau_x, &tau_x);
    diagnostic.init_field_ptr(twodads::field_t::f_tau_y, &tau_y);
    diagnostic.init_field_ptr(twodads::field_t::f_strmf, &strmf);
    diagnostic.init_field_ptr(twodads::field_t::f_strmf_x, &strmf_x);
    diagnostic.init_field_ptr(twodads::field_t::f_strmf_y, &strmf_y);
}


void slab_bc :: dft_r2c(const twodads::field_t fname, const size_t tidx)
{
    arr_real* arr{get_field_by_name.at(fname)};
    assert(((*arr).is_transformed(tidx) == false) && "slab_bc :: dft_r2c: Array is already transformed");

    (*myfft).dft_r2c((*arr).get_tlev_ptr(tidx), reinterpret_cast<twodads::cmplx_t*>((*arr).get_tlev_ptr(tidx)));
    (*arr).set_transformed(tidx, true);
}


void slab_bc :: dft_c2r(const twodads::field_t fname, const size_t tidx)
{
    arr_real* arr{get_field_by_name.at(fname)};
    assert((*arr).is_transformed(tidx) && "slab_bc :: dft_c2r: Array is not transformed");

    (*myfft).dft_c2r(reinterpret_cast<twodads::cmplx_t*>((*arr).get_tlev_ptr(tidx)), (*arr).get_tlev_ptr(tidx));
    utility :: normalize(*arr, tidx);
    (*arr).set_transformed(tidx, false);
}


void slab_bc :: initialize()
{
    // Initialize the fields according to input.json and calculate the derivatives

    // Make tuples dyn_field : (field, field_x, field_y)
    auto tuple_theta = std::make_tuple(twodads::field_t::f_theta, 
                                       twodads::field_t::f_theta_x, 
                                       twodads::field_t::f_theta_y);

    auto tuple_omega = std::make_tuple(twodads::field_t::f_omega, 
                                       twodads::field_t::f_omega_x,
                                       twodads::field_t::f_omega_y);

    auto tuple_tau = std::make_tuple(twodads::field_t::f_tau, 
                                     twodads::field_t::f_tau_x, 
                                     twodads::field_t::f_tau_y);

    // Map a dynamic field name to its real field, x and y derivatives
    auto map_fields = std::map<twodads::dyn_field_t, decltype(tuple_theta)>
        {{twodads::dyn_field_t::f_theta, tuple_theta},
         {twodads::dyn_field_t::f_omega, tuple_omega}, 
         {twodads::dyn_field_t::f_tau, tuple_tau}};

    // Iterate over the dynamic fields and initialize them
    for(auto it : map_fields)
    {
        // The field we are going to initialize
        arr_real* field{get_field_by_name.at(std::get<0>(it.second))};

        // Get the initial conditions from the config file
        std::vector<twodads::real_t> initvals{get_config().get_initc(it.first)};

        // Initializing at the largest time level of the array
        const size_t tidx{get_config().get_tint_params(it.first).get_tlevs() - 1};

        twodads::real_t iv0{0.0};
        twodads::real_t iv1{0.0};
        twodads::real_t iv2{0.0};
        twodads::real_t iv3{0.0};
        twodads::real_t iv4{0.0};

        switch(get_config().get_init_func_t(it.first)) 
        {
        case twodads::init_fun_t::init_constant:
            std::cout << "Initializing constant" << initvals[0] << std::endl;

            assert(initvals.size() == 1 && "Initializing a constant requires at least one initvals");
            iv0 = initvals[0];
            (*field).apply([=] LAMBDACALLER (twodads::real_t input, const size_t n, const size_t m, twodads::slab_layout_t geom) -> twodads::real_t
            {
                return(iv0);
            }, tidx);
            field -> set_transformed(tidx, false);
           break;

        case twodads::init_fun_t::init_gaussian:
            std::cout << "Initializing gaussian: ";
            for(auto it : initvals)
                std::cout << it << "\t";
            std::cout << std::endl;

            // We need 4 initial values
            assert(initvals.size() > 4 && "Initializing a Gaussian requires at least 5 initvals");
            iv0 = initvals[0];
            iv1 = initvals[1];
            iv2 = initvals[2];
            iv3 = initvals[3];
            iv4 = initvals[4];

            (*field).apply([=] LAMBDACALLER (twodads::real_t input, const size_t n, const size_t m, twodads::slab_layout_t geom) -> twodads::real_t
            {
                const twodads::real_t x{geom.get_x(n)};
                const twodads::real_t y{geom.get_y(m)};
                return(iv0 + iv1 * exp(-1.0 * ((x - iv2) * (x - iv2) + 
                                               (y - iv3) * (y - iv3)) / (2.0 * iv4)));                
            }, tidx);
            field -> set_transformed(tidx, false);
            break;

        case twodads::init_fun_t::init_lamb_dipole:
                throw not_implemented_error(std::string("Initializing lamb dipole: not implemented yet"));
            break;

        case twodads::init_fun_t::init_mode:
            std::cout << "Initializing mode" << std::endl;

            (*field).apply([=] LAMBDACALLER (twodads::real_t input, const size_t n, const size_t m, twodads::slab_layout_t geom) -> twodads::real_t
            {
                const twodads::real_t x{geom.get_x(n)};
                //const twodads::real_t y{geom.get_y(m)};
                return(x);
            }, tidx);
            field -> set_transformed(tidx, false);
            break;

        case twodads::init_fun_t::init_sine:
            std::cout << "Initializing sine" << std::endl;
            assert(initvals.size() == 2 && "Initializing a sine requires at least two initial values");
            iv0 = initvals[0];
            iv1 = initvals[1];
            
            (*field).apply([=] LAMBDACALLER (twodads::real_t input, const size_t n, const size_t m, twodads::slab_layout_t geom) -> twodads::real_t
            {
                const twodads::real_t x{geom.get_x(n)};
                const twodads::real_t y{geom.get_y(m)};
                return(sin(iv0 * x) + sin(iv1 * y));
            }, tidx);
            field -> set_transformed(tidx, false);
            break;

        case twodads::init_fun_t::init_turbulent_bath:
                throw not_implemented_error(std::string("Initializing turbulent bath: not implemented yet"));
            break;

        case twodads::init_fun_t::init_NA:
                throw not_implemented_error(std::string("Initialization routine not available"));
            break;
        }

        if(get_config().get_log_theta())
        {
            theta.elementwise([=] LAMBDACALLER(twodads::real_t lhs, twodads::real_t dummy) -> twodads::real_t
                              {return(log(lhs));}, tidx, 0);
        }
        
        if(get_config().get_log_tau())
        {
            tau.elementwise([=] LAMBDACALLER(twodads::real_t lhs, twodads::real_t dummy) -> twodads::real_t
                            {return(log(lhs));}, tidx, 0);
        }
        
        // Transform fields to compute the derivatives
        switch(get_config().get_grid_type())
        {
            case twodads::grid_t::cell_centered:
                // Semi-spectral: Compute x-deriv in real space, y-deriv in fourier space
                d_dx(std::get<0>(it.second), std::get<1>(it.second), 1, tidx, 0);
                dft_r2c(std::get<0>(it.second), tidx);
                d_dy(std::get<0>(it.second), std::get<2>(it.second), 1, tidx, 0); 
                break;

            case twodads::grid_t::vertex_centered:
                // Bispectral: Compute both derivs in real space
                dft_r2c(std::get<0>(it.second), tidx);
                d_dx(std::get<0>(it.second), std::get<1>(it.second), 1, tidx, 0);
                d_dy(std::get<0>(it.second), std::get<2>(it.second), 1, tidx, 0);
                break;                
        }
    }
}


// Compute x-derivative
void slab_bc :: d_dx(const twodads::field_t fname_src, const twodads::field_t fname_dst,
                     const size_t order, const size_t t_src, const size_t t_dst)
{
    arr_real* arr_src{get_field_by_name.at(fname_src)};
    arr_real* arr_dst{get_field_by_name.at(fname_dst)};

    switch(get_config().get_grid_type())
    {
        case twodads::grid_t::cell_centered:
            // Cell-centered grids use finite difference schemes. Nothing to do here.
            break;
        
        case twodads::grid_t::vertex_centered:
            // Vertex-centered grid uses spectral methods. transform
            if(arr_src -> is_transformed(t_src) == false)
                dft_r2c(fname_src, t_src);
            break;
    }

    my_derivs -> dx((*arr_src), (*arr_dst), t_src, t_dst, order);
}


// Compute y-derivative
void slab_bc :: d_dy(const twodads::field_t fname_src, const twodads::field_t fname_dst,
                     const size_t order, const size_t t_src, const size_t t_dst)
{
    arr_real* arr_src = get_field_by_name.at(fname_src);
    arr_real* arr_dst = get_field_by_name.at(fname_dst);

    // Cell-centered grid uses spectral derivation in y. Transform.
    // Vertex-centered grid uses spectral derivation in y. Transform.
    assert(arr_src -> is_transformed(t_src));

    my_derivs -> dy((*arr_src), (*arr_dst), t_src, t_dst, order);
    // Mark output as transformed
    arr_dst -> set_transformed(t_dst, true);
}


// Invert the laplace equation
void slab_bc :: invert_laplace(const twodads::field_t in, const twodads::field_t out, const size_t t_src, const size_t t_dst)
{
    arr_real* in_arr{get_field_by_name.at(in)};
    arr_real* out_arr{get_field_by_name.at(out)};

    assert(in_arr -> is_transformed(t_src));
    
    my_derivs -> invert_laplace((*in_arr), (*out_arr), t_src, t_dst);

    assert(in_arr -> is_transformed(t_src));
    assert(out_arr -> is_transformed(t_dst));
}

// Integrate the field in time
// fname names the field to integrate
// order gives the order of the time integration routine
void slab_bc :: integrate(const twodads::dyn_field_t fname, const size_t order)
{
    const size_t tlevs{get_config().get_tint_params(fname).get_tlevs()};
    assert(order > 0 && order < tlevs);

    arr_real* arr = get_dfield_by_name.at(fname);
    arr_real* arr_rhs{nullptr};

#ifdef HOST
    integrator_base_t<value_t, allocator_host>* tint_ptr{nullptr};
#endif
#ifdef DEVICE
    integrator_base_t<value_t, allocator_device>* tint_ptr{nullptr};
#endif

    switch(fname)
    {
        case twodads::dyn_field_t::f_theta:
            tint_ptr = tint_theta;
            arr_rhs = &theta_rhs;
            break;
        case twodads::dyn_field_t::f_omega:
            tint_ptr = tint_omega;
            arr_rhs = &omega_rhs;
            break;
        case twodads::dyn_field_t::f_tau:
            tint_ptr = tint_tau;
            arr_rhs = &tau_rhs;
            break;
    }

    // The driver code for slab-objects shold ensure that for first order integration
    // arr[tlevs - 1], arr[tlevs - 2], and arr_rhs[tlevs - 2] are real, irrespective of
    // the used grid when entering slab_bc :: integrate.
    // If we have a vertex centered grid they need to be transformed into fourier space.
    // Second and third order integrators need more fields to be transformed. 
    
    if(get_config().get_grid_type() == twodads::grid_t::vertex_centered)
    {
        std::vector<size_t> arr_idx;
        std::vector<size_t> arr_rhs_idx;
        if(order == 1)
        {
            arr_idx.push_back(tlevs - 1);
            (*arr).set_transformed(tlevs - 2, true);
            arr_rhs_idx.push_back(tlevs - 2);
        } 
        else if (order == 2)
        {
            arr_idx.push_back(tlevs - 2);
            (*arr).set_transformed(tlevs - 3, true);
            arr_rhs_idx.push_back(tlevs - 3);
        } 
        else if (order == 3)
        {
            arr_idx.push_back(tlevs - 3);
            (*arr).set_transformed(0, true);
            arr_rhs_idx.push_back(tlevs - 4);
        }

        for(auto tidx : arr_idx)
        {
            assert(arr -> is_transformed(tidx) == false);
            (*myfft).dft_r2c((*arr).get_tlev_ptr(tidx), 
                             reinterpret_cast<twodads::cmplx_t*>((*arr).get_tlev_ptr(tidx)));
            (*arr).set_transformed(tidx, true);
        }
        for(auto tidx : arr_rhs_idx)
        {
            assert(arr_rhs -> is_transformed(tidx) == false);
            (*myfft).dft_r2c((*arr_rhs).get_tlev_ptr(tidx), 
                              reinterpret_cast<twodads::cmplx_t*>((*arr_rhs).get_tlev_ptr(tidx)));
            (*arr_rhs).set_transformed(tidx, true);
        }
    }

    // tint leaves gives the newest time step transformed.
    if(order == 1)
    {
        // second order integration: Source is at tlevs - 1,
        // next time step data is writte to tlevs - 2 
        tint_ptr -> integrate((*arr), (*arr_rhs), tlevs - 1, 0, 0, tlevs - 2, order);
    }
    else if (order == 2)
    {
        // Third order:
        // Sources at tlevs - 1, tlevs - 2
        // Next time step data is written to tlevs - 3
        tint_ptr -> integrate((*arr), (*arr_rhs), tlevs - 2, tlevs - 1, 0, tlevs - 3, order);
    }
    else if (order == 3)
    {
        tint_ptr -> integrate((*arr), (*arr_rhs), tlevs - 3, tlevs - 2, tlevs - 1, tlevs - 4, order);
    }
}


/*
 * Update all real fields. 
 * Input is taken from t_src, or 0 in case of strmf.
 * Output is written back to t_src or 0 respectively.
 */
void slab_bc :: update_real_fields(const size_t t_src)
{
    assert(theta.is_transformed(t_src) == true);
    assert(omega.is_transformed(t_src) == true);
    assert(tau.is_transformed(t_src) == true);
    assert(strmf.is_transformed(0) == true);
    switch(get_config().get_grid_type())
    {
        // Using semi-spectral methods, compute the y derivatives in fourier space
        // and the x derivatives in real space
        case twodads::grid_t::cell_centered:
            d_dy(twodads::field_t::f_theta, twodads::field_t::f_theta_y, 1, t_src, 0);
            d_dy(twodads::field_t::f_omega, twodads::field_t::f_omega_y, 1, t_src, 0);
            d_dy(twodads::field_t::f_tau, twodads::field_t::f_tau_y, 1, t_src, 0);
            d_dy(twodads::field_t::f_strmf, twodads::field_t::f_strmf_y, 1, 0, 0);

            dft_c2r(twodads::field_t::f_theta, t_src);
            dft_c2r(twodads::field_t::f_theta_y, 0);

            dft_c2r(twodads::field_t::f_omega, t_src);
            dft_c2r(twodads::field_t::f_omega_y, 0);

            dft_c2r(twodads::field_t::f_tau, t_src);
            dft_c2r(twodads::field_t::f_tau_y, 0);

            dft_c2r(twodads::field_t::f_strmf, 0);
            dft_c2r(twodads::field_t::f_strmf_y, 0);

            d_dx(twodads::field_t::f_theta, twodads::field_t::f_theta_x, 1, t_src, 0);
            d_dx(twodads::field_t::f_omega, twodads::field_t::f_omega_x, 1, t_src, 0);
            d_dx(twodads::field_t::f_tau, twodads::field_t::f_tau_x, 1, t_src, 0);
            d_dx(twodads::field_t::f_strmf, twodads::field_t::f_strmf_x, 1, 0, 0);

            break;

        //Using bispectral methods we compute the derivative in fourier space
        case twodads::grid_t::vertex_centered:  
            d_dx(twodads::field_t::f_theta, twodads::field_t::f_theta_x, 1, t_src, 0);
            d_dy(twodads::field_t::f_theta, twodads::field_t::f_theta_y, 1, t_src, 0);         
            
            d_dx(twodads::field_t::f_omega, twodads::field_t::f_omega_x, 1, t_src, 0);
            d_dy(twodads::field_t::f_omega, twodads::field_t::f_omega_y, 1, t_src, 0);

            d_dx(twodads::field_t::f_tau, twodads::field_t::f_tau_x, 1, t_src, 0);
            d_dy(twodads::field_t::f_tau, twodads::field_t::f_tau_y, 1, t_src, 0);

            d_dx(twodads::field_t::f_strmf, twodads::field_t::f_strmf_x, 1, 0, 0);
            d_dy(twodads::field_t::f_strmf, twodads::field_t::f_strmf_y, 1, 0, 0);

            dft_c2r(twodads::field_t::f_theta, t_src);
            dft_c2r(twodads::field_t::f_theta_x, 0);
            dft_c2r(twodads::field_t::f_theta_y, 0);

            dft_c2r(twodads::field_t::f_omega, t_src);
            dft_c2r(twodads::field_t::f_omega_x, 0);
            dft_c2r(twodads::field_t::f_omega_y, 0);

            dft_c2r(twodads::field_t::f_tau, t_src);
            dft_c2r(twodads::field_t::f_tau_x, 0);
            dft_c2r(twodads::field_t::f_tau_y, 0);

            dft_c2r(twodads::field_t::f_strmf, 0);
            dft_c2r(twodads::field_t::f_strmf_x, 0);
            dft_c2r(twodads::field_t::f_strmf_y, 0);
            break;
    }

    assert(theta.is_transformed(t_src) == false);
    assert(theta_x.is_transformed(0) == false);
    assert(theta_y.is_transformed(0) == false);

    assert(omega.is_transformed(t_src) == false);
    assert(omega_x.is_transformed(0) == false);
    assert(omega_y.is_transformed(0) == false);

    assert(tau.is_transformed(t_src) == false);
    assert(tau_x.is_transformed(0) == false);
    assert(tau_y.is_transformed(0) == false);

    assert(strmf.is_transformed(0) == false);
    assert(strmf_x.is_transformed(0) == false);
    assert(strmf_y.is_transformed(0) == false);
}


// Advance the fields in time
void slab_bc :: advance()
{
    theta.advance();
    omega.advance();
    tau.advance();

    theta_rhs.advance();
    omega_rhs.advance();
    tau_rhs.advance();
} 


void slab_bc :: write_output(const size_t t_src, const twodads::real_t time)
{
    arr_real* arr{nullptr};
    size_t t_out{0};
    // Iterate over list of fields we want in the HDF file
    for(auto it : get_config().get_output())
    { 
        arr = get_output_by_name.at(it);

        if(it == twodads::output_t::o_theta ||
           it == twodads::output_t::o_omega ||
           it == twodads::output_t::o_tau)
        {
            t_out = t_src;
        }
        else
        {
            t_out = 0;
        }

        assert(t_out < arr -> get_tlevs());
        assert(arr -> is_transformed(t_out) == false);
#ifdef DEVICE
        output.surface(it, utility :: create_host_vector(arr), t_out, time);
#endif
#ifdef HOST
        output.surface(it, arr, t_out, time);
#endif
    }
    output.increment_output_counter();
}

void slab_bc :: diagnose(const size_t t_src, const twodads::real_t time)
{
    // Assert that all fields are real
    assert(get_array_ptr(twodads::field_t::f_theta) -> is_transformed(t_src) == false);
    assert(get_array_ptr(twodads::field_t::f_theta_x) -> is_transformed(0) == false);
    assert(get_array_ptr(twodads::field_t::f_theta_y) -> is_transformed(0) == false);
    assert(get_array_ptr(twodads::field_t::f_omega) -> is_transformed(t_src) == false);
    assert(get_array_ptr(twodads::field_t::f_omega_x) -> is_transformed(0) == false);
    assert(get_array_ptr(twodads::field_t::f_omega_y) -> is_transformed(0) == false);
    assert(get_array_ptr(twodads::field_t::f_tau) -> is_transformed(t_src) == false);
    assert(get_array_ptr(twodads::field_t::f_tau_x) -> is_transformed(0) == false);
    assert(get_array_ptr(twodads::field_t::f_tau_y) -> is_transformed(0) == false);
    assert(get_array_ptr(twodads::field_t::f_strmf) -> is_transformed(0) == false);
    assert(get_array_ptr(twodads::field_t::f_strmf_x) -> is_transformed(0) == false);
    assert(get_array_ptr(twodads::field_t::f_strmf_y) -> is_transformed(0) == false);

    // Treat logarithmic density and temperature fields
    if(get_config().get_log_theta())
    {
        theta.elementwise([=] LAMBDACALLER (twodads::real_t lhs, twodads::real_t dummy) -> twodads::real_t
                          {return(exp(lhs));}, 0, 0);
    }
    if(get_config().get_log_tau())
        tau.elementwise([=] LAMBDACALLER (twodads::real_t lhs, twodads::real_t dummy) -> twodads::real_t
                        {return(exp(lhs));}, 0, 0);
    {
    }

    diagnostic.write_diagnostics(time, get_config());
    if(get_config().get_log_theta())
    {
        theta.elementwise([=] LAMBDACALLER (twodads::real_t lhs, twodads::real_t dummy) -> twodads::real_t
                          {return(log(lhs));}, 0, 0);
    }
    if(get_config().get_log_tau())
    {
        tau.elementwise([=] LAMBDACALLER (twodads::real_t lhs, twodads::real_t dummy) -> twodads::real_t
                          {return(log(lhs));}, 0, 0);
    }

}



void slab_bc :: rhs(const size_t t_dst, const size_t t_src)
{
    std::cout << "slab_bc::rhs: t_dst = " << t_dst << ", t_src = " << t_src << std::endl;
    assert(theta.is_transformed(t_src) == false);
    assert(theta_x.is_transformed(0) == false);
    assert(theta_y.is_transformed(0) == false);

    assert(omega.is_transformed(t_src) == false);
    assert(omega_x.is_transformed(0) == false);
    assert(omega_y.is_transformed(0) == false);

    assert(strmf.is_transformed(0) == false);
    assert(strmf_x.is_transformed(0) == false);
    assert(strmf_y.is_transformed(0) == false);

    assert(tau.is_transformed(t_src) == false);
    assert(tau_x.is_transformed(0) == false);
    assert(tau_y.is_transformed(0) == false);            

    (this ->* theta_rhs_func)(t_dst, t_src);
    (this ->* omega_rhs_func)(t_dst, t_src);
    (this ->* tau_rhs_func)(t_dst, t_src);

    assert(theta_rhs.is_transformed(t_dst) == false);
    assert(omega_rhs.is_transformed(t_dst) == false);
    assert(tau_rhs.is_transformed(t_dst) == false);
}


inline void slab_bc :: rhs_theta_null(const size_t t_dst, const size_t t_src)
{
}


void slab_bc :: rhs_theta_lin(const size_t t_dst, const size_t t_src)
{
    // Compute poisson bracket, {theta, phi}
    // theta_rhs <- {phi, theta}
    switch(get_config().get_grid_type())
    {
        case twodads::grid_t::vertex_centered:
            // Derivative fields have only 1 time index. Do not use t_src here.
            // Store in t_dst time index of RHS
            // Store theta_x strmf_y - theta_y strmf_x in theta_rhs
            my_derivs -> pbracket(theta_x, theta_y, strmf_x, strmf_y, theta_rhs, 0, 0, t_dst);
            break;

        case twodads::grid_t::cell_centered:
            // Input to Arakawa scheme is from in-place DFTs of dynamic fields.
            // Get data from t_src.
            // Store in t_dst time index of RHS
            //my_derivs -> pbracket(theta, strmf, theta_rhs, t_src, 0, t_dst);
            my_derivs -> pbracket(strmf, theta, theta_rhs, 0, t_src, t_dst);
            break;
    }
}


inline void slab_bc :: rhs_omega_null(const size_t t_dst, const size_t t_src)
{
    // Do nothing
}


void slab_bc :: rhs_omega_ic(const size_t t_dst, const size_t t_src)
{
    const std::vector<twodads::real_t> model_params{conf.get_model_params(twodads::dyn_field_t::f_omega)};
    const twodads::real_t ic{model_params[1]};

    // Compute poisson bracket
    // omega_rhs <- {phi, omega}
    switch(get_config().get_grid_type())
    {
        case twodads::grid_t::vertex_centered:
            // Store in t_dst time index of RHS
            // omega_rhs <- omega_x strmf_y - omega_y strmf_x
            my_derivs -> pbracket(omega_x, omega_y, strmf_x, strmf_y, omega_rhs, 0, 0, t_dst);
            break;

        case twodads::grid_t::cell_centered:
            // Store in t_dst time index of RHS
            //my_derivs -> pbracket(omega, strmf, omega_rhs, t_src, 0, t_dst);
            my_derivs -> pbracket(strmf, omega, omega_rhs, 0, t_src, t_dst);
            break;
    }
    

    // omega_rhs <- {omega, phi} - ic * theta_y
    omega_rhs.elementwise([=] LAMBDACALLER (twodads::real_t lhs, twodads::real_t rhs) -> twodads::real_t 
                          {return (lhs - ic * rhs);}, theta_y, 0, t_dst);           
}


inline void slab_bc :: rhs_tau_null(const size_t t_dst, const size_t t_src)
{
    // Do nothing
}



slab_bc :: ~slab_bc()
{
    delete tint_tau;
    delete tint_omega;
    delete tint_theta;
    delete my_derivs;
    delete myfft;
}

// End of file slab_bc.cpp
