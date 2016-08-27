/*
 * Implementation of slab_bc
 */

#include "slab_bc.h"

using namespace std;

map<twodads::rhs_t, slab_bc::rhs_func_ptr> slab_bc :: rhs_func_map = slab_bc::create_rhs_func_map();

//slab_bc :: slab_bc(const twodads::slab_layout_t _sl, const twodads::bvals_t<twodads::real_t> _bc, const twodads::stiff_params_t _sp) :

slab_bc :: slab_bc(const slab_config_js& _conf) :
    conf(_conf),
    myfft{new dft_t(get_geom(), twodads::dft_t::dft_1d)},
    my_derivs{new deriv_t(get_geom())},
    tint_theta{new integrator_t(get_config().get_geom(), get_config().get_bvals(twodads::field_t::f_theta), get_config().get_stiff_params(twodads::dyn_field_t::f_theta))},
    tint_omega{new integrator_t(get_config().get_geom(), get_config().get_bvals(twodads::field_t::f_omega), get_config().get_stiff_params(twodads::dyn_field_t::f_omega))},
    tint_tau{new integrator_t(get_config().get_geom(), get_config().get_bvals(twodads::field_t::f_tau), get_config().get_stiff_params(twodads::dyn_field_t::f_tau))},
    theta(get_config.get_geom(),       get_config().get_bvals(twodads::field_t::f_theta), get_config().get_tlevs()), 
    theta_x(get_config().get_geom(),   get_config().get_bvals(twodads::field_t::f_theta), 1), 
    theta_y(get_config().get_geom(),   get_config().get_bvals(twodads::field_t::f_theta), 1),
    omega(get_config().get_geom(),     get_config().get_bvals(twodads::field_t::f_omega), get_config().get_tlevs()), 
    omega_x(get_config().get_geom(),   get_config().get_bvals(twodads::field_t::f_omega), 1), 
    omega_y(get_config().get_geom(),   get_config().get_bvals(twodads::field_t::f_omega), 1),
    tau(get_config().get_geom(),       get_config().get_bvals(twodads::field_t::f_tau), get_config().get_tlevs()),   
    tau_x(get_config().get_geom(),     get_config().get_bvals(twodads::field_t::f_tau), 1),   
    tau_y(get_config().get_geom(),     get_config().get_bvals(twodads::field_t::f_tau), 1),
    strmf(get_config().get_geom(),     get_config().get_bvals(twodads::field_t::f_strmf), get_config().get_tlevs()), 
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
                       {twodads::field_t::f_omega_rhs, &omega_rhs}},
    theta_rhs_func{rhs_func_map[twodads::rhs_t::theta_rhs_null]},
    omega_rhs_func{rhs_func_map[twodads::rhs_t::omega_rhs_null]},
    tau_rhs_func{rhs_func_map[twodads::rhs_t::tau_rhs_null]}
{
    std::cout << "Created new slab:" << std::endl;
    std::cout << "Boundaries for theta: " << get_config().get_bvals(twodads::field_t::f_theta) << std::endl;
    std::cout << "Boundaries for omega: " << get_config().get_bvals(twodads::field_t::f_omega) << std::endl;
    std::cout << "Boundaries for tau: " << get_config().get_bvals(twodads::field_t::f_tau) << std::endl;
    std::cout << "Boundaries for strmf: " << get_config().get_bvals(twodads::field_t::f_strmf << std::endl;
}


void slab_bc :: dft_r2c(const twodads::field_t fname, const size_t tidx)
{
    cuda_arr_real* arr{get_field_by_name.at(fname)};

    if(!((*arr).is_transformed(tidx)))
    {
        (*myfft).dft_r2c((*arr).get_tlev_ptr(tidx), reinterpret_cast<twodads::cmplx_t*>((*arr).get_tlev_ptr(tidx)));
        (*arr).set_transformed(tidx, true);
    }
    else
    {
        std::cerr << "Array is already transformed, skipping dft r2c" << std::endl;
    }
}


void slab_bc :: dft_c2r(const twodads::field_t fname, const size_t tidx)
{
    cuda_arr_real* arr{get_field_by_name.at(fname)};

    if((*arr).is_transformed(tidx))
    {
        (*myfft).dft_c2r(reinterpret_cast<twodads::cmplx_t*>((*arr).get_tlev_ptr(tidx)), (*arr).get_tlev_ptr(tidx));
        utility :: normalize(*arr, tidx);
        (*arr).set_transformed(tidx, false);
    }
    else
    {
        std::cerr << "Array is not transformed, skipping dft c2r" << std::endl;
    }
}

void slab_bc :: initialize_invlaplace(const twodads::field_t fname, const size_t tidx)
{
    cuda_arr_real* arr = get_field_by_name.at(fname);
    
    (*arr).apply([] LAMBDACALLER (twodads::real_t dummy, const size_t n, const size_t m, twodads::slab_layout_t geom) -> twodads::real_t
                {
                    const value_t x{geom.get_x(n)};
                    const value_t y{geom.get_y(m)};
                    return(exp(-0.5 * (x * x + y * y)) * (-2.0 + x * x + y * y));
                }, tidx);
}


void slab_bc :: initialize_sine(const twodads::field_t fname, const size_t tidx)
{
    cuda_arr_real* arr = get_field_by_name.at(fname);
    (*arr).apply([] LAMBDACALLER (twodads::real_t dummy, size_t n, size_t m, twodads::slab_layout_t geom) -> value_t
                 {
                    return(sin(twodads::TWOPI * geom.get_x(n)) + 0.0 * sin(twodads::TWOPI * geom.get_y(m)));
                 }, tidx);
}


/*
void slab_bc :: initialize_arakawa(const twodads::field_t fname1, const twodads::field_t fname2, const size_t tlev)
{
    cuda_arr_real* arr1 = get_field_by_name.at(fname1);
    cuda_arr_real* arr2 = get_field_by_name.at(fname2);
  
    // arr1 =-sin^2(2 pi y) sin^2(2 pi x). Dirichlet bc, f(-1, y) = f(1, y) = 0.0
    (*arr1).apply([] LAMBDACALLER (twodads::real_t dummy, size_t n, size_t m, twodads::slab_layout_t geom) -> value_t
                  {
                    value_t x{geom.get_x(n)};
                    value_t y{geom.get_y(m)};
                    return(-1.0 * sin(twodads::TWOPI * y) * sin(twodads::TWOPI * y) * sin(twodads::TWOPI * x) * sin(twodads::TWOPI * x));
                  }, tlev);

    // arr2 = sin(pi x) sin (pi y). Dirichlet BC: g(-1, y) = g(1, y) = 0.0
    (*arr2).apply([] LAMBDACALLER (twodads::real_t dummy, size_t n, size_t m, twodads::slab_layout_t geom) -> value_t
                  {
                    return(sin(twodads::PI * geom.get_x(n)) * sin(twodads::PI * geom.get_y(m)));
                  }, tlev);
}
*/

void slab_bc :: initialize_derivativesx(const twodads::field_t fname, const size_t tidx)
{
    cuda_arr_real* arr = get_field_by_name.at(fname);

    (*arr).apply([] LAMBDACALLER (twodads::real_t dummy, size_t n, size_t m, twodads::slab_layout_t geom) -> value_t
            {       
                return(sin(twodads::TWOPI * geom.get_x(n)));
            }, tidx);
}

void slab_bc :: initialize_derivativesy(const twodads::field_t fname, const size_t tidx)
{
    cuda_arr_real* arr = get_field_by_name.at(fname);
    (*arr).apply([] LAMBDACALLER (twodads::real_t dummy, size_t n, size_t m, twodads::slab_layout_t geom) -> value_t
            {       
                value_t y{geom.get_y(m)};
                return(exp(-50.0 * y * y)); 
            }, tidx);
}


void slab_bc :: initialize_dfttest(const twodads::field_t fname, const size_t tidx)
{
    cuda_arr_real* arr = get_field_by_name.at(fname);
    (*arr).apply([] LAMBDACALLER (twodads::real_t dummy, size_t n, size_t m, twodads::slab_layout_t geom) -> value_t
           {       
               return(sin(twodads::TWOPI * geom.get_y(m)));
           }, tidx);
}


void slab_bc :: initialize_gaussian(const twodads::field_t fname, const size_t tidx)
{
    cuda_arr_real* arr = get_field_by_name.at(fname);
    (*arr).apply([] LAMBDACALLER (twodads::real_t dummy, size_t n, size_t m, twodads::slab_layout_t geom) -> value_t
            {
                const twodads::real_t x{geom.get_x(n)};
                const twodads::real_t y{geom.get_y(m)};
                return(exp(-0.5 * (x * x + y * y)));
            }, tidx);
}


// Compute x-derivative
void slab_bc :: d_dx(const twodads::field_t fname_src, const twodads::field_t fname_dst,
                     const size_t d, const size_t t_src, const size_t t_dst)
{
    cuda_arr_real* arr_src{get_field_by_name.at(fname_src)};
    cuda_arr_real* arr_dst{get_field_by_name.at(fname_dst)};

    if(d == 1)
    {
        my_derivs -> dx_1((*arr_src), (*arr_dst), t_src, t_dst);   
    }
    else if (d == 2)
    {
        my_derivs -> dx_2((*arr_src), (*arr_dst), t_src, t_dst);
    }
}


// Compute y-derivative
void slab_bc :: d_dy(const twodads::field_t fname_src, const twodads::field_t fname_dst,
                     const size_t d, const size_t t_src, const size_t t_dst)
{
    cuda_arr_real* arr_src = get_field_by_name.at(fname_src);
    cuda_arr_real* arr_dst = get_field_by_name.at(fname_dst);

    if(d == 1)
    {
        my_derivs->dy_1((*arr_src), (*arr_dst), t_src, t_dst);   
    }
    else if (d == 2)
    {
        my_derivs->dy_2((*arr_src), (*arr_dst), t_src, t_dst);
    }
}


// Compute Arakawa brackets
// res = {f, g} = f_y g_x - g_y f_x
// Input:
// fname_arr_f: array where f is stored
// fname_arr_g: array where g is stored
// fname_arr_res: array where we store the result
// t_src: time level at which f and g are taken
// t_dst: time level where we store the result in res
void slab_bc :: arakawa(const twodads::field_t fname_arr_f, const twodads::field_t fname_arr_g, const twodads::field_t fname_arr_res,
                        const size_t t_src, const size_t t_dst)
{
    cuda_arr_real* f_arr = get_field_by_name.at(fname_arr_f);
    cuda_arr_real* g_arr = get_field_by_name.at(fname_arr_g);
    cuda_arr_real* res_arr = get_field_by_name.at(fname_arr_res);

    my_derivs -> arakawa((*f_arr), (*g_arr), (*res_arr), t_src, t_dst);
}


// Invert the laplace equation
void slab_bc :: invert_laplace(const twodads::field_t in, const twodads::field_t out, const size_t t_src, const size_t t_dst)
{
    cuda_arr_real* in_arr{get_field_by_name.at(in)};
    cuda_arr_real* out_arr{get_field_by_name.at(out)};
    my_derivs -> invert_laplace((*in_arr), (*out_arr), t_src, t_dst);
}

// Integrate the field in time
// t_dst is the time level where the result is stored
// t_src is the source
// tlev is the time level of the integration
void slab_bc :: integrate(const twodads::field_t fname, const size_t tlev)
{
    const size_t tlevs{get_tint_params().get_tlevs()};
    assert(tlev > 0 && tlev < tlevs);
    cuda_arr_real* arr = get_field_by_name.at(fname);

    for(size_t t  = 0; t < arr -> get_tlevs(); t++)
        assert(arr -> is_transformed(t) == false);

    // Pass real arrays to the time integration routine
    // tint leaves gives the newest time step transformed.
    if(tlev == 1)
    {
        // second order integration: Source is at tlevs - 1,
        // next time step data is writte to tlevs - 2 
        tint -> integrate((*arr), tlevs - 1, 0, 0, tlevs - 2, tlev);

    }
    else if (tlev == 2)
    {
        // Third order:
        // Sources at tlevs - 1, tlevs - 2
        // Next time step data is written to tlevs - 3
        tint -> integrate((*arr), tlevs - 2, tlevs - 1, 0, tlevs - 3, tlev);
    }
    else if (tlev == 3)
    {
        tint -> integrate((*arr), tlevs - 3, tlevs - 2, tlevs - 1, tlevs - 4, tlev);
    }
}


void slab_bc :: update_real_fields(const size_t tlev)
{
    d_dx(twodads::field_t::f_theta, twodads::field_t::f_theta_x, 1, tlev, 0);
    d_dy(twodads::field_t::f_theta, twodads::field_t::f_theta_y, 1, tlev, 0);

    d_dx(twodads::field_t::f_omega, twodads::field_t::f_omega_x, 1, tlev, 0);
    d_dy(twodads::field_t::f_omega, twodads::field_t::f_omega_y, 1, tlev, 0);

    d_dx(twodads::field_t::f_tau, twodads::field_t::f_tau_x, 1, tlev, 0);
    d_dy(twodads::field_t::f_tau, twodads::field_t::f_tau_y, 1, tlev, 0);

    d_dx(twodads::field_t::f_strmf, twodads::field_t::f_strmf_x, 1, 0, 0);
    d_dy(twodads::field_t::f_strmf, twodads::field_t::f_strmf_y, 1, 0, 0);
}


void slab_bc :: rhs(const size_t tsrc)
{
    (this ->* theta_rhs_func)(tsrc);
    (this ->* omega_rhs_func)(tsrc);
    (this ->* tau_rhs_func)(tsrc);
}

// Advance the fields in time
void slab_bc :: advance()
{
    theta.advance();
    omega.advance();
    tau.advance();
} 


slab_bc :: ~slab_bc()
{
    delete myfft;
}

// End of file slab_bc.cu