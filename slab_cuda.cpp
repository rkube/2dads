#include "include/slab_cuda.h"
#include "include/initialize.h"

slab_cuda :: slab_cuda(slab_config my_config) :
    config(my_config),
    Nx(my_config.get_nx()),
    My(my_config.get_my()),
    tlevs(my_config.get_tlevs()),
    theta(1, Nx, My), theta_x(1, Nx, My), theta_y(1, Nx, My),
    tmp_array(1, Nx, My), 
    theta_hat(tlevs, Nx, My / 2 + 1), theta_x_hat(1, Nx, My / 2 + 1), theta_y_hat(1, Nx, My / 2 + 1),
    tmp_array_hat(1, Nx, My / 2 + 1), 
    theta_rhs_hat(tlevs - 1, Nx, My / 2 + 1),
    dft_is_initialized(init_dft())
{
    cout << "slab_cuda :: slab_cuda(config my_config)\n";

    // Setting RHS pointer
    switch(config.get_theta_rhs_type())
    {
        case twodads::rhs_t::theta_rhs_lin:
            theta_rhs_fun = &slab_cuda::theta_rhs_lin;
            break;
        case twodads::rhs_t::theta_rhs_log:
            theta_rhs_fun = &slab_cuda::theta_rhs_log;
            break;
        case twodads::rhs_t::theta_rhs_hw:
            // Not implemented yet
            break;
        case twodads::rhs_t::theta_rhs_hwmod:
            // Not implemented yet
            break;
        case twodads::rhs_t::theta_rhs_ic:
            //Not implemented yet
            break;
        case twodads::rhs_t::rhs_null:
            theta_rhs_fun = &slab_cuda::rhs_null;
            break;
    }
    // Setup auxiliary structures
    stiff_params_theta.delta_t = config.get_deltat();
    stiff_params_theta.length_x = config.get_lengthx();
    stiff_params_theta.length_y = config.get_lengthy();
    stiff_params_theta.diff = config.get_model_param(0);
    stiff_params_theta.S = 4.0 * twodads::PI * twodads::PI * config.get_model_param(0) * config.get_deltat();
    stiff_params_theta.level = config.get_tlevs();
    cout << "stiff_params_theta.delta_t = " << stiff_params_theta.delta_t << "\n"; 
    cout << "stiff_params_theta.length_x = " << stiff_params_theta.length_x << "\n"; 
    cout << "stiff_params_theta.length_y = " << stiff_params_theta.length_y << "\n"; 
    cout << "stiff_params_theta.diff = " << stiff_params_theta.diff << "\n"; 
    cout << "stiff_params_theta.S = " << stiff_params_theta.S << "\n"; 
    cout << "stiff_params_theta.level = " << stiff_params_theta.level << "\n"; 

    stiff_params_omega.delta_t = config.get_deltat();
    stiff_params_omega.length_x = config.get_lengthx();
    stiff_params_omega.length_y = config.get_lengthy();
    stiff_params_omega.diff = config.get_model_param(1);
    stiff_params_omega.S = 4.0 * twodads::PI * twodads::PI * config.get_model_param(1) * config.get_deltat();
    stiff_params_omega.level = config.get_tlevs();
    cout << "stiff_params_omega.delta_t = " << stiff_params_omega.delta_t << "\n"; 
    cout << "stiff_params_omega.length_x = " << stiff_params_omega.length_x << "\n"; 
    cout << "stiff_params_omega.length_y = " << stiff_params_omega.length_y << "\n"; 
    cout << "stiff_params_omega.diff = " << stiff_params_omega.diff << "\n"; 
    cout << "stiff_params_omega.S = " << stiff_params_omega.S << "\n"; 
    cout << "stiff_params_omega.level = " << stiff_params_omega.level << "\n"; 

}


slab_cuda :: ~slab_cuda()
{
    cout << "slab_cuda :: ~slab_cuda()\n";
    finish_dft();
}

// convert field type to internal pointer
cuda_array<cuda::real_t>* slab_cuda :: get_field_by_name(twodads::field_t field)
{
    switch(field)
    {
        case twodads::field_t::f_theta:
            return &theta;
            break;
        case twodads::field_t::f_theta_x:
            return &theta_x;
            break;
        case twodads::field_t::f_theta_y:
            return &theta_y;
            break;
        /*
        case twodads::field_t::f_omega:
            return &omega;
            break;
        case twodads::field_t::f_omega_x:
            return &omega_x;
            break;
        case twodads::field_t::f_omega_y:
            return &omega_y;
            break;
        case twodads::field_t::f_strmf:
            return &strmf;
            break;
        case twodads::field_t::f_strmf_x:
            return &strmf_x;
            break;
        case twodads::field_t::f_strmf_y:
            return &strmf_y;
            break;
        */
        default:
            stringstream err_str;
            err_str << "Invalid field name\n";
            throw name_error(err_str.str());
    }
}


cuda_array<cuda::cmplx_t>* slab_cuda :: get_field_by_name(twodads::field_k_t field)
{
    switch(field)
    {
        case twodads::field_k_t::f_theta_hat:
            return &theta_hat;
            break;
        case twodads::field_k_t::f_theta_x_hat:
            return &theta_x_hat;
            break;
        case twodads::field_k_t::f_theta_y_hat:
            return &theta_y_hat;
            break;
        /*
        case twodads::field_k_t::f_omega_hat:
            return &omega_hat;
            break;
        case twodads::field_k_t::f_omega_x_hat:
            return &omega_x_hat;
            break;
        case twodads::field_k_t::f_omega_y_hat:
            return &omega_y_hat;
            break;
        
        case twodads::field_k_t::f_strmf_hat:
            return &strmf_hat;
            break;
        case twodads::field_k_t::f_strmf_x_hat:
            return &strmf_x_hat;
            break;
        case twodads::field_k_t::f_strmf_y_hat:
            return &strmf_y_hat;
            break;
        case twodads::field_k_t::f_omega_rhs_hat:
            return &omega_rhs_hat;
            break;
        */
        case twodads::field_k_t::f_theta_rhs_hat:
            return &theta_rhs_hat;
            break;
        case twodads::field_k_t::f_tmp_hat:
            return &tmp_array_hat;
            break;
        default:
            stringstream err_str;
            err_str << "Invalid field name\n";
            throw name_error(err_str.str());
    }
}



bool slab_cuda :: init_dft()
{
    cout << "slab_cuda :: init_dft()\n";
    cufftResult err;
    err = cufftPlan2d(&plan_r2c, Nx, My, CUFFT_D2Z);
    if (err != 0)
    {
        stringstream err_str;
        err_str << "Error planning D2Z DFT: " << err << "\n";
        throw gpu_error(err_str.str());
    }

    err = cufftPlan2d(&plan_c2r, Nx, My, CUFFT_Z2D);
    if (err != 0)
    {
        stringstream err_str;
        err_str << "Error planning D2Z DFT: " << err << "\n";
        throw gpu_error(err_str.str());
    }

    return(true);
}

void slab_cuda :: finish_dft()
{
    cout << "slab_cuda :: finish_dft()\n";
    cufftDestroy(plan_r2c);
    cufftDestroy(plan_c2r);
}

void slab_cuda :: test_slab_config()
{
    cout << "slab_cuda :: test_slab_config()\n";
}


void slab_cuda :: advance(twodads::field_k_t fname)
{
    cuda_array<cuda::cmplx_t>* arr = get_field_by_name(fname);
    //theta_hat.advance();
    //omega_hat.advance();
    arr -> advance();
}


void slab_cuda :: theta_rhs_lin()
{
    cout << "Not implemented yet: slab_cuda :: theta_rhs_lin()\n";
}


void slab_cuda :: theta_rhs_log()
{
    cout << "Not implemented yet: slab_cuda :: theta_rhs_log()\n";
}


void slab_cuda :: rhs_null()
{
    cout << "Not implemented yet: slab_cuda :: rhs_null()\n";
}


void slab_cuda :: dft_r2c(twodads::field_t fname_r, twodads::field_k_t fname_c, uint t)
{
    cufftResult err;
    cout << "dft_r2c\n";
    cuda_array<cuda::real_t>* arr_r = get_field_by_name(fname_r);
    cuda_array<cuda::cmplx_t>* arr_c = get_field_by_name(fname_c);
    err = cufftExecD2Z(plan_r2c, arr_r -> get_array_d(), arr_c -> get_array_d(t));
    if (err != CUFFT_SUCCESS)
        throw;
}


void slab_cuda :: dft_c2r(twodads::field_k_t fname_c, twodads::field_t fname_r, uint t)
{
    cufftResult err;
    cuda_array<cuda::cmplx_t>* arr_c= get_field_by_name(fname_c);
    cuda_array<cuda::real_t>* arr_r = get_field_by_name(fname_r);
    err = cufftExecZ2D(plan_c2r, arr_c -> get_array_d(t), arr_r -> get_array_d());
    if (err != CUFFT_SUCCESS)
        throw;
    // Normalize
    arr_r -> normalize();
}


void slab_cuda :: dump_field(twodads::field_t field_name)
{
    cuda_array<cuda::real_t>* field = get_field_by_name(field_name);
    cout << *field << "\n";
}


void slab_cuda :: dump_field(twodads::field_k_t field_name)
{
    cuda_array<cuda::cmplx_t>* field = get_field_by_name(field_name);
    cout << *field << "\n";
}


void slab_cuda :: move_t(twodads::field_t fname, uint t_dst, uint t_src)
{
    cuda_array<cuda::real_t>* arr = get_field_by_name(fname);
    arr -> move(t_dst, t_src);
}


void slab_cuda :: move_t(twodads::field_k_t fname, uint t_dst, uint t_src)
{
    cuda_array<cuda::cmplx_t>* arr = get_field_by_name(fname);
    arr -> move(t_dst, t_src);
}


void slab_cuda :: initialize()
{
    cout << "slab_cuda :: initialize()\n";
    switch(config.get_init_function())
    {
        case twodads::init_fun_t::init_NA:
            cout << "Initializing not available\n";
            break;
        case twodads::init_fun_t::init_theta_gaussian:
            cout << "Initizlizing theta gaussian\n";
            init_gaussian(&theta, config.get_initc(), config.get_deltax(),
                    config.get_deltay(), config.get_xleft(), config.get_ylow());
            dft_r2c(twodads::field_t::f_theta, twodads::field_k_t::f_theta_hat, config.get_tlevs() - 1);
            break;
        /*
        case twodads::init_fun_t::init_omega_random_k:
            cout << "init_omega_random_k\n";
            break;
        case twodads::init_fun_t::init_omega_random_k2:
            cout << "init_omega_random_k2\n";
            break;
        case twodads::init_fun_t::init_both_random_k:
            cout << "init_both_random_k\n";
            break;
        case twodads::init_fun_t::init_omega_exp_k:
            cout << "init_omega_expk\n";
            break;
        case twodads::init_fun_t::init_omega_dlayer:
            cout << "init_omega_dlayer\n";
            break;
        case twodads::init_fun_t::init_const_k:
            cout << "init_const_k\n";
            break;
        case twodads::init_fun_t::init_theta_mode:
            cout << "init_theta_mode\n";
            break;
        case twodads::init_fun_t::init_theta_const:
            cout << "init_theta_const\n";
            break;
        */
        case twodads::init_fun_t::init_simple_sine:
            cout << "init_simple_sine\n";
            init_simple_sine(&theta, config.get_initc(), config.get_deltax(),
                    config.get_deltay(), config.get_xleft(), config.get_ylow());
            // Initialize omega, strmf with const. But this is done in the
            // cuda_array constructor

            // DFT of initial condition
            dft_r2c(twodads::field_t::f_theta, twodads::field_k_t::f_theta_hat, config.get_tlevs() - 1);
            //omega_hat.set_all(0.0);
            //strmf_hat.set_all(0.0);
            break;
        case twodads::init_fun_t::init_test:
            cout << "init_test\n";
            init_invlapl(&theta, config.get_initc(), config.get_deltax(),
                    config.get_deltay(), config.get_xleft(), config.get_ylow());

            dft_r2c(twodads::field_t::f_theta, twodads::field_k_t::f_theta_hat, config.get_tlevs() - 1);
            //omega_hat.set_all(0.0);
            //strmf_hat.set_all(0.0);
        case twodads::init_fun_t::init_file:
            cout << "init_file\n";
            break;
    }
}

// end of file slab_cuda.cpp
