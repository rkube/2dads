#include "include/slab_cuda.h"
#include "include/initialize.h"

slab_cuda :: slab_cuda(slab_config my_config) :
    config(my_config),
    Nx(my_config.get_nx()),
    My(my_config.get_my()),
    tlevs(my_config.get_tlevs()),
    theta(1, Nx, My),
    theta_hat(tlevs, Nx, My / 2 + 1),
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
        // Rest is not implemented yet...
        default:
            throw;
    }
}


cuda_array<cuda::cmplx_t>* slab_cuda :: get_field_by_name(twodads::field_k_t field)
{
    switch(field)
    {
        case twodads::field_k_t::f_theta_hat:
            return &theta_hat;
            break;
        // Rest is not implmeneted yet...
        default:
            throw;
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


void slab_cuda :: advance()
{
    cout << "slab_cuda :: advance()\n";
    theta_hat.advance();
    //omega_hat.advance();
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

void slab_cuda :: integrate_stiff(twodads::dyn_field_t field, uint tlev)
{
    cout << "slab_cuda :: integrate_stiff()\n";
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
            //init_gaussian(&theta, config.get_initc(), config.get_deltax(),
            //        config.get_deltay(), config.get_xleft(), config.get_ylow());
            break;
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
        case twodads::init_fun_t::init_theta_mode:
            cout << "init_theta_mode\n";
            break;
        case twodads::init_fun_t::init_theta_const:
            cout << "init_theta_const\n";
            break;
        case twodads::init_fun_t::init_file:
            cout << "init_file\n";
            break;
    }
}

