#include <iostream>
#include "slab_cuda.h"

using namespace std;

int main(void)
{
    slab_config my_config;
    my_config.consistency();

    slab_cuda slab(my_config);

    twodads::real_t time(0.0);
    twodads::real_t delta_t(my_config.get_deltat());

    const unsigned int tout_full(ceil(my_config.get_tout() / my_config.get_deltat()));
    const unsigned int tout_diag(ceil(my_config.get_tdiag() / my_config.get_deltat()));
    const unsigned int num_tsteps(ceil(my_config.get_tend() / my_config.get_deltat()));
    const unsigned int tlevs = my_config.get_tlevs();
    unsigned int t = 1;

    slab.init_dft();
    slab.initialize();
    slab.write_output(time);

    
    cout << "Output every " << tout_full << " steps\n";

    // Integrate the first two steps with a lower order scheme
    for(t = 1; t < tlevs - 1; t++)
    {
        cout << "========================================t = " << t << "========================================\n";
        slab.integrate_stiff(twodads::dyn_field_t::d_theta, t + 1);
        slab.integrate_stiff(twodads::dyn_field_t::d_omega, t + 1);
        time += delta_t;
        slab.inv_laplace(twodads::field_k_t::f_omega_hat, twodads::field_k_t::f_strmf_hat, my_config.get_tlevs() - t - 1);
        slab.update_real_fields(my_config.get_tlevs() - t - 1);
        //slab.dft_c2r(twodads::field_k_t::f_theta_hat, twodads::field_t::f_theta, my_config.get_tlevs() - t - 1);
        //slab.dft_c2r(twodads::field_k_t::f_omega_hat, twodads::field_t::f_omega, my_config.get_tlevs() - t - 1);
        //slab.dft_c2r(twodads::field_k_t::f_strmf_hat, twodads::field_t::f_strmf, 0);
        slab.rhs_fun();

        slab.write_output(time);
        slab.move_t(twodads::field_k_t::f_theta_rhs_hat, my_config.get_tlevs() - t - 1, 0);
        slab.move_t(twodads::field_k_t::f_omega_rhs_hat, my_config.get_tlevs() - t - 1, 0);
        //slab.dump_field(twodads::f_theta_hat);
    }

    for(; t < num_tsteps + 1; t++)
    {
        cout << t << "/" << num_tsteps << "\n";
        slab.integrate_stiff(twodads::dyn_field_t::d_theta, tlevs);
        slab.integrate_stiff(twodads::dyn_field_t::d_omega, tlevs);
        time += delta_t;
        slab.inv_laplace(twodads::field_k_t::f_omega_hat, twodads::field_k_t::f_strmf_hat, 0);
        slab.update_real_fields(0);
        slab.rhs_fun();

        slab.advance();
        if(t % tout_full == 0)
            slab.write_output(time);
        if(t % tout_diag == 0)
            slab.write_diagnostics(time);
        
    }
    return(0);
}


