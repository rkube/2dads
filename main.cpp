#include <iostream>
#include "include/slab_cuda.h"
#include "include/diagnostics.h"
extern template class diag_array<double>;


using namespace std;

int main(void)
{
    slab_config my_config;
    my_config.consistency();

    slab_cuda slab(my_config);
    diagnostics slab_diag(my_config);

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
        slab.integrate_stiff(twodads::dyn_field_t::d_theta, t + 1);
        slab.integrate_stiff(twodads::dyn_field_t::d_omega, t + 1);
        time += delta_t;
        slab.inv_laplace(twodads::field_k_t::f_omega_hat, twodads::field_k_t::f_strmf_hat, my_config.get_tlevs() - t - 1);
        slab.update_real_fields(my_config.get_tlevs() - t - 1);
        slab.rhs_fun(my_config.get_tlevs() - t - 1);

        if ( t == 1)
        {
            slab.move_t(twodads::field_k_t::f_theta_rhs_hat, my_config.get_tlevs() - t - 2, 0);
            slab.move_t(twodads::field_k_t::f_omega_rhs_hat, my_config.get_tlevs() - t - 2, 0);
        }
    }

    for(; t < num_tsteps + 1; t++)
    {
        slab.integrate_stiff(twodads::dyn_field_t::d_theta, tlevs);
        slab.integrate_stiff(twodads::dyn_field_t::d_omega, tlevs);
        slab.advance();
        time += delta_t;
        // Compute all fields from theta_hat(1,:), omega_hat(1,:) since we called advance
        slab.inv_laplace(twodads::field_k_t::f_omega_hat, twodads::field_k_t::f_strmf_hat, 1);
        slab.update_real_fields(1);
        // Result of RHS_fun is put in theta_rhs_hat(0,:)
        slab.rhs_fun(1);

        if(t % tout_full == 0)
        {
            cout << t << "/" << num_tsteps << "...writing output\n";
            slab.write_output(time);
        }
        if(t % tout_diag == 0)
        {
            cout << t << "/" << num_tsteps << "...writing diagnostics\n";
            slab_diag.update_arrays(slab);
            slab_diag.write_diagnostics(time, my_config);
        }
        
    }
    return(0);
}


