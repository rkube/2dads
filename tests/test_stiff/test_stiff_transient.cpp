/*
 * test_stiff_transient.cpp
 *
 * Integrate the first transient time steps with full debugging output
 */

#include <iostream>
#include <slab_cuda.h>

using namespace std;

int main(void)
{
    slab_config my_config;
    my_config.consistency();

    slab_cuda slab(my_config);
    slab.initialize();
    cout << "omega_hat=\n";
    slab.print_field(twodads::field_k_t::f_omega_hat);
    cout << "theta_hat=\n";
    slab.print_field(twodads::field_k_t::f_theta_hat);
    cout << "strmf_hat=\n";
    slab.print_field(twodads::field_k_t::f_strmf_hat);
    cout << "strmf_y_hat=\n";
    slab.print_field(twodads::field_k_t::f_strmf_y_hat);
    cout << "omega_rhs_hat=\n";
    slab.print_field(twodads::field_k_t::f_omega_rhs_hat);
    cout << "theta_rhs_hat=\n";
    slab.print_field(twodads::field_k_t::f_theta_rhs_hat);

    twodads::real_t time(0.0);
    twodads::real_t delta_t(my_config.get_deltat());
    const unsigned int num_tsteps(ceil(my_config.get_tend() / my_config.get_deltat()));
    const unsigned int tlevs = my_config.get_tlevs();
    unsigned int t = 1;

    for(t = 1; t < tlevs - 1; t++)
    {
        cout << "=========================================================================\n";
        cout << "integrating\n";
        slab.integrate_stiff_debug(twodads::field_k_t::f_omega_hat, t + 1, uint(2), uint(0));
        slab.integrate_stiff(twodads::field_k_t::f_omega_hat, t + 1); 
        slab.integrate_stiff(twodads::field_k_t::f_theta_hat, t + 1); 
        cout << "omega_hat=\n";
        slab.print_field(twodads::field_k_t::f_omega_hat);
        cout << "theta_hat=\n";
        slab.print_field(twodads::field_k_t::f_theta_hat);
        cout << "strmf_hat=\n";
        slab.print_field(twodads::field_k_t::f_strmf_hat);
        cout << "strmf_y_hat=\n";
        slab.print_field(twodads::field_k_t::f_strmf_y_hat);

        slab.inv_laplace(twodads::field_k_t::f_omega_hat, twodads::field_k_t::f_strmf_hat, tlevs - t - 1);
        slab.update_real_fields(tlevs - t - 1);
        slab.rhs_fun(tlevs - t - 1);

        if(t == 1)
        {
            slab.move_t(twodads::field_k_t::f_theta_rhs_hat, tlevs - t - 2, 0);
            slab.move_t(twodads::field_k_t::f_omega_rhs_hat, tlevs - t - 2, 0);
        }
        cout << "omega_rhs_hat=\n";
        slab.print_field(twodads::field_k_t::f_omega_rhs_hat);
        cout << "theta_rhs_hat=\n";
        slab.print_field(twodads::field_k_t::f_theta_rhs_hat);
    }

    for(; t < num_tsteps + 1; t++)
    {
        cout << "========================================t = " << t << "========================================\n";
        slab.integrate_stiff_debug(twodads::field_k_t::f_omega_hat, tlevs, uint(2), uint(0));
        slab.integrate_stiff(twodads::field_k_t::f_theta_hat, tlevs);
        slab.integrate_stiff(twodads::field_k_t::f_omega_hat, tlevs);
        slab.advance();
        time += delta_t;
        cout << "omega_hat=\n";
        slab.print_field(twodads::field_k_t::f_omega_hat);
        cout << "theta_hat=\n";
        slab.print_field(twodads::field_k_t::f_theta_hat);
        cout << "strmf_hat=\n";
        slab.print_field(twodads::field_k_t::f_strmf_hat);
        cout << "strmf_y_hat=\n";
        slab.print_field(twodads::field_k_t::f_strmf_y_hat);
        // Compute all fields from theta_hat(1,:), omega_hat(1,:) since we called advance
        slab.inv_laplace(twodads::field_k_t::f_omega_hat, twodads::field_k_t::f_strmf_hat, 1);
        slab.update_real_fields(1);
        slab.rhs_fun(1);

        cout << "omega_rhs_hat=\n";
        slab.print_field(twodads::field_k_t::f_omega_rhs_hat);
        cout << "theta_rhs_hat=\n";
        slab.print_field(twodads::field_k_t::f_theta_rhs_hat);
    }        

}
