/*
 *
 * test_stiff.cpp
 *
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
    cout << "omega_rhs_hat=\n";
    slab.print_field(twodads::field_k_t::f_omega_rhs_hat);
    cout << "theta_rhs_hat=\n";
    slab.print_field(twodads::field_k_t::f_theta_rhs_hat);
    cout << "integrating\n";
    //slab.integrate_stiff_debug(twodads::field_k_t::f_omega_hat, uint(2), uint(7), uint(2));
    slab.integrate_stiff(twodads::field_k_t::f_omega_hat, uint(2)); 
    slab.integrate_stiff(twodads::field_k_t::f_theta_hat, uint(2)); 
    cout << "omega_hat=\n";
    slab.print_field(twodads::field_k_t::f_omega_hat);
    cout << "theta_hat=\n";
    slab.print_field(twodads::field_k_t::f_theta_hat);
}
