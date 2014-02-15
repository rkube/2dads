#include <iostream>
#include "include/slab_cuda.h"

using namespace std;

int main(void)
{
    slab_config my_config;
    my_config.consistency();

    slab_cuda slab(my_config);
    slab.initialize();

    cout << "Initializing slab\n";

    cout << "theta_hat = \n";
    slab.dump_field(twodads::field_k_t::f_theta_hat);
    cout << "theta_rhs_hat = \n";
    slab.dump_field(twodads::field_k_t::f_theta_rhs_hat);
    cout << "omega_hat = \n";
    slab.dump_field(twodads::field_k_t::f_omega_hat);
    cout << "omega_rhs_hat = \n";
    slab.dump_field(twodads::field_k_t::f_omega_rhs_hat);
    cout << "strmf_hat= \n";
    slab.dump_field(twodads::field_k_t::f_strmf_hat);
    cout << "theta=\n";
    slab.dump_field(twodads::field_t::f_theta);
    cout << "theta_x=\n";
    slab.dump_field(twodads::field_t::f_theta_x);
    cout << "theta_y=\n";
    slab.dump_field(twodads::field_t::f_theta_y);
    cout << "omega=\n";
    slab.dump_field(twodads::field_t::f_omega);
    cout << "omega_x=\n";
    slab.dump_field(twodads::field_t::f_omega_x);
    cout << "omega_y=\n";
    slab.dump_field(twodads::field_t::f_omega_y);
    cout << "strmf=\n";
    slab.dump_field(twodads::field_t::f_strmf);
    cout << "strmf_x=\n";
    slab.dump_field(twodads::field_t::f_strmf_x);
    cout << "strmf_y=\n";
    slab.dump_field(twodads::field_t::f_strmf_y);

}


