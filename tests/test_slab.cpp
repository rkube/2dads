/*
 *
 * test_slab.cpp
 *
 */

#include <iostream>
#include "include/slab_cuda.h"

using namespace std;

int main(void)
{
    slab_config my_config;
    my_config.consistency();
    cout << "Hello, World!\n";
    cout << "config.nx = " << my_config.get_nx() << ", My = " << my_config.get_my() << "\n";

    slab_cuda slab(my_config);

    // Initialize
    slab.initialize();
    slab.move_t(twodads::field_k_t::f_theta_hat, 0, my_config.get_tlevs() - 1);
    cout << "theta = \n";
    slab.dump_field(twodads::field_t::f_theta);
    //cout << "theta_hat = \n";
    //slab.dump_field(twodads::field_k_t::f_theta_hat);
    //slab.d_dy(twodads::field_k_t::f_theta_hat, twodads::field_k_t::f_theta_y_hat);
    slab.inv_laplace(twodads::field_k_t::f_theta_hat, twodads::field_k_t::f_tmp_hat, 0);
    cout << "tmp_hat = \n";
    slab.dump_field(twodads::field_k_t::f_tmp_hat);
    cout << "c2r, normalizing...\n";
    slab.dft_c2r(twodads::field_k_t::f_tmp_hat, twodads::field_t::f_theta, 0);
    slab.dump_field(twodads::field_t::f_theta);

    return(0);
}
