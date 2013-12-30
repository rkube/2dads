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
    cout << "Hello, World!\n";
    cout << "config.nx = " << my_config.get_nx() << ", My = " << my_config.get_my() << "\n";

    slab_cuda slab(my_config);
    slab.dump_field(twodads::field_k_t::f_theta_hat);

    // Initialize
    slab.initialize();
    slab.dump_field(twodads::field_t::f_theta);
    slab.dump_field(twodads::field_k_t::f_theta_hat);


    return(0);
}
