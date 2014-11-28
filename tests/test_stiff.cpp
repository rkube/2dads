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
    slab.integrate_stiff_debug(twodads::field_k_t::f_theta_hat, uint(4), uint(2), uint(2));
    slab.print_field(twodads::field_k_t::f_theta_hat);
}
