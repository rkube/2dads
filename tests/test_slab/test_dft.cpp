#include <iostream>
#include <vector>
#include "slab_cuda.h"

using namespace std;

int main(void)
{

    slab_config my_config;
    my_config.consistency();

    slab_cuda slab(my_config);
    slab.initialize();

    slab.print_field(twodads::field_t::f_theta);
    slab.dft_r2c(twodads::field_t::f_theta, twodads::field_k_t::f_theta_hat, 0);
    slab.print_field(twodads::field_k_t::f_theta_hat);


    return(0);
}

