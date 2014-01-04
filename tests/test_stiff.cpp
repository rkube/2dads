/*
 *
 * test_stiff.cpp
 *
 */

#include <iostream>
#include <include/slab_cuda.h>

using namespace std;

int main(void)
{
    slab_config my_config;
    my_config.consistency();

    slab_cuda slab(my_config);
    slab.dump_stiff_params();
    slab.initialize();
    slab.integrate_stiff(twodads::dyn_field_t::d_theta, 4);
    cudaDeviceSynchronize();
}
