/*
 *
 * test_output.cpp
 *
 * See if output really works
 *
 */

#include <iostream>
#include "include/slab_cuda.h"

using namespace std;

int main(void)
{
    cout << "Hello, World!\n";
    slab_config my_config;
    my_config.consistency();

    slab_cuda slab(my_config);
    slab.initialize();
    twodads::real_t time = 0.0;
    slab.write_output(time);
    slab.dump_field(twodads::field_t::f_theta);
    slab.write_output(time);
}
