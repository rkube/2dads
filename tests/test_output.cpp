/*
 *
 * test_output.cpp
 *
 * See if output really works
 *
 */

#include <iostream>
#include "include/slab_cuda.h"
#include "include/output.h"

using namespace std;

int main(void)
{
    cout << "Hello, World!\n";
    slab_config my_config;
    my_config.consistency();

    slab_cuda slab(my_config);
    output_h5 output_t(my_config);
    //diagnostics diag_t(my_config);


    slab.initialize();
    twodads::real_t time = 0.0;
    output_t.write_output(slab, time);
    //slab.dump_field(twodads::field_t::f_theta);
    time+= my_config.get_deltat();
    output_t.write_output(slab, time);
}
