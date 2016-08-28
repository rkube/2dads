/*
 * Time integration using the new boundary value array
 */


#include <iostream>
#include "slab_bc.h"
//#include "diagonstics.h"
#include "output.h"

using namespace std;


int main(void)
{
    slab_config_js my_config(std::string("input.json"));
    const size_t tlevs{4};

    {
        slab_bc my_slab(my_config);
        my_slab.initialize();

        my_slab.invert_laplace(twodads::field_t::f_omega, twodads::field_t::f_strmf, tlevs - 1, 0);
        //my_slab.update_real_field(tlevs - 1);

        my_slab.write_output(tlevs - 1);
    }
}
