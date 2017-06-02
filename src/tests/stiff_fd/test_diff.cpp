/* Test the diffusion equation, the implicit part of the 
   time integrator
*/

#include <iostream>
#include "slab_bc.h"
#include "output.h"

using namespace std;

int main(void)
{
    slab_config_js my_config(std::string("input_test_diff_fd.json"));
    const size_t order{my_config.get_tint_params(twodads::dyn_field_t::f_theta).get_tlevs()};

    size_t tstep{0};
    const size_t num_tsteps{static_cast<size_t>(std::round(my_config.get_tend() / my_config.get_deltat()))};
    const size_t output_step{static_cast<size_t>(std::round(my_config.get_tout() / my_config.get_deltat()))};

    {
        slab_bc my_slab(my_config);
        my_slab.initialize();
        std::cout << "Slab initialized" << std::endl;
        my_slab.update_real_fields(order - 1);
        my_slab.write_output(order - 1, 0.0);

        ////////////////////////////////////////////////////////////////////////
        my_slab.integrate(twodads::dyn_field_t::f_theta, 1);
        my_slab.update_real_fields(order - 2);
        my_slab.write_output(order - 2, tstep * my_config.get_deltat());
        tstep++;

        ////////////////////////////////////////////////////////////////////////
        my_slab.integrate(twodads::dyn_field_t::f_theta, 2);
        my_slab.update_real_fields(order - 3);
        my_slab.write_output(order - 3, tstep * my_config.get_deltat());
        tstep++;

        for(; tstep < num_tsteps; tstep++)
        {
            my_slab.integrate(twodads::dyn_field_t::f_theta, 3);
            my_slab.advance();
            my_slab.update_real_fields(1);

            if((tstep % output_step) == 0)
            {
                std::cout << "step " << tstep << "/" << num_tsteps << ": writing output" << std::endl;
                my_slab.write_output(1, tstep * my_config.get_deltat());
            }
        }
    }

}