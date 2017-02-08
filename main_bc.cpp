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
    const size_t order{my_config.get_tint_params(twodads::dyn_field_t::f_theta).get_tlevs()};
    std::cout << "Integrating: order = " << order << std::endl;
    size_t tstep{0};
    const size_t num_tsteps{static_cast<size_t>(std::round(my_config.get_tend() / my_config.get_deltat()))};
    const size_t output_step{static_cast<size_t>(std::round(my_config.get_tout() / my_config.get_deltat()))};

    {
        slab_bc my_slab(my_config);
        my_slab.initialize();
        // output:
        // FD: all fields are complex
        // BS: all fields are complex

        std::cout << "Inverting laplace" << std::endl;
        // input:
        // FD: src.is_transformed(t_src) = true
        // BS: src.is_transformed(t_src) = true
        my_slab.invert_laplace(twodads::field_t::f_omega, twodads::field_t::f_strmf, order - 1, 0);
        // output:
        // FD: dst.is_transformed(t_dst) = true 
        // BS: dst.is_transformed(t_dst) = true
        std::cout << "...done. Calculating RHS" << std::endl;
        // input:
        // FD:
        // BS: field.is_transformed(t_src) = true
        my_slab.update_real_fields(order - 1);
        // output:
        // FD:
        // BS: field.is_transformed(t_src) = false
        std::cout << "...done. Updating real fields";

        // input:
        // FD: field.is_transformed(t_src) = false
        // BS: field.is_transformed(t_src) = false
        my_slab.write_output(order - 1, tstep * my_config.get_deltat());
        // output:
        // FD: field.is_transformed(t_src) = false
        // BS: field.is_transformed(t_src) = false

        // input:
        // FD:
        // BS: field.is_tranformed(t_src) = false
        my_slab.rhs(order - 2, order - 1);
        // output:
        // FD:
        // BS: rhs.is_transformed(t_dst) = true


        tstep++;

/////////////////////////////////////////////////////////////////////////
        my_slab.integrate(twodads::dyn_field_t::f_theta, 1);
        my_slab.integrate(twodads::dyn_field_t::f_omega, 1);
        my_slab.integrate(twodads::dyn_field_t::f_tau, 1);

        my_slab.invert_laplace(twodads::field_t::f_omega, twodads::field_t::f_strmf, order - 2, 0);
        my_slab.update_real_fields(order - 2);
        my_slab.write_output(order - 2, tstep * my_config.get_deltat());
        my_slab.rhs(order - 3, order - 2);
        tstep++;

/////////////////////////////////////////////////////////////////////////
        my_slab.integrate(twodads::dyn_field_t::f_theta, 2);
        my_slab.integrate(twodads::dyn_field_t::f_omega, 2);
        my_slab.integrate(twodads::dyn_field_t::f_tau, 2);

        my_slab.invert_laplace(twodads::field_t::f_omega, twodads::field_t::f_strmf, order - 3, 0);
        my_slab.update_real_fields(order - 3);
        my_slab.write_output(order - 3, tstep * my_config.get_deltat());
        my_slab.rhs(0, order - 3);

/////////////////////////////////////////////////////////////////////////
        for(; tstep < num_tsteps; tstep++)
        {
        	std::cout << tstep << "/" << num_tsteps << std::endl;
            my_slab.integrate(twodads::dyn_field_t::f_theta, 3);
            my_slab.integrate(twodads::dyn_field_t::f_omega, 3);
            my_slab.integrate(twodads::dyn_field_t::f_tau, 3);

            my_slab.invert_laplace(twodads::field_t::f_omega, 
                                   twodads::field_t::f_strmf, 0, 0);
            my_slab.update_real_fields(0);

            if((tstep % output_step) == 0)
            {
                std::cout << "step " << tstep << "/" << num_tsteps << ": writing output" << std::endl;
                my_slab.write_output(0);
            }
            my_slab.rhs(0, 0);
            my_slab.advance();
        }

    }
    std::cout << "Leaving scope" << std::endl;
}
