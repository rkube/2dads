/*
 * Test time integration 
 */

#include <iostream>
#include <sstream>
#include "slab_bc.h"
//#include "output.h"

using namespace std;

int main(void)
{
    constexpr size_t num_tstep{10};
    slab_config_js my_config(std::string("input_test_stiff_fd.json"));
    const size_t order{my_config.get_tint_params(twodads::dyn_field_t::f_theta).get_tlevs()};
    stringstream fname;
    {
        slab_bc my_slab(my_config);
        // Semispectral version: initializes leaves all fields (order - 1) fourier transformed
        my_slab.initialize();
        size_t tstep{0};

        // invert_laplace: semispectral version
        // input:
        //      omega is transformed
        //      strmf is irrelevant
        //my_slab.invert_laplace(twodads::field_t::f_omega, twodads::field_t::f_strmf, order - 1, 0);
        //my_slab.update_real_fields(order - 1);
        
        
        for(size_t tl = 0; tl < order; tl++)
        {
            fname.str(string(""));
            fname << "test_stiff_theta_" << my_config.get_nx() << "_a" << tl << "_t" << tstep << "_host.dat";
            utility :: print((*my_slab.get_array_ptr(twodads::field_t::f_theta)), tl, fname.str());        
        }
        my_slab.rhs(order - 2, order - 1);

        // Integrate first time step
        std::cout << "Integrating: t = " << tstep << std::endl;
        tstep = 1;
        my_slab.integrate(twodads::dyn_field_t::f_theta, 1);
        my_slab.integrate(twodads::dyn_field_t::f_omega, 1);
        my_slab.integrate(twodads::dyn_field_t::f_tau, 1);

        my_slab.invert_laplace(twodads::field_t::f_omega, twodads::field_t::f_strmf, order - 2, 0);
        my_slab.update_real_fields(order - 2);

        for(size_t tl = 0; tl < order; tl++)
        {
            fname.str(string(""));
            fname << "test_stiff_solnum_" << my_config.get_nx() << "_a" << tl << "_t" << tstep << "_host.dat";
            utility :: print((*my_slab.get_array_ptr(twodads::field_t::f_theta)), tl, fname.str());  
        }
        my_slab.rhs(order - 3, order - 2);

        // Integrate second time step
        std::cout << "Integrating: t = " << tstep << std::endl;
        tstep = 2;
        my_slab.integrate(twodads::dyn_field_t::f_theta, 2);
        my_slab.integrate(twodads::dyn_field_t::f_omega, 2);
        my_slab.integrate(twodads::dyn_field_t::f_tau, 2);

        my_slab.invert_laplace(twodads::field_t::f_omega, twodads::field_t::f_strmf, order - 3, 0);
        my_slab.update_real_fields(order - 3);
        
        for(size_t tl = 0; tl < order; tl++)
        {
            fname.str(string(""));
            fname << "test_stiff_solnum_" << my_config.get_nx() << "_a" << tl << "_t" << tstep << "_host.dat";
            utility :: print((*my_slab.get_array_ptr(twodads::field_t::f_theta)), tl, fname.str());      
        }
        my_slab.rhs(order - 4, order - 2);
    
        for(; tstep < num_tstep; tstep++)
        {
            // Integrate with third order scheme now
            std::cout << "Integrating: t = " << tstep << std::endl;
            my_slab.integrate(twodads::dyn_field_t::f_theta, 3);
            my_slab.integrate(twodads::dyn_field_t::f_omega, 3);
            my_slab.integrate(twodads::dyn_field_t::f_tau, 3);

            my_slab.invert_laplace(twodads::field_t::f_omega, twodads::field_t::f_strmf, 0, 0);
            my_slab.update_real_fields(0);

            for(size_t tl = 0; tl < order; tl++)
            {
                fname.str(string(""));
                fname << "test_stiff_solnum_" << my_config.get_nx() << "_a" << tl << "_t" << tstep << "_host.dat";
                utility :: print((*my_slab.get_array_ptr(twodads::field_t::f_theta)), tl, fname.str());        
            }
            my_slab.rhs(0, 0);
            my_slab.advance();
        }
    }
}

