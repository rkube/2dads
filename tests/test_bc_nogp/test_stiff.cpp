/*
 * Test time integration 
 */

#include <iostream>
#include <sstream>
#include "slab_bc.h"

using namespace std;

int main(void)
{
    size_t Nx{0};
    size_t My{0};

    const twodads::real_t x_l{-10.0};
    const twodads::real_t Lx{20.0};
    const twodads::real_t y_l{-10.0};
    const twodads::real_t Ly{20.0};

    const size_t tlevs{4};

    const twodads::real_t deltat{0.1};
    const twodads::real_t diff{0.1};
    const twodads::real_t hv{0.0};

    const twodads::real_t num_tsteps{50};

    cout << "Enter Nx:";
    cin >> Nx;
    cout << "Enter My:";
    cin >> My;

    stringstream(fname);
    
    twodads::slab_layout_t my_geom(x_l, Lx / double(Nx), y_l, Ly / double(My), Nx, 0, My, 2, twodads::grid_t::cell_centered);
    twodads::bvals_t<double> my_bvals{twodads::bc_t::bc_dirichlet, twodads::bc_t::bc_dirichlet, twodads::bc_t::bc_periodic, twodads::bc_t::bc_periodic,
                                   0.0, 0.0, 0.0, 0.0};
    twodads::stiff_params_t params(deltat, Lx, Ly, diff, hv, my_geom.get_nx(), (my_geom.get_my() + my_geom.get_pad_y()) / 2, tlevs);
    {
        slab_bc my_slab(my_geom, my_bvals, params);
        my_slab.initialize_gaussian(twodads::field_t::theta, tlevs - 1);

        my_slab.update_derivatives();


        size_t tstep{0};
        for(size_t tl = 0; tl < tlevs; tl++)
        {
            fname.str(string(""));
            fname << "test_stiff_solnum_" << Nx << "_a" << tl << "_t" << tstep << "_host.dat";
            utility :: print((*my_slab.get_array_ptr(twodads::field_t::theta)), tl, fname.str());        
        }

        // Integrate first time step
        std::cout << "Integrating: t = " << tstep << std::endl;
        tstep = 1;
        my_slab.integrate(twodads::field_t::theta, 1);
        for(size_t tl = 0; tl < tlevs; tl++)
        {
            fname.str(string(""));
            fname << "test_stiff_solnum_" << Nx << "_a" << tl << "_t" << tstep << "_host.dat";
            utility :: print((*my_slab.get_array_ptr(twodads::field_t::theta)), tl, fname.str());  
        }


        // Integrate second time step
        std::cout << "Integrating: t = " << tstep << std::endl;
        tstep = 2;
        my_slab.integrate(twodads::field_t::theta, 2);
        for(size_t tl = 0; tl < tlevs; tl++)
        {
            fname.str(string(""));
            fname << "test_stiff_solnum_" << Nx << "_a" << tl << "_t" << tstep << "_host.dat";
            utility :: print((*my_slab.get_array_ptr(twodads::field_t::theta)), tl, fname.str());      
        }
        
        //tstep++;
    
        for(; tstep < num_tsteps; tstep++)
        {
            // Integrate with third order scheme now
            std::cout << "Integrating: t = " << tstep << std::endl;
            my_slab.integrate(twodads::field_t::theta, 3);
            for(size_t tl = 0; tl < tlevs; tl++)
            {
                fname.str(string(""));
                fname << "test_stiff_solnum_" << Nx << "_a" << tl << "_t" << tstep << "_host.dat";
                utility :: print((*my_slab.get_array_ptr(twodads::field_t::theta)), tl, fname.str());        
            }
            my_slab.advance();
        }
    }
}

