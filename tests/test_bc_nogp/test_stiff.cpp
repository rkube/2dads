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

    cout << "Enter Nx:";
    cin >> Nx;
    cout << "Enter My:";
    cin >> My;
    
    twodads::slab_layout_t my_geom(x_l, Lx / double(Nx), y_l, Ly / double(My), Nx, 0, My, 2, twodads::grid_t::cell_centered);
    twodads::bvals_t<double> my_bvals{twodads::bc_t::bc_dirichlet, twodads::bc_t::bc_dirichlet, twodads::bc_t::bc_periodic, twodads::bc_t::bc_periodic,
                                   0.0, 0.0, 0.0, 0.0};
    twodads::stiff_params_t params(0.1, Lx, Ly, 0.1, 0.0, my_geom.get_nx(), (my_geom.get_my() + my_geom.get_pad_y()) / 2, 2);
    {
        slab_bc my_slab(my_geom, my_bvals, params);
        my_slab.initialize_gaussian(test_ns::field_t::arr1, 1);

        for(size_t t = 0; t < 1; t++)
        {
            std::cout << "Integrating: t = " << t << std::endl;
            my_slab.dft_r2c(test_ns::field_t::arr1, 1);
            my_slab.integrate(test_ns::field_t::arr1, 1);
            my_slab.dft_c2r(test_ns::field_t::arr1, 1);
        }

        my_slab.dft_c2r(test_ns::field_t::arr1, 0);
        
        std::cout << "T=0" << std::endl;
        utility :: print((*my_slab.get_array_ptr(test_ns::field_t::arr1)), 0, std::cout);
        std::cout << "T=1" << std::endl;
        utility :: print((*my_slab.get_array_ptr(test_ns::field_t::arr1)), 1, std::cout);

    }

}

