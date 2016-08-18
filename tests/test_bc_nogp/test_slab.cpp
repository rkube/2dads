/*
 * Test Fourier Transformation
 */

#include <iostream>
#include "slab_bc.h"

using namespace std;

int main(void)
{ 
    size_t Nx{16};
    size_t My{16};

    cout << "Enter Nx: " <<endl ;
    cin >> Nx;
    cout << "Enter My: " << endl;
    cin >> My;

    constexpr twodads::real_t Lx{1.0};
    constexpr twodads::real_t x_l{0.0};
    constexpr twodads::real_t Ly{1.0};
    constexpr twodads::real_t y_l{0.0};

    twodads::bvals_t<double> my_bvals{twodads::bc_t::bc_dirichlet, twodads::bc_t::bc_dirichlet, twodads::bc_t::bc_periodic, twodads::bc_t::bc_periodic, 0.0, 0.0, 0.0, 0.0};
    twodads::slab_layout_t my_geom(x_l, (Lx - x_l) / double(Nx), y_l, (Ly - y_l) / double(My), Nx, 0, My, 2, twodads::grid_t::vertex_centered);
    twodads::stiff_params_t my_stiff_params{0.1, 1.0, 1.0, 0.1, 0.0, Nx, My / 2 + 1, 3};
    {
        slab_bc my_slab(my_geom, my_bvals, my_stiff_params);

        my_slab.initialize_dfttest(test_ns::field_t::arr1);
        my_slab.print_field(test_ns::field_t::arr1);

        my_slab.dft_r2c(test_ns::field_t::arr1, 0);
        cout << "===========================================================================================" << endl;

        //my_slab.dft_c2r(test_ns::field_t::arr1, 0);
        my_slab.print_field(test_ns::field_t::arr1);
    } // Let managed memory go out of scope before calling cudaDeviceReset()
}
