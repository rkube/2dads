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

    cout << "Enter Nx: ";
    cin >> Nx;
    cout << "Enter My: ";
    cin >> My;

    constexpr twodads::real_t Lx{1.0};
    constexpr twodads::real_t x_l{0.0};
    constexpr twodads::real_t Ly{1.0};
    constexpr twodads::real_t y_l{0.0};

    twodads::bvals_t<double> my_bvals{twodads::bc_t::bc_dirichlet, twodads::bc_t::bc_dirichlet, twodads::bc_t::bc_periodic, twodads::bc_t::bc_periodic, 0.0, 0.0, 0.0, 0.0};
    twodads::slab_layout_t my_geom(x_l, (Lx - x_l) / double(Nx), y_l, (Ly - y_l) / double(My), Nx, 0, My, 2, twodads::grid_t::vertex_centered);
    twodads::stiff_params_t my_stiff_params{0.1, 1.0, 1.0, 0.1, 0.0, Nx, My / 2 + 1, 1};
    {
        slab_bc my_slab(my_geom, my_bvals, my_stiff_params);
        std::cout << "I got slab" << std::endl;

        my_slab.initialize_dfttest(twodads::field_t::f_theta, 0);
        std::cout << "Hello!" << std::endl;
        utility :: print((*my_slab.get_array_ptr(twodads::field_t::f_theta)), 0, std::cout);

        my_slab.dft_r2c(twodads::field_t::f_theta, 0);
        cout << "===============================================================================================" << endl;

        //my_slab.dft_c2r(twodads::field_t::f_theta, 0);
        utility :: print((*my_slab.get_array_ptr(twodads::field_t::f_theta)), 0, std::cout);
    } // Let managed memory go out of scope before calling cudaDeviceReset()
    cudaDeviceReset();
}
