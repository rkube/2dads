/*
 * Solve the diffusion eqation
 */

#include <iostream>
#include <sstream>
#include "slab_bc.h"

using namespace std;
int main(void)
{
    size_t Nx{16};
    size_t My{16};
    const size_t num_tsteps{1000};

    cout << "Enter Nx: ";
    cin >> Nx;

    cout << "Enter My: ";
    cin >> My;

    stringstream fname;

    cuda::bvals_t<double> my_bvals{cuda::bc_t::bc_dirichlet, cuda::bc_t::bc_dirichlet,
                                   cuda::bc_t::bc_periodic, cuda::bc_t::bc_periodic,
                                   0.0, 0.0, 0.0, 0.0};
    cuda::slab_layout_t my_geom(-10.0, 20.0 / double(Nx), -10.0, 20.0 / double(My), Nx, 0, My, 2, cuda::grid_t::cell_centered);
    cuda::stiff_params_t stiff_params(0.01, 20.0, 20.0, 0.1, 0.0, My, Nx / 2 + 1, 2);
    {
        slab_bc my_slab(my_geom, my_bvals, stiff_params);
        //cuda_array_bc_nogp<my_allocator_device<cuda::real_t>> sol(my_geom, my_bvals, 1);

        my_slab.initialize_gaussian(test_ns::field_t::arr1);
        my_slab.initialize_tint(test_ns::field_t::arr1);

        //my_slab.integrate((cuDoubleComplex*) sol.get_array_d(0));
        //size_t t{0};
        //for(t = 0; t < num_tsteps; t++)
        //{
        //    my_slab.integrate(test_ns::field_t::arr1);
        //}

        //fname.str(string(""));
        //fname << "test_diffusion_Nx" << Nx << "_t" << t << "_out.dat";
        //my_slab.print_field(test_ns::field_t::arr1, fname.str());
    }
}
