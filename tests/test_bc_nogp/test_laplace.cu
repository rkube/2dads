/*
 * Invert the laplace equation with boundary conditions in x using the cusparse tridiagonal solver with the new datatype
 *
 * Invert
 * g(x,y) = exp(-(x^2 + y^2) / 2)
 * \nabla^2 g(x,y) = f(x,y) 
 * where
 * f(x,y) = exp(-(x^2 + y^2) / 2) (-2 + x^2 + y^2)
 *
 * Goal: Given f(x,y) find g(x,y)
 */


#include <iostream>
#include <cassert>
#include "slab_bc.h"
#include "derivatives.h"
#include <sstream>

using namespace std;

int main(void){
    size_t Nx{128};
    size_t My{128};

    cout << "Enter Nx: ";
    cin >> Nx;
    cout << "Enter My: ";
    cin >> My;

    stringstream fname;

    cuda::slab_layout_t my_geom(-10.0, 20.0 / double(Nx), -10.0, 20.0 / double(My), Nx, 0, My, 2, cuda::grid_t::cell_centered);
    cuda::bvals_t<double> my_bvals{cuda::bc_t::bc_dirichlet, cuda::bc_t::bc_dirichlet, cuda::bc_t::bc_periodic, cuda::bc_t::bc_periodic,
                                   0.0, 0.0, 0.0, 0.0};
    {
        slab_bc my_slab(my_geom, my_bvals);
        my_slab.initialize_invlaplace(test_ns::field_t::arr1);

        cuda_array_bc_nogp<my_allocator_device<cuda::real_t>> sol_an(my_geom, my_bvals, 1);
        sol_an.evaluate([=] __device__ (size_t n, size_t m, cuda::slab_layout_t geom) -> cuda::real_t
                {
                    const cuda::real_t x{geom.get_x(n)};
                    const cuda::real_t y{geom.get_y(m)};
                    return((-2.0 + x * x + y * y) * exp(-0.5 * (x * x + y * y)));
                },
            0);

        // Print input to inv_laplace routine into array arr1_nx.dat
        fname << "test_laplace_arr1_" << Nx << ".dat";
        my_slab.print_field(test_ns::field_t::arr1, fname.str());
        my_slab.dft_r2c(test_ns::field_t::arr1, 0);

        my_slab.invert_laplace(test_ns::field_t::arr2, test_ns::field_t::arr1, 0, 0);
        my_slab.dft_c2r(test_ns::field_t::arr2, 0);

        // Write numerical solution to file
        fname.str(string(""));
        fname << "test_laplace_solnum_" << Nx << ".dat";
        my_slab.print_field(test_ns::field_t::arr2, fname.str());

        // Get the analytic solution
        cuda_array_bc_nogp<my_allocator_device<cuda::real_t>> sol_num(my_slab.get_array_ptr(test_ns::field_t::arr2));
        sol_num -= sol_an;
        cout << "Nx = " << Nx << ", My = " << My << ", L2 = " << sol_num.L2(0) << endl;

    } // Let managed memory go out of scope before calling cudaDeviceReset()
    cudaDeviceReset();
}
