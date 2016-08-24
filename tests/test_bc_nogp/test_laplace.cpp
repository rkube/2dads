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
#include <sstream>
#include "slab_bc.h"

using namespace std;

int main(void){
    size_t Nx{128};
    size_t My{128};
    const twodads::real_t x_l{-10.0};
    const twodads::real_t Lx{20.0};
    const twodads::real_t y_l{-10.0};
    const twodads::real_t Ly{20.0};

    cout << "Enter Nx: ";
    cin >> Nx;
    cout << "Enter My: ";
    cin >> My;

    stringstream fname;

    twodads::slab_layout_t my_geom(x_l, Lx / double(Nx), y_l, Ly / double(My), Nx, 0, My, 2, twodads::grid_t::cell_centered);
    twodads::bvals_t<double> my_bvals{twodads::bc_t::bc_dirichlet, twodads::bc_t::bc_dirichlet, twodads::bc_t::bc_periodic, twodads::bc_t::bc_periodic,
                                   0.0, 0.0, 0.0, 0.0};
    twodads::stiff_params_t params(0.001, 20.0, 20.0, 0.001, 0.0, my_geom.get_nx(), (my_geom.get_my() + my_geom.get_pad_y()) / 2, 1);
    {
        slab_bc my_slab(my_geom, my_bvals, params);
        my_slab.initialize_invlaplace(test_ns::field_t::arr1);
        fname << "test_laplace_input_" << Nx << "_host.dat";
        utility :: print((*my_slab.get_array_ptr(test_ns::field_t::arr1)), 0, fname.str());
        fname.str(string(""));

        cuda_array_bc_nogp<twodads::real_t, allocator_host> sol_an(my_geom, my_bvals, 1);
        sol_an.apply([] (twodads::real_t dummy, size_t n, size_t m, twodads::slab_layout_t geom) -> twodads::real_t
                {
                    const twodads::real_t x{geom.get_x(n)};
                    const twodads::real_t y{geom.get_y(m)};
                    return(exp(-0.5 * (x * x + y * y)));
                    //return(sin(twodads::TWOPI * y));
                },
            0);

        fname << "test_laplace_solan_" << Nx << "_host.dat";
        utility :: print(sol_an, 0, fname.str());

        my_slab.dft_r2c(test_ns::field_t::arr1, 0);
        my_slab.invert_laplace(test_ns::field_t::arr1, test_ns::field_t::arr2, 0, 0);
        my_slab.dft_c2r(test_ns::field_t::arr2, 0);

        // Write numerical solution to file
        fname.str(string(""));
        fname << "test_laplace_solnum_" << Nx << "_host.dat";
        utility :: print((*my_slab.get_array_ptr(test_ns::field_t::arr2)), 0, fname.str());

        // Get the analytic solution
        cuda_array_bc_nogp<twodads::real_t, allocator_host> sol_num(my_slab.get_array_ptr(test_ns::field_t::arr2));
        sol_num -= sol_an;
        cout << "Nx = " << Nx << ", My = " << My << ", L2 = " << utility :: L2(sol_num, 0) << endl;
    } // Let managed memory go out of scope before calling cudaDeviceReset()
}