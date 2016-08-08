/*
 * Test derivatives
 *
 * Input:
 *      f(x, y) = sin(2 * pi * x)
 *      f_x = 2 * pi * cos(2 pi x)
 *      f_xx =- -4 * pi * sin(2 pi x)
 *      -> Initializes arr1
 *
 *      g(x, y) = exp(-50 * (y-0.5) * (y-0.5))
 *      g_y = -100 * (y - 0.5) * exp(-50 * (y - 0.5) * (y - 0.5))
 *      g_yy = 10000 * (0.24 - y + y^2) * exp(-50 * (y - 0.5) * (y - 0.5))
 *      -> Initializes arr2
 *
 *
 *
 */

#include <iostream>
#include <fstream>
#include <sstream>
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

    stringstream fname;
    ofstream of;

    cuda::bvals_t<double> my_bvals{cuda::bc_t::bc_dirichlet, cuda::bc_t::bc_dirichlet, cuda::bc_t::bc_periodic, cuda::bc_t::bc_periodic, 0.0, 0.0, 0.0, 0.0};
    cuda::slab_layout_t my_geom(0.0, 1.0 / double(Nx), 0.0, 1.0 / double(My), Nx, 0, My, 2);

    {
        slab_bc my_slab(my_geom, my_bvals);
        my_slab.initialize_derivatives(test_ns::field_t::arr1, test_ns::field_t::arr2);

        // Initialize analytic solution for first derivative
        cuda_array_bc_nogp<my_allocator_device<cuda::real_t>> sol_an(my_geom, my_bvals, 1);
        sol_an.evaluate([=] __device__(size_t n, size_t m, cuda::slab_layout_t geom) -> cuda::real_t
            {
                cuda::real_t x{geom.get_xleft() + (cuda::real_t(n) + 0.5) * geom.get_deltax()};
                //cuda::real_t y{geom.get_ylo() + (cuda::real_t(m) + 0.5) * geom.get_deltay()};
                return(cuda::TWOPI * cos(cuda::TWOPI * x));
            }, 0);

        // Write analytic solution to file
        fname.str(string(""));
        fname << "test_derivs_ddx1_solan_" << Nx << "_out.dat";
        of.open(fname.str());
        of << sol_an;
        of.close();

        //// compute first x-derivative
        my_slab.d_dx(test_ns::field_t::arr1, test_ns::field_t::arr3, 1, 0, 0);

        fname.str(string(""));
        fname << "test_derivs_ddx1_solnum_" << Nx << "_out.dat";
        my_slab.print_field(test_ns::field_t::arr3, fname.str());

        cuda_array_bc_nogp<my_allocator_device<cuda::real_t>> sol_num(my_slab.get_array_ptr(test_ns::field_t::arr3));
        sol_num -= sol_an;
        cout << "First x-derivative: sol_num - sol_an: Nx = " << Nx << ", My = " << My << ", L2 = " << sol_num.L2(0) << endl;

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Initialize analytic solution for second derivative
        sol_an.evaluate([=] __device__(size_t n, size_t m, cuda::slab_layout_t geom) -> cuda::real_t
            {
                cuda::real_t x{geom.get_xleft() + (cuda::real_t(n) + 0.5) * geom.get_deltax()};
                return(-1.0 * cuda::TWOPI * cuda::TWOPI * sin(cuda::TWOPI * x));
            }, 0);

        // Write analytic solution to file
        fname.str(string(""));
        fname << "test_derivs_ddx2_solan_" << Nx << "_out.dat";
        of.open(fname.str());
        of << sol_an;
        of.close();

        //// compute first x-derivative
        my_slab.d_dx(test_ns::field_t::arr1, test_ns::field_t::arr3, 2, 0, 0);

        fname.str(string(""));
        fname << "test_derivs_ddx2_solnum_" << Nx << "_out.dat";
        my_slab.print_field(test_ns::field_t::arr3, fname.str());

        sol_num = my_slab.get_array_ptr(test_ns::field_t::arr3);
        sol_num -= sol_an;
        cout << "Second x-derivative: sol_num - sol_an: Nx = " << Nx << ", My = " << My << ", L2 = " << sol_num.L2(0) << endl;

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Test first y-derivative
        // g_y = -100 * (y - 0.5) * exp(-50 * (y - 0.5) * (y - 0.5))
        sol_an.evaluate([=] __device__(size_t n, size_t m, cuda::slab_layout_t geom) -> cuda::real_t
            {
                //cuda::real_t x{geom.get_xleft() + (cuda::real_t(n) + 0.5) * geom.get_deltax()};
                cuda::real_t y{geom.get_ylo() + (cuda::real_t(m) + 0.5) * geom.get_deltay()};
                return(-100 * (y - 0.5) * exp(-50.0 * (y - 0.5) * (y - 0.5)));
            }, 0);

        // Write analytic solution to file
        fname.str(string(""));
        fname << "test_derivs_ddy1_solan_" << Nx << "_out.dat";
        of.open(fname.str());
        of << sol_an;
        of.close();

        //// compute first x-derivative
        my_slab.d_dy(test_ns::field_t::arr2, test_ns::field_t::arr3, 1, 0, 0);

        fname.str(string(""));
        fname << "test_derivs_ddy1_solnum_" << Nx << "_out.dat";
        my_slab.print_field(test_ns::field_t::arr3, fname.str());

        sol_num = my_slab.get_array_ptr(test_ns::field_t::arr3);
        sol_num -= sol_an;
        cout << "First y-derivative: sol_num - sol_an: Nx = " << Nx << ", My = " << My << ", L2 = " << sol_num.L2(0) << endl;
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Test second y-derivative
        // g_yy = 10000 * (0.24 - y + y^2) * exp(-50 * (y - 0.5) * (y - 0.5))
        sol_an.evaluate([=] __device__(size_t n, size_t m, cuda::slab_layout_t geom) -> cuda::real_t
            {
                cuda::real_t y{geom.get_ylo() + (cuda::real_t(m) + 0.5) * geom.get_deltay()};
                return(10000 * (0.24 - y + y * y) * exp(-50.0 * (y - 0.5) * (y - 0.5)));
            }, 0);

        // Write analytic solution to file
        fname.str(string(""));
        fname << "test_derivs_ddy2_solan_" << Nx << "_out.dat";
        of.open(fname.str());
        of << sol_an;
        of.close();

        //// compute first x-derivative
        my_slab.d_dy(test_ns::field_t::arr2, test_ns::field_t::arr3, 2, 0, 0);

        fname.str(string(""));
        fname << "test_derivs_ddy2_solnum_" << Nx << "_out.dat";
        my_slab.print_field(test_ns::field_t::arr3, fname.str());

        sol_num = my_slab.get_array_ptr(test_ns::field_t::arr3);
        sol_num -= sol_an;
        cout << "First y-derivative: sol_num - sol_an: Nx = " << Nx << ", My = " << My << ", L2 = " << sol_num.L2(0) << endl;

    }
}
