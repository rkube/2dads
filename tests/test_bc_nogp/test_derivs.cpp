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
#include <sstream>
#include "2dads_types.h"
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

    constexpr twodads::real_t x_l{-1.0};
    constexpr twodads::real_t Lx{2.0};
    constexpr twodads::real_t y_l{-1.0};
    constexpr twodads::real_t Ly{2.0};
    constexpr twodads::real_t deltat{0.1};
    constexpr twodads::real_t diff{0.1};
    constexpr twodads::real_t hv{0};

    twodads::bvals_t<double> my_bvals{twodads::bc_t::bc_dirichlet, twodads::bc_t::bc_dirichlet, twodads::bc_t::bc_periodic, twodads::bc_t::bc_periodic, 0.0, 0.0, 0.0, 0.0};
    twodads::slab_layout_t my_geom(x_l, (Lx - x_l) / twodads::real_t(Nx), y_l, (Ly - y_l)  / twodads::real_t(My), Nx, 0, My, 2, twodads::grid_t::cell_centered);
    twodads::stiff_params_t stiff_params(deltat, Lx, Ly, diff, hv, my_geom.get_nx(), (my_geom.get_my() + my_geom.get_pad_y()) / 2, 1);

    {
        slab_bc my_slab(my_geom, my_bvals, stiff_params);
        my_slab.initialize_derivatives(twodads::field_t::f_theta, twodads::field_t::f_theta, 0);

        // Initialize analytic solution for first derivative
        cuda_array_bc_nogp<twodads::real_t, allocator_host> sol_an(my_geom, my_bvals, 1);
        sol_an.apply([=] (twodads::real_t dummy, size_t n, size_t m, twodads::slab_layout_t geom) -> twodads::real_t
            {
                return(twodads::TWOPI * cos(twodads::TWOPI * geom.get_x(n)));
            }, 0);

        // Write analytic solution to file
        fname.str(string(""));
        fname << "test_derivs_ddx1_solan_" << Nx << "_out.dat";
        utility :: print(sol_an, 0, fname.str());

        // compute first x-derivative
        my_slab.d_dx(test_ns::field_t::arr1, test_ns::field_t::arr3, 1, 0, 0);

        fname.str(string(""));
        fname << "test_derivs_ddx1_solnum_" << Nx << "_out.dat";
        utility :: print((*my_slab.get_array_ptr(test_ns::field_t::arr3)), 0, fname.str());

        cuda_array_bc_nogp<twodads::real_t, allocator_host> sol_num(my_slab.get_array_ptr(test_ns::field_t::arr3));
        sol_num -= sol_an;
        cout << "First x-derivative: sol_num - sol_an: Nx = " << Nx << ", My = " << My << ", L2 = " << utility :: L2(sol_num, 0) << endl;

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Initialize analytic solution for second derivative
        sol_an.apply([=] (twodads::real_t dummy, size_t n, size_t m, twodads::slab_layout_t geom) -> twodads::real_t
            {
                return(-1.0 * twodads::TWOPI * twodads::TWOPI * sin(twodads::TWOPI * geom.get_x(n)));
            }, 0);

        // Write analytic solution to file
        fname.str(string(""));
        fname << "test_derivs_ddx2_solan_" << Nx << "_out.dat";
        //of.open(fname.str());
        utility :: print(sol_an, 0, fname.str());

        //// compute first x-derivative
        my_slab.d_dx(test_ns::field_t::arr1, test_ns::field_t::arr3, 2, 0, 0);

        fname.str(string(""));
        fname << "test_derivs_ddx2_solnum_" << Nx << "_out.dat";
        utility :: print((*my_slab.get_array_ptr(test_ns::field_t::arr3)), 0, fname.str());

        sol_num = my_slab.get_array_ptr(test_ns::field_t::arr3);
        sol_num -= sol_an;
        cout << "Second x-derivative: sol_num - sol_an: Nx = " << Nx << ", My = " << My << ", L2 = " << utility :: L2(sol_num, 0) << endl;

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Test first y-derivative
        // g_y = -100 * (y - 0.5) * exp(-50 * (y - 0.5) * (y - 0.5))
        sol_an.apply([] (twodads::real_t dummy, size_t n, size_t m, twodads::slab_layout_t geom) -> twodads::real_t
            {
                twodads::real_t y{geom.get_y(m)};
                return(-100 * y * exp(-50.0 * y * y));
            }, 0);

        // Write analytic solution to file
        fname.str(string(""));
        fname << "test_derivs_ddy1_solan_" << Nx << "_out.dat";
        utility :: print(sol_an, 0, fname.str());

        //// compute first y-derivative
        my_slab.d_dy(test_ns::field_t::arr2, test_ns::field_t::arr3, 1, 0, 0);

        fname.str(string(""));
        fname << "test_derivs_ddy1_solnum_" << Nx << "_out.dat";
        utility :: print((*my_slab.get_array_ptr(test_ns::field_t::arr3)), 0, fname.str());

        sol_num = my_slab.get_array_ptr(test_ns::field_t::arr3);
        sol_num -= sol_an;
        cout << "First y-derivative: sol_num - sol_an: Nx = " << Nx << ", My = " << My << ", L2 = " << utility :: L2(sol_num, 0) << endl;
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Test second y-derivative
        // g_yy = 10000 * (0.24 - y + y^2) * exp(-50 * (y - 0.5) * (y - 0.5))
        sol_an.apply([] (twodads::real_t dummy, size_t n, size_t m, twodads::slab_layout_t geom) -> twodads::real_t
            {
                twodads::real_t y{geom.get_y(m)};
                return(100 * (-1.0 + 100 * y * y) * exp(-50.0 * y * y));
            }, t_src);

        // Write analytic solution to file
        fname.str(string(""));
        fname << "test_derivs_ddy2_solan_" << Nx << "_out.dat";
        utility :: print(sol_an, 0, fname.str());

        //// compute first x-derivative
        my_slab.d_dy(test_ns::field_t::arr2, test_ns::field_t::arr3, 2, 0, 0);

        fname.str(string(""));
        fname << "test_derivs_ddy2_solnum_" << Nx << "_out.dat";
        utility :: print((*my_slab.get_array_ptr(test_ns::field_t::arr3)), 0, fname.str());

        sol_num = my_slab.get_array_ptr(test_ns::field_t::arr3);
        sol_num -= sol_an;
        cout << "First y-derivative: sol_num - sol_an: Nx = " << Nx << ", My = " << My << ", L2 = " << utility :: L2(sol_num, 0) << endl;
    }
}
