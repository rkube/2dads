/*
 * Test Arakawa bracket operators
 * Compute {f,g} = -f_y g_x + g_y f_x
 * 
 * Input:
 *     f(x, y) = -sin(2 pi x)^2 sin(2 pi y)^2
 *     f_x = -4 pi (cos 2 pi x)sin(2 pi x) sin(2 pi y)^2
 *     f_y = -4 pi(cos 2 pi y) sin(2 pi y) sin(2 pi x)^2
 *     -> initializes arr1
 * 
 *     g(x, y) = sin(pi x) sin(pi y)
 *     g_x = pi cos(pi x) sin(pi y)
 *     g_y = pi sin(pi x) cos(pi y)
 *     -> initializes arr2
 *
 * Output
 *     {f,g} = 16 pi^2 cos(pi x) cos(pi y) [-(cos(2 pi x) + cos(2 pi y))sin (pi x)^2 sin(pi y)^2
 *     -> stored in arr3
 *
 *
 */

#include <iostream>
#include <sstream>
#include "slab_bc.h"

using namespace std;


int main(void){
    constexpr twodads::real_t x_l{-1.0};
    constexpr twodads::real_t Lx{2.0};
    constexpr twodads::real_t y_l{-1.0};
    constexpr twodads::real_t Ly{2.0};
    constexpr size_t tlevs{1};
    constexpr size_t t_src{0};
    constexpr twodads::real_t diff{0.1};
    constexpr twodads::real_t hv{0.0};

    size_t Nx{128};
    size_t My{128};
    cout << "Enter Nx: ";
    cin >> Nx;
    cout << "Enter My: ";
    cin >> My;

    stringstream fname;
    ofstream of;

    twodads::slab_layout_t my_geom(x_l, Lx / twodads::real_t(Nx), y_l, Ly / twodads::real_t(My), Nx, 0, My, 2, twodads::grid_t::cell_centered);
    twodads::bvals_t<double> my_bvals{twodads::bc_t::bc_dirichlet, twodads::bc_t::bc_dirichlet, twodads::bc_t::bc_periodic, twodads::bc_t::bc_periodic,
        0.0, 0.0, 0.0, 0.0};
    twodads::stiff_params_t stiff_params(0.1, Lx, Ly, diff, hv, Nx, My / 2 + 1, tlevs);

    {
        slab_bc my_slab(my_geom, my_bvals, stiff_params);
        cuda_array_bc_nogp<twodads::real_t, allocator_host> sol_an(my_geom, my_bvals, tlevs);
        sol_an.apply([] (twodads::real_t dummy, size_t n, size_t m, twodads::slab_layout_t geom) -> twodads::real_t 
                {
                    twodads::real_t x{geom.get_x(n)};
                    twodads::real_t y{geom.get_y(m)};
                    return(16.0 * twodads::PI * twodads::PI * cos(twodads::PI * x) * cos(twodads::PI * y) * (cos(twodads::TWOPI * x) - cos(twodads::TWOPI * y)) * sin(twodads::PI * x) * sin(twodads::PI * x) * sin(twodads::PI * y) * sin(twodads::PI * y));
                }, 
                t_src);

        fname.str(string(""));
        fname << "test_arakawa_solan_" << Nx << "_out.dat";
        utility :: print(sol_an, t_src, fname.str());

        //cerr << "Initializing fields..." << endl;
        my_slab.initialize_arakawa(test_ns::field_t::arr1, test_ns::field_t::arr2, t_src);
        // Print input to inv_laplace routine into array arr1_nx.dat
        fname.str(string(""));
        fname << "test_arakawa_f_" << Nx << "_in.dat";
        utility :: print((*my_slab.get_array_ptr(test_ns::field_t::arr1)), t_src, fname.str());

        fname.str(string(""));
        fname << "test_arakawa_g_" << Nx << "_in.dat";
        utility :: print((*my_slab.get_array_ptr(test_ns::field_t::arr2)), t_src, fname.str());

        my_slab.arakawa(test_ns::field_t::arr1, test_ns::field_t::arr2, test_ns::field_t::arr3, t_src, t_src);

        fname.str(string(""));
        fname << "test_arakawa_solnum_" << Nx << "_out.dat";
        utility :: print((*my_slab.get_array_ptr(test_ns::field_t::arr3)), t_src, fname.str());
       
        cuda_array_bc_nogp<twodads::real_t, allocator_host> sol_num(my_slab.get_array_ptr(test_ns::field_t::arr3));
        sol_num -= sol_an;

        cout << "sol_num - sol_an: Nx = " << Nx << ", My = " << My << ", L2 = " << utility :: L2(sol_num, t_src) << endl;

        fname.str(string(""));
        fname << "test_arakawa_diff_" << Nx << "_out.dat";
        utility :: print(sol_num, t_src, fname.str());
    }
}

// End of file test_arakawa.cpp