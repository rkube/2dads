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
#include <fstream>
#include <sstream>
#include "slab_bc.h"

using namespace std;


int main(void){
    size_t Nx{128};
    size_t My{128};
    cout << "Enter Nx: ";
    cin >> Nx;
    cout << "Enter My: ";
    cin >> My;

    stringstream fname;
    ofstream of;

    cuda::slab_layout_t my_geom(-1.0, 2.0 / double(Nx), -1.0, 2.0 / double(My), Nx, 0, My, 2);
    cuda::bvals_t<double> my_bvals{cuda::bc_t::bc_dirichlet, cuda::bc_t::bc_dirichlet, cuda::bc_t::bc_periodic, cuda::bc_t::bc_periodic,
        0.0, 0.0, 0.0, 0.0};

    {
        slab_bc my_slab(my_geom, my_bvals);
        cuda_array_bc_nogp<my_allocator_device<cuda::real_t>> sol_an(my_geom, my_bvals, 1);
        sol_an.evaluate([=] __device__ (size_t n, size_t m, cuda::slab_layout_t geom) -> cuda::real_t
                {
                    cuda::real_t x{geom.get_xleft() + (cuda::real_t(n) + 0.5) * geom.get_deltax()};
                    cuda::real_t y{geom.get_ylo() + (cuda::real_t(m) + 0.5) * geom.get_deltay()};
                    return(16.0 * cuda::PI * cuda::PI * cos(cuda::PI * x) * cos(cuda::PI * y) * (cos(cuda::TWOPI * x) - cos(cuda::TWOPI * y)) * sin(cuda::PI * x) * sin(cuda::PI * x) * sin(cuda::PI * y) * sin(cuda::PI * y));
                }, 
                0);

        fname.str(string(""));
        fname << "test_arakawa_solan_" << Nx << "_out.dat";
        of.open(fname.str());
        of << sol_an;
        of.close();

        //cerr << "Initializing fields..." << endl;
        my_slab.initialize_arakawa(test_ns::field_t::arr1, test_ns::field_t::arr2);
        // Print input to inv_laplace routine into array arr1_nx.dat
        fname.str(string(""));
        fname << "test_arakawa_f_" << Nx << "_in.dat";
        my_slab.print_field(test_ns::field_t::arr1, fname.str());

        fname.str(string(""));
        fname << "test_arakawa_g_" << Nx << "_in.dat";
        my_slab.print_field(test_ns::field_t::arr2, fname.str());

        my_slab.arakawa(test_ns::field_t::arr1, test_ns::field_t::arr2, test_ns::field_t::arr3, size_t(0), size_t(0));

        fname.str(string(""));
        fname << "test_arakawa_solnum_" << Nx << "_out.dat";
        my_slab.print_field(test_ns::field_t::arr3, fname.str());
       
        cuda_array_bc_nogp<my_allocator_device<cuda::real_t>> sol_num(my_slab.get_array_ptr(test_ns::field_t::arr3));
        sol_num -= sol_an;

        cout << "sol_num - sol_an: Nx = " << Nx << ", My = " << My << ", L2 = " << sol_num.L2(0) << endl;

        fname.str(string(""));
        fname << "test_arakawa_diff_" << Nx << "_out.dat";
        of.open(fname.str());
        of << sol_num;
        of.close();
    }
    cudaDeviceReset();
}

