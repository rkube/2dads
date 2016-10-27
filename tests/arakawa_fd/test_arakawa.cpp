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
using real_arr = cuda_array_bc_nogp<twodads::real_t, allocator_host>;

int main(void)
{ 
    stringstream fname;
    slab_config_js my_config(std::string("input_test_arakawa_fd.json"));
    const size_t t_src{my_config.get_tlevs() - 1};
    {
        slab_bc my_slab(my_config);
        my_slab.initialize();

        real_arr sol_an(my_config.get_geom(), my_config.get_bvals(twodads::field_t::f_theta), 1);

        sol_an.apply([] (twodads::real_t dummy, size_t n, size_t m, twodads::slab_layout_t geom) -> twodads::real_t 
                {
                    twodads::real_t x{geom.get_x(n)};
                    twodads::real_t y{geom.get_y(m)};
                    return(16.0 * twodads::PI * twodads::PI * cos(twodads::PI * x) * cos(twodads::PI * y) * (cos(twodads::TWOPI * x) - cos(twodads::TWOPI * y)) * sin(twodads::PI * x) * sin(twodads::PI * x) * sin(twodads::PI * y) * sin(twodads::PI * y));
                }, 
                0);

        fname.str(string(""));
        fname << "test_arakawa_solan_" << my_config.get_nx() << "_out.dat";
        utility :: print(sol_an, 0, fname.str());

        fname.str(string(""));
        fname << "test_arakawa_f_" << my_config.get_nx() << "_in.dat";
        utility :: print((*my_slab.get_array_ptr(twodads::field_t::f_theta)), t_src, fname.str());

        fname.str(string(""));
        fname << "test_arakawa_g_" << my_config.get_nx() << "_in.dat";
        utility :: print((*my_slab.get_array_ptr(twodads::field_t::f_omega)), t_src, fname.str());

        my_slab.pbracket(twodads::field_t::f_theta, twodads::field_t::f_omega, twodads::field_t::f_strmf, t_src, t_src, 0);

        fname.str(string(""));
        fname << "test_arakawa_solnum_" << my_config.get_nx() << "_out.dat";
        utility :: print((*my_slab.get_array_ptr(twodads::field_t::f_strmf)), 0, fname.str());
       
        real_arr sol_num(my_slab.get_array_ptr(twodads::field_t::f_strmf));
        sol_num -= sol_an;

        cout << "sol_num - sol_an: Nx = " << my_config.get_nx() << ", My = " << my_config.get_my() << ", L2 = " << utility :: L2(sol_num, 0) << endl;

        fname.str(string(""));
        fname << "test_arakawa_diff_" << my_config.get_nx() << "_out.dat";
        utility :: print(sol_num, 0, fname.str());
    }
}

// End of file test_arakawa.cpp