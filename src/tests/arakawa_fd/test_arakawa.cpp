/**********************************************************************************
 * Test Arakawa bracket operators
 * Compute {f,g} = f_x * g_y -f_y * g_x 
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
 *     {f,g} = 16 pi^2 cos(pi x) cos(pi y) [(cos(2 pi y) - cos(2 pi x)] sin (pi x)^2 sin(pi y)^2
 *     -> stored in arr3
 **********************************************************************************/

#include <iostream>
#include <sstream>
#include "cuda_array_bc_nogp.h"
#include "utility.h"
#include "derivatives.h"
#include "slab_config.h"

using namespace std;
using real_arr = cuda_array_bc_nogp<twodads::real_t, allocator_host>;
using my_derivs = deriv_fd_t<twodads::real_t, allocator_host>;

int main(void)
{ 
    stringstream fname;
    slab_config_js my_config(std::string("input_test_arakawa_fd.json"));
    my_derivs der(my_config.get_geom());

    const size_t t_src{0};
    const size_t t_dst{0};

    real_arr f(my_config.get_geom(), my_config.get_bvals(twodads::field_t::f_theta), 1);
    real_arr g(my_config.get_geom(), my_config.get_bvals(twodads::field_t::f_theta), 1);
    real_arr sol_num(my_config.get_geom(), my_config.get_bvals(twodads::field_t::f_theta), 1);
    real_arr sol_an(my_config.get_geom(), my_config.get_bvals(twodads::field_t::f_theta), 1);
    

    f.apply([] (twodads::real_t dummy, size_t n, size_t m, twodads::slab_layout_t geom) -> twodads::real_t
        {
            twodads::real_t x{geom.get_x(n)};
            twodads::real_t y{geom.get_y(m)};
            return (-1.0 * sin(twodads::TWOPI * x) * sin(twodads::TWOPI * x) * sin(twodads::TWOPI * y) * sin(twodads::TWOPI * y));
        }, t_src);

    g.apply([] (twodads::real_t dummy, size_t n, size_t m, twodads::slab_layout_t geom) -> twodads::real_t
        {
            twodads::real_t x{geom.get_x(n)};
            twodads::real_t y{geom.get_y(m)};
            return (sin(twodads::PI * x) * sin(twodads::PI * y));
        }, t_src);

    sol_an.apply([] (twodads::real_t dummy, size_t n, size_t m, twodads::slab_layout_t geom) -> twodads::real_t 
            {
                twodads::real_t x{geom.get_x(n)};
                twodads::real_t y{geom.get_y(m)};
                return(16.0 * twodads::PI * twodads::PI * cos(twodads::PI * x) * cos(twodads::PI * y) * (cos(twodads::TWOPI * y) - cos(twodads::TWOPI * x)) * sin(twodads::PI * x) * sin(twodads::PI * x) * sin(twodads::PI * y) * sin(twodads::PI * y));
            }, 
            t_src);

    fname.str(string(""));
    fname << "test_arakawa_solan_" << my_config.get_nx() << "_out.dat";
    utility :: print(sol_an, t_src, fname.str());

    fname.str(string(""));
    fname << "test_arakawa_f_" << my_config.get_nx() << "_in.dat";
    utility :: print(f, t_src, fname.str());

    fname.str(string(""));
    fname << "test_arakawa_g_" << my_config.get_nx() << "_in.dat";
    utility :: print(g, t_src, fname.str());

    der.pbracket(f, g, sol_num, t_src, t_src, t_dst);

    fname.str(string(""));
    fname << "test_arakawa_solnum_" << my_config.get_nx() << "_out.dat";
    utility :: print(sol_num, t_src, fname.str());
    
    sol_num.elementwise([] LAMBDACALLER (twodads::real_t lhs, twodads::real_t rhs) -> twodads::real_t
        {
            return(lhs - rhs);
        }, sol_an, t_src, t_src);

    cout << "sol_num - sol_an: Nx = " << my_config.get_nx() << ", My = " << my_config.get_my() << ", L2 = " << utility :: L2(sol_num, t_src) << endl;

    fname.str(string(""));
    fname << "test_arakawa_diff_" << my_config.get_nx() << "_out.dat";
    utility :: print(sol_num, t_src, fname.str());
}

// End of file test_arakawa.cpp