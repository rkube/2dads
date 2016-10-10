/*
 * Test bispectral derivatives
 *
 * Input:
 *      f(x, y) = exp(-50 * x * x) 
 *      f_x = -100 * x * exp(-50 * x * x)
 *      f_xx = 100 * (-1.0 + 100 * x * x) * exp(-50.0 * x * x)
 *      -> Initializes arr1
 *
 *      g(x, y) = exp(-50 * y * y)
 *      g_y = -100 * y * exp(-50 * y * y)
 *      g_yy = 100 * (-1.0 + 100 * y * y) * exp(-50.0 * y * y)
 *      -> Initializes arr2
 */

#include <iostream>
#include <sstream>
#include "slab_bc.h"
#include "output.h"

using namespace std;
using real_arr = cuda_array_bc_nogp<twodads::real_t, allocator_host>;

int main(void)
{
    const size_t t_src{0};

    slab_config_js my_config(std::string("input_test_derivatives_spectral_1.json"));
    {
        slab_bc my_slab(my_config);  
        real_arr sol_an(my_config.get_geom(), my_config.get_bvals(twodads::field_t::f_theta), 1);
        real_arr sol_num(my_config.get_geom(), my_config.get_bvals(twodads::field_t::f_theta), 1);
        stringstream fname;
     
        real_arr* arr_ptr{my_slab.get_array_ptr(twodads::field_t::f_theta)};
        (*arr_ptr).apply([] (twodads::real_t dummy, size_t n, size_t m, twodads::slab_layout_t geom) -> twodads::real_t
        {       
            twodads::real_t x{geom.get_x(n)};
            return(exp(-50. * x * x));
        }, t_src);

        // Initialize analytic solution for first derivative
        sol_an.apply([] (twodads::real_t dummy, size_t n, size_t m, twodads::slab_layout_t geom) -> twodads::real_t
            {
                twodads::real_t x{geom.get_x(n)};
                return(-100 * x * exp(-50.0 * x * x));
            }, t_src);

        // Write analytic solution to file
        fname.str(string(""));
        fname << "test_derivs_bispectral_ddx1_solan_" << my_config.get_nx() << "_device.dat";
        utility :: print(sol_an, t_src, fname.str());

        // compute first x-derivative
        my_slab.d_dx(twodads::field_t::f_theta, twodads::field_t::f_theta_x, 1, t_src, t_src);
        sol_num = my_slab.get_array_ptr(twodads::field_t::f_theta_x);

        fname.str(string(""));
        fname << "test_derivs_bispectral_ddx1_solnum_" << my_config.get_nx() << "_device.dat";
        utility :: print(sol_num, t_src, fname.str());

        sol_num -= sol_an;
        cout << "First x-derivative: sol_num - sol_an: Nx = " << my_config.get_nx() << ", My = " << my_config.get_my() << ", L2 = " << utility :: L2(sol_num, t_src) << endl;

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Initialize analytic solution for second derivative
        sol_an.apply([] (twodads::real_t dummy, size_t n, size_t m, twodads::slab_layout_t geom) -> twodads::real_t
            {
                twodads::real_t x{geom.get_x(n)};
                return(100 * (-1.0 + 100 * x * x) * exp(-50.0 * x * x));
            }, t_src);

        // Write analytic solution to file
        fname.str(string(""));
        fname << "test_derivs_bispectral_ddx2_solan_" << my_config.get_nx() << "_device.dat";
        //of.open(fname.str());
        utility :: print(sol_an, t_src, fname.str());

        //// compute first x-derivative
        my_slab.d_dx(twodads::field_t::f_theta, twodads::field_t::f_theta_x, 2, t_src, t_src);
        sol_num = my_slab.get_array_ptr(twodads::field_t::f_theta_x);

        fname.str(string(""));
        fname << "test_derivs_bispectral_ddx2_solnum_" << my_config.get_nx() << "_device.dat";
        utility :: print(sol_num, t_src, fname.str());

        sol_num -= sol_an;
        cout << "Second x-derivative: sol_num - sol_an: Nx = " << my_config.get_nx() << ", My = " << my_config.get_my() << ", L2 = " << utility :: L2(sol_num, t_src) << endl;

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Test first y-derivative
        // g_y = -100 * (y - 0.5) * exp(-50 * (y - 0.5) * (y - 0.5))
        // Initialize input for numeric derivation
        (*arr_ptr).apply([] (twodads::real_t dummy, size_t n, size_t m, twodads::slab_layout_t geom) -> twodads::real_t
        {       
            twodads::real_t y{geom.get_y(m)};
            return(exp(-50.0 * y * y)); 
        }, t_src);
  
        // Initialize analytic first derivative
        sol_an.apply([] (twodads::real_t dummy, size_t n, size_t m, twodads::slab_layout_t geom) -> twodads::real_t
            {
                twodads::real_t y{geom.get_y(m)};
                return(-100 * y * exp(-50.0 * y * y));
            }, t_src);

        // Write analytic solution to file
        fname.str(string(""));
        fname << "test_derivs_bispectral_ddy1_solan_" << my_config.get_nx() << "_device.dat";
        utility :: print(sol_an, 0, fname.str());

        //// compute first y-derivative
        my_slab.d_dy(twodads::field_t::f_theta, twodads::field_t::f_theta_y, 1, t_src, t_src);
        sol_num = my_slab.get_array_ptr(twodads::field_t::f_theta_y);

        fname.str(string(""));
        fname << "test_derivs_bispectral_ddy1_solnum_" << my_config.get_nx() << "_device.dat";
        utility :: print((*my_slab.get_array_ptr(twodads::field_t::f_theta_y)), t_src, fname.str());

        sol_num -= sol_an;
        cout << "First y-derivative: sol_num - sol_an: Nx = " << my_config.get_nx() << ", My = " << my_config.get_my() << ", L2 = " << utility :: L2(sol_num, t_src) << endl;

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
        fname << "test_derivs_bispectral_ddy2_solan_" << my_config.get_nx() << "_device.dat";
        utility :: print(sol_an, t_src, fname.str());

        //// compute first x-derivative
        my_slab.d_dy(twodads::field_t::f_theta, twodads::field_t::f_theta_y, 2, t_src, t_src);
        sol_num = my_slab.get_array_ptr(twodads::field_t::f_theta_y);

        fname.str(string(""));
        fname << "test_derivs_bispectral_ddy2_solnum_" << my_config.get_nx() << "_device.dat";
        utility :: print(sol_num, t_src, fname.str());

        sol_num -= sol_an;
        cout << "First y-derivative: sol_num - sol_an: Nx = " << my_config.get_nx() << ", My = " << my_config.get_my() << ", L2 = " << utility :: L2(sol_num, t_src) << endl;
    }
}