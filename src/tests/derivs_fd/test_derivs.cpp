/*
 * Test derivatives
 *
 * Input:
 *      f(x, y) = sin(2 * pi * x)
 * Derivatives:
 *      f_x = 2 * pi * cos(2 pi x)
 *      f_xx =- -4 * pi * sin(2 pi x)
 *
 *      g(x, y) = exp(-50 * (y-0.5) * (y-0.5))
 *      g_y = -100 * (y - 0.5) * exp(-50 * (y - 0.5) * (y - 0.5))
 *      g_yy = 10000 * (0.24 - y + y^2) * exp(-50 * (y - 0.5) * (y - 0.5))
 */

#include <iostream>
#include <sstream>
#include "slab_bc.h"

using namespace std;
using real_arr = cuda_array_bc_nogp<twodads::real_t, allocator_host>;
using deriv_t = deriv_fd_t<twodads::real_t, allocator_host>;

#if defined(HOST)
using dft_t = fftw_object_t<twodads::real_t>;
#endif //HOST

#if defined(DEVICE)
using dft_t = cufft_object_t<twodads::real_t>;
#endif //DEVICE


int main(void)
{
    const size_t t_src{0};
    slab_config_js my_config(std::string("input_test_derivatives_1.json"));
    {
        real_arr f(my_config.get_geom(), my_config.get_bvals(twodads::field_t::f_theta), 1);
        real_arr sol_an(my_config.get_geom(), my_config.get_bvals(twodads::field_t::f_theta), 1);
        real_arr sol_num(my_config.get_geom(), my_config.get_bvals(twodads::field_t::f_theta), 1);
        stringstream fname;

        deriv_t der(my_config.get_geom());
        dft_t dft(my_config.get_geom(), my_config.get_dft_t());

        f.apply([] (twodads::real_t dummy, size_t n, size_t m, twodads::slab_layout_t geom) -> twodads::real_t
            {       
                return(sin(twodads::TWOPI * geom.get_x(n)));
            }, t_src);

        // Initialize analytic solution for first derivative
        sol_an.apply([] (twodads::real_t dummy, size_t n, size_t m, twodads::slab_layout_t geom) -> twodads::real_t
            {
                return(twodads::TWOPI * cos(twodads::TWOPI * geom.get_x(n)));
            }, t_src);

        // Write analytic solution to file
        fname.str(string(""));
        fname << "test_derivs_ddx1_solan_" << my_config.get_nx() << ".dat";
        utility :: print(sol_an, t_src, fname.str());

        // compute first x-derivative
        der.dx(f, sol_num, t_src, t_src, 1);

        fname.str(string(""));
        #if defined(HOST)
        fname << "test_derivs_ddx1_solnum_" << my_config.get_nx() << "_host.dat";
        #endif
        #if defined(DEVICE)
        fname << "test_derivs_ddx1_solnum_" << my_config.get_nx() << "_device.dat";
        #endif

        utility :: print(sol_num, t_src, fname.str());

        //sol_num -= sol_an;
        sol_num.elementwise([] LAMBDACALLER (twodads::real_t lhs, twodads::real_t rhs) -> twodads::real_t
            {
                return(lhs - rhs);
            }, sol_an, 0, 0);
        cout << "First x-derivative: sol_num - sol_an: Nx = " << my_config.get_nx() << ", My = " << my_config.get_my() << ", L2 = " << utility :: L2(sol_num, t_src) << endl;

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Initialize analytic solution for second derivative
        sol_an.apply([=] (twodads::real_t dummy, size_t n, size_t m, twodads::slab_layout_t geom) -> twodads::real_t
            {
                return(-1.0 * twodads::TWOPI * twodads::TWOPI * sin(twodads::TWOPI * geom.get_x(n)));
            }, t_src);

        // Write analytic solution to file
        fname.str(string(""));
        fname << "test_derivs_ddx2_solan_" << my_config.get_nx() << "_device.dat";
        utility :: print(sol_an, t_src, fname.str());

        //// compute first x-derivative
        der.dx(f, sol_num, t_src, t_src, 2);

        fname.str(string(""));
        #if defined(HOST)
        fname << "test_derivs_ddx2_solnum_" << my_config.get_nx() << "_host.dat";
        #endif
        #if defined(DEVICE)
        fname << "test_derivs_ddx2_solnum_" << my_config.get_nx() << "_device.dat";
        #endif
        utility :: print(sol_num, t_src, fname.str());

        sol_num.elementwise([] LAMBDACALLER (twodads::real_t lhs, twodads::real_t rhs) -> twodads::real_t
            {
                return(lhs - rhs);
            }, sol_an, 0, 0);
        
        cout << "Second x-derivative: sol_num - sol_an: Nx = " << my_config.get_nx() << ", My = " << my_config.get_my() << ", L2 = " << utility :: L2(sol_num, t_src) << endl;


        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Test first y-derivative
        // g_y = -100 * (y - 0.5) * exp(-50 * (y - 0.5) * (y - 0.5))

        // Initialize input for numeric derivation and transform
        f.apply([] (twodads::real_t dummy, size_t n, size_t m, twodads::slab_layout_t geom) -> twodads::real_t
        {       
            twodads::real_t y{geom.get_y(m)};
            return(exp(-50.0 * y * y)); 
        }, t_src);

        std::cout << std::endl;
        utility :: print(f, t_src, std::cout);
        std::cout << std::endl;

        dft.dft_r2c(f.get_tlev_ptr(t_src), reinterpret_cast<twodads::cmplx_t*>(f.get_tlev_ptr(t_src)));
        f.set_transformed(t_src, true);

        std::cout << std::endl;
        utility :: print(f, t_src, std::cout);
        std::cout << std::endl;

        // Initialize analytic first derivative
        sol_an.apply([] (twodads::real_t dummy, size_t n, size_t m, twodads::slab_layout_t geom) -> twodads::real_t
            {
                twodads::real_t y{geom.get_y(m)};
                return(-100 * y * exp(-50.0 * y * y));
            }, t_src);

        // Write analytic solution to file
        fname.str(string(""));
        fname << "test_derivs_ddy1_solan_" << my_config.get_nx() << ".dat";
        utility :: print(sol_an, 0, fname.str());

        //// compute first y-derivative
        der.dy(f, sol_num, t_src, t_src, 1);
        dft.dft_c2r(reinterpret_cast<twodads::cmplx_t*>(sol_num.get_tlev_ptr(t_src)), sol_num.get_tlev_ptr(t_src));
        utility :: normalize(sol_num, t_src);
        sol_num.set_transformed(t_src, false);

        // Write numerical solution to file
        fname.str(string(""));
        #if defined(HOST)
        fname << "test_derivs_ddy1_solnum_" << my_config.get_nx() << "_host.dat";
        #endif
        #if defined(DEVICE)
        fname << "test_derivs_ddy1_solnum_" << my_config.get_nx() << "_device.dat";
        #endif
        utility :: print(sol_num, t_src, fname.str());

        sol_num.elementwise([] LAMBDACALLER (twodads::real_t lhs, twodads::real_t rhs) -> twodads::real_t
            {
                return(lhs - rhs);
            }, sol_an, 0, 0);
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
        fname << "test_derivs_ddy2_solan_" << my_config.get_nx() << ".dat";
        utility :: print(sol_an, t_src, fname.str());

        //// compute first x-derivative
        der.dy(f, sol_num, t_src, t_src, 2);
        dft.dft_c2r(reinterpret_cast<twodads::cmplx_t*>(sol_num.get_tlev_ptr(t_src)), sol_num.get_tlev_ptr(t_src));
        utility :: normalize(sol_num, t_src);
        sol_num.set_transformed(t_src, false);

        fname.str(string(""));
        #if defined(HOST)
        fname << "test_derivs_ddy2_solnum_" << my_config.get_nx() << "_host.dat";
        #endif
        #if defined(DEVICE)
        fname << "test_derivs_ddy2_solnum_" << my_config.get_nx() << "_device.dat";
        #endif
        utility :: print(sol_num, t_src, fname.str());

        sol_num.elementwise([] LAMBDACALLER (twodads::real_t lhs, twodads::real_t rhs) -> twodads::real_t
            {
                return(lhs - rhs);
            }, sol_an, 0, 0);
        cout << "Second y-derivative: sol_num - sol_an: Nx = " << my_config.get_nx() << ", My = " << my_config.get_my() << ", L2 = " << utility :: L2(sol_num, t_src) << endl;
    }
}
