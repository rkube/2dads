/**********************************************************************************
 * Test Poisson bracket computation with bispectral derivatives
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
 *     {f,g} = 16 pi^2 cos(pi x) cos(pi y) [-(cos(2 pi x) + cos(2 pi y))sin (pi x)^2 sin(pi y)^2
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
using my_derivs = deriv_spectral_t<twodads::real_t, allocator_host>;

int main(void)
{ 
    stringstream fname;
    slab_config_js my_config(std::string("input_test_pbrackets_bs.json"));
    my_derivs der(my_config.get_geom());
    fftw_object_t<twodads::real_t> myfft(my_config.get_geom(), my_config.get_dft_t());

    const size_t t_src{0};
    const size_t t_dst{0};

    // The bispectral derivative class requires the numerical derivatives as input
    // parameters. compute them directly and pass them.
    real_arr f(my_config.get_geom(), my_config.get_bvals(twodads::field_t::f_theta), 1);
    real_arr f_x(my_config.get_geom(), my_config.get_bvals(twodads::field_t::f_theta), 1);
    real_arr f_y(my_config.get_geom(), my_config.get_bvals(twodads::field_t::f_theta), 1);
    real_arr g(my_config.get_geom(), my_config.get_bvals(twodads::field_t::f_theta), 1);
    real_arr g_x(my_config.get_geom(), my_config.get_bvals(twodads::field_t::f_theta), 1);
    real_arr g_y(my_config.get_geom(), my_config.get_bvals(twodads::field_t::f_theta), 1);

    real_arr sol_num(my_config.get_geom(), my_config.get_bvals(twodads::field_t::f_theta), 1);
    real_arr sol_an(my_config.get_geom(), my_config.get_bvals(twodads::field_t::f_theta), 1);

    // Initialize input    
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

    // Initialize analytic solution
    sol_an.apply([] (twodads::real_t dummy, size_t n, size_t m, twodads::slab_layout_t geom) -> twodads::real_t 
            {
                twodads::real_t x{geom.get_x(n)};
                twodads::real_t y{geom.get_y(m)};
                return(16.0 * twodads::PI * twodads::PI * cos(twodads::PI * x) * cos(twodads::PI * y) * (cos(twodads::TWOPI * y) - cos(twodads::TWOPI * x)) * sin(twodads::PI * x) * sin(twodads::PI * x) * sin(twodads::PI * y) * sin(twodads::PI * y));
            }, 
            t_src);

    fname.str(string(""));
    fname << "test_pbrackets_solan_" << my_config.get_nx() << "_out.dat";
    utility :: print(sol_an, t_src, fname.str());

    fname.str(string(""));
    fname << "test_pbrackets_f_" << my_config.get_nx() << "_in.dat";
    utility :: print(f, t_src, fname.str());

    fname.str(string(""));
    fname << "test_pbrackets_g_" << my_config.get_nx() << "_in.dat";
    utility :: print(g, t_src, fname.str());

    /******************************************************************************************
     *                                                                                        *
     *                                Compute Poisson brackets                                *
     *                                                                                        *
     ******************************************************************************************/

    // Fourier transform
    myfft.dft_r2c(f.get_tlev_ptr(t_src), reinterpret_cast<twodads::cmplx_t*>(f.get_tlev_ptr(t_src)));
    f.set_transformed(t_src, true);
    myfft.dft_r2c(g.get_tlev_ptr(t_src), reinterpret_cast<twodads::cmplx_t*>(g.get_tlev_ptr(t_src)));
    g.set_transformed(t_src, true);

    // Compute derivatives
    der.dx(f, f_x, t_src, t_dst, 1);
    der.dy(f, f_y, t_src, t_dst, 1);
    der.dx(g, g_x, t_src, t_dst, 1);
    der.dy(g, g_y, t_src, t_dst, 1);
    // Inverse transformation of derivatives
    myfft.dft_c2r(reinterpret_cast<twodads::cmplx_t*>(f_x.get_tlev_ptr(t_src)), f_x.get_tlev_ptr(t_src));
    utility :: normalize(f_x, t_src);
    f_x.set_transformed(t_src, false);

    myfft.dft_c2r(reinterpret_cast<twodads::cmplx_t*>(f_y.get_tlev_ptr(t_src)), f_y.get_tlev_ptr(t_src));
    utility :: normalize(f_y, t_src);
    f_y.set_transformed(t_src, false);

    myfft.dft_c2r(reinterpret_cast<twodads::cmplx_t*>(g_x.get_tlev_ptr(t_src)), g_x.get_tlev_ptr(t_src));
    utility :: normalize(g_x, t_src);
    g_x.set_transformed(t_src, false);

    myfft.dft_c2r(reinterpret_cast<twodads::cmplx_t*>(g_y.get_tlev_ptr(t_src)), g_y.get_tlev_ptr(t_src));
    utility :: normalize(g_y, t_src);
    g_y.set_transformed(t_src, false);

    // Compute poisson brackets
    der.pbracket(f_x, f_y, g_x, g_y, sol_num, t_src, t_src, t_dst);

    fname.str(string(""));
    fname << "test_pbrackets_solnum_" << my_config.get_nx() << "_out.dat";
    utility :: print(sol_num, t_src, fname.str());
    
    sol_num.elementwise([] LAMBDACALLER (twodads::real_t lhs, twodads::real_t rhs) -> twodads::real_t
        {
            return(lhs - rhs);
        }, sol_an, t_src, t_src);

    cout << "sol_num - sol_an: Nx = " << my_config.get_nx() << ", My = " << my_config.get_my() << ", L2 = " << utility :: L2(sol_num, t_src) << endl;

    fname.str(string(""));
    fname << "test_pbrackets_diff_" << my_config.get_nx() << "_out.dat";
    utility :: print(sol_num, t_src, fname.str());
}

// End of file test_arakawa.cpp