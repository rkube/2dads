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
using real_arr = cuda_array_bc_nogp<twodads::real_t, allocator_host>;
using deriv_t = deriv_spectral_t<twodads::real_t, allocator_host>;
using fft_t = fftw_object_t<twodads::real_t>;

int main(void)
{
    const size_t t_src{0};
    slab_config_js my_config(std::string("input_test_laplace_bs.json"));
    deriv_t der{my_config.get_geom()};
    fft_t myfft{my_config.get_geom(), my_config.get_dft_t()};

    real_arr f(my_config.get_geom(), my_config.get_bvals(twodads::field_t::f_theta), 1);
    real_arr sol_an(my_config.get_geom(), my_config.get_bvals(twodads::field_t::f_theta), 1);
    real_arr sol_num(my_config.get_geom(), my_config.get_bvals(twodads::field_t::f_theta), 1);
    
    stringstream fname;

    // Initialize input for laplace solver
    f.apply([] (twodads::real_t dummy, size_t n, size_t m, twodads::slab_layout_t geom) -> twodads::real_t
    {
        const twodads::real_t x{geom.get_x(n)};
        const twodads::real_t y{geom.get_y(m)};
        return(exp(-0.5 * (x * x + y * y)) * (-2.0 + x * x + y * y));
    }, t_src);
    myfft.dft_r2c(f.get_tlev_ptr(0), reinterpret_cast<twodads::cmplx_t*>(f.get_tlev_ptr(0)));
    f.set_transformed(t_src, true);
    sol_num.set_transformed(t_src, true);
    der.invert_laplace(f, sol_num, t_src, t_src);
    myfft.dft_c2r(reinterpret_cast<twodads::cmplx_t*>(sol_num.get_tlev_ptr(0)), sol_num.get_tlev_ptr(0));
    utility :: normalize(sol_num, t_src);
    sol_num.set_transformed(t_src, false);

    // Write numerical solution to file
    fname.str(string(""));
    fname << "test_laplace_solnum_" << my_config.get_nx() << "_host.dat";
    utility :: print(sol_num, t_src, fname.str());

    // Initialize analytic solution
    sol_an.apply([] LAMBDACALLER (twodads::real_t dummy, size_t n, size_t m, twodads::slab_layout_t geom) -> twodads::real_t
            {
                const twodads::real_t x{geom.get_x(n)};
                const twodads::real_t y{geom.get_y(m)};
                return(exp(-0.5 * (x * x + y * y)));
            },
    t_src);
    fname.str(string(""));
    fname << "test_laplace_solan_" << my_config.get_nx() << "_host.dat";
    utility :: print(sol_an, t_src, fname.str());
    // The spectral solver returns an array without mean. 
    // Find the mean of the analytic solution and subtract it.
    const twodads::real_t mean{utility :: mean(sol_an, t_src)};
    sol_an.elementwise([=] LAMBDACALLER (twodads::real_t lhs, twodads::real_t rhs) -> twodads::real_t
    {
        return(lhs - mean);
    }, 0, 0);

    //sol_an -= my_slab.get_array_ptr(twodads::field_t::f_strmf);
    sol_num.elementwise([] LAMBDACALLER (twodads::real_t lhs, twodads::real_t rhs) -> twodads::real_t
    {
        return(lhs - rhs);
    }, sol_an, 0, 0);

    cout << "Nx = " << my_config.get_nx() << ", My = " << my_config.get_my() << ", L2 = " << utility :: L2(sol_num, t_src) << endl;
}