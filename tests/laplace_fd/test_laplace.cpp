/*
 * Invert the laplace equation with boundary conditions in x using the cusparse tridiagonal solver 
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
using dft_t = fftw_object_t<twodads::real_t>;
using deriv_t = deriv_fd_t<twodads::real_t, allocator_host>;

int main(void)
{
    const size_t t_src{0};
    slab_config_js my_config(std::string("input_test_laplace_fd.json"));
    deriv_t der(my_config.get_geom());
    dft_t dft(my_config.get_geom(), my_config.get_dft_t());

    {
        // Analytic solution
        real_arr sol_an(my_config.get_geom(), my_config.get_bvals(twodads::field_t::f_theta), 1);
        // Input array
        real_arr input(my_config.get_geom(), my_config.get_bvals(twodads::field_t::f_theta), 1);
        // numerical solution
        real_arr sol_num(my_config.get_geom(), my_config.get_bvals(twodads::field_t::f_theta), 1);
        stringstream fname;

        // Initialize input for laplace solver
        input.apply([] (twodads::real_t dummy, size_t n, size_t m, twodads::slab_layout_t geom) -> twodads::real_t
        {
            const twodads::real_t x{geom.get_x(n)};
            const twodads::real_t y{geom.get_y(m)};
            return(exp(-0.5 * (x * x + y * y)) * (-2.0 + x * x + y * y));
        }, t_src);

        // Initialize analytic solution
        sol_an.apply([] (twodads::real_t dummy, size_t n, size_t m, twodads::slab_layout_t geom) -> twodads::real_t
                {
                    const twodads::real_t x{geom.get_x(n)};
                    const twodads::real_t y{geom.get_y(m)};
                    return(exp(-0.5 * (x * x + y * y)));
                },
        t_src);

        fname.str(string(""));
        fname << "test_laplace_solan_" << my_config.get_nx() << "_host.dat";
        utility :: print(sol_an, t_src, fname.str());

        dft.dft_r2c(input.get_tlev_ptr(t_src), reinterpret_cast<twodads::cmplx_t*>(input.get_tlev_ptr(t_src)));
        input.set_transformed(t_src, true);

        der.invert_laplace(input, sol_num, t_src, t_src);

        dft.dft_c2r(reinterpret_cast<twodads::cmplx_t*>(sol_num.get_tlev_ptr(t_src)), sol_num.get_tlev_ptr(t_src));
        utility :: normalize(sol_num, t_src);
        sol_num.set_transformed(t_src, false);

        // Write numerical solution to file
        fname.str(string(""));
        fname << "test_laplace_solnum_" << my_config.get_nx() << "_host.dat";
        utility :: print(sol_num, t_src, fname.str());

        // Get the analytic solution
        sol_num.elementwise([] LAMBDACALLER (twodads::real_t lhs, twodads::real_t rhs) -> twodads::real_t
            {
                return(lhs - rhs);
            }, sol_an, t_src, t_src);
        cout << "Nx = " << my_config.get_nx() << ", My = " << my_config.get_my() << ", L2 = " << utility :: L2(sol_num, t_src) << endl;
    } // Let managed memory go out of scope before calling cudaDeviceReset()
}
