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

int main(void)
{
    const size_t t_src{0};
    slab_config_js my_config(std::string("input_test_laplace_fd.json"));

    {
        slab_bc my_slab(my_config);
        real_arr sol_an(my_config.get_geom(), my_config.get_bvals(twodads::field_t::f_theta), 1);
        real_arr sol_num(my_config.get_geom(), my_config.get_bvals(twodads::field_t::f_theta), 1);
        
        //my_slab.initialize_invlaplace(twodads::field_t::f_omega, tsrc);
        //fname << "test_laplace_input_" << Nx << "_host.dat";
        //utility :: print((*my_slab.get_array_ptr(twodads::field_t::f_omega)), tsrc, fname.str());
        //cuda_array_bc_nogp<twodads::real_t, allocator_host> sol_an(my_geom, my_bvals, tlevs);

        stringstream fname;
        real_arr* arr_ptr{my_slab.get_array_ptr(twodads::field_t::f_omega)};

        // Initialize input for laplace solver
        (*arr_ptr).apply([] (twodads::real_t dummy, size_t n, size_t m, twodads::slab_layout_t geom) -> twodads::real_t
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

        my_slab.invert_laplace(twodads::field_t::f_omega, twodads::field_t::f_strmf, t_src, t_src);

        // Write numerical solution to file
        fname.str(string(""));
        fname << "test_laplace_solnum_" << my_config.get_nx() << "_host.dat";
        utility :: print((*my_slab.get_array_ptr(twodads::field_t::f_strmf)), t_src, fname.str());

        // Get the analytic solution
        sol_an -= my_slab.get_array_ptr(twodads::field_t::f_strmf);
        cout << "Nx = " << my_config.get_nx() << ", My = " << my_config.get_my() << ", L2 = " << utility :: L2(sol_an, t_src) << endl;
    } // Let managed memory go out of scope before calling cudaDeviceReset()
}