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

int main(void){
    size_t Nx{128};
    size_t My{128};
    constexpr twodads::real_t x_l{-10.0};
    constexpr twodads::real_t Lx{20.0};
    constexpr twodads::real_t y_l{-10.0};
    constexpr twodads::real_t Ly{20.0};
    constexpr size_t tlevs{1};         // time levels for all arrays
    constexpr size_t tsrc{0};          // the time level we operate on

    cout << "Enter Nx: ";
    cin >> Nx;
    cout << "Enter My: ";
    cin >> My;

    stringstream fname;

    twodads::slab_layout_t my_geom(x_l, Lx / double(Nx), y_l, Ly / double(My), Nx, 0, My, 2, twodads::grid_t::cell_centered);
    twodads::bvals_t<double> my_bvals{twodads::bc_t::bc_dirichlet, twodads::bc_t::bc_dirichlet, twodads::bc_t::bc_periodic, twodads::bc_t::bc_periodic,
                                   0.0, 0.0, 0.0, 0.0};
    twodads::stiff_params_t params(0.001, 20.0, 20.0, 0.001, 0.0, my_geom.get_nx(), (my_geom.get_my() + my_geom.get_pad_y()) / 2, tlevs);
    {
        slab_bc my_slab(my_geom, my_bvals, params);
        my_slab.initialize_invlaplace(twodads::field_t::f_omega, tsrc);

        fname << "test_laplace_input_" << Nx << "_device.dat";
        utility :: print((*my_slab.get_array_ptr(twodads::field_t::f_omega)), tsrc, fname.str());

        cuda_array_bc_nogp<twodads::real_t, allocator_device> sol_an(my_geom, my_bvals, tlevs);
        sol_an.apply([] __device__ (twodads::real_t dummy, size_t n, size_t m, twodads::slab_layout_t geom) -> twodads::real_t
                {
                    const twodads::real_t x{geom.get_x(n)};
                    const twodads::real_t y{geom.get_y(m)};
                    return(exp(-0.5 * (x * x + y * y)));
                },
            tsrc);

        fname.str(string(""));
        fname << "test_laplace_solan_" << Nx << "_device.dat";
        utility :: print(sol_an, tsrc, fname.str());

        my_slab.invert_laplace(twodads::field_t::f_omega, twodads::field_t::f_strmf, tsrc, tsrc);

        // Write numerical solution to file
        fname.str(string(""));
        fname << "test_laplace_solnum_" << Nx << "_device.dat";
        utility :: print((*my_slab.get_array_ptr(twodads::field_t::f_strmf)), tsrc, fname.str());

        // Get the analytic solution
        sol_an -= my_slab.get_array_ptr(twodads::field_t::f_strmf);
        cout << "Nx = " << Nx << ", My = " << My << ", L2 = " << utility :: L2(sol_an, tsrc) << endl;
    } // Let managed memory go out of scope before calling cudaDeviceReset()
    // However cublas_handle_t survivs this scoping and we get a segfault from its destructor
    /*
    warning: Cuda API error detected: cudaFree returned (0x11)

warning: Cuda API error detected: cudaEventDestroy returned (0x1e)

warning: Cuda API error detected: cudaEventDestroy returned (0x1e)


Program received signal SIGSEGV, Segmentation fault.
0x00007fffeb3e903a in cuMemGetAttribute_v2 () from /usr/lib/x86_64-linux-gnu/libcuda.so.1
(cuda-gdb) bt
#0  0x00007fffeb3e903a in cuMemGetAttribute_v2 () from /usr/lib/x86_64-linux-gnu/libcuda.so.1
#1  0x00007fffeb3598a6 in cuVDPAUCtxCreate () from /usr/lib/x86_64-linux-gnu/libcuda.so.1
#2  0x00007fffeb33293a in cuEventDestroy_v2 () from /usr/lib/x86_64-linux-gnu/libcuda.so.1
#3  0x00007fffefad7724 in cublasSgemmEx () from /usr/local/cuda/lib64/libcublas.so.7.5
#4  0x00007fffefb0b984 in cublasSgemmEx () from /usr/local/cuda/lib64/libcublas.so.7.5
#5  0x00007fffef8e4797 in ?? () from /usr/local/cuda/lib64/libcublas.so.7.5
#6  0x00007fffef91932d in cublasDestroy_v2 () from /usr/local/cuda/lib64/libcublas.so.7.5
#7  0x0000000000414e38 in solvers::cublas_handle_t::~cublas_handle_t (this=0x728100 <solvers::cublas_handle_t::get_handle()::h>, __in_chrg=<optimized out>) at /home/rku000/source/2dads/include/solvers.h:49
#8  0x00007fffec1d8b29 in secure_getenv () from /lib/x86_64-linux-gnu/libc.so.6
#9  0x00007fffec1d8b75 in exit () from /lib/x86_64-linux-gnu/libc.so.6
#10 0x00007fffec1c2b4c in __libc_start_main () from /lib/x86_64-linux-gnu/libc.so.6

*/
    //cudaDeviceReset();
}