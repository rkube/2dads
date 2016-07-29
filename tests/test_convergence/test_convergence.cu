/*
 * Test convergence of poisson brackets
 */

#include <iostream>
#include <fstream>
#include <string>
#include <derivatives.h>
#include <cuda_array4.h>
#include <cuda_darray.h>

#ifdef __CUDA_ARCH__
#define CUDAMEMBER __device__ 
#endif

#ifndef __CUDA_ARCH__
#define CUDAMEMBER
#endif

using namespace std;


#define XNULL 0.1
#define YNULL 0.1

template <typename T>
class fun1 {
    // f(x, y) = exp(-(x - x_0)^2 - (y - y_0)^2)
    public:
        CUDAMEMBER T operator() (T x, T y) const { return ( exp(-(x - XNULL) * (x - XNULL) - (y - YNULL) * (y - YNULL)) );};
};

template <typename T>
class fun1_x {
    // f_x(x, y) = -2 (x - x_0) exp(-(x - x_0)^2 - (y - y_0)^2)
    public: 
        CUDAMEMBER T operator() (T x, T y) const { return( -2.0 * (x - XNULL) * exp(-(x - XNULL) * (x - XNULL) - (y - YNULL) * (y - YNULL)) );};
};

template <typename T>
class fun1_y {
    // f_y(x, y) = -2 (y - y_0) exp(-(x - x_0)^2 - (y - y_0)^2)
    public:
        CUDAMEMBER T operator() (T x, T y) const { return( -2.0 * (y - YNULL) * exp(-(x - XNULL) * (x - XNULL) - (y - YNULL) * (y - YNULL)) );};
};


template <typename T>
class fun1_xx_yy {
    // nabla^2 f(x,y) = 4. * (-1. + (x-x_0)^2 + (y-y_0)^2)  * exp( -(x - x_0)^2 - (y - y_0)^2)
    public:
        CUDAMEMBER T operator() (T x, T y) const { return(4. * (-1. + (x - XNULL) * (x - XNULL) + (y - YNULL) * (y - YNULL)) * exp(-(x - XNULL) * (x - XNULL) - (y - YNULL) * (y - YNULL))); };
};

template <typename T>
class fun2 {
    // g(x, y) = exp(-(x + x_0)^2 - (y + y_0)^2)
    public:
        CUDAMEMBER T operator() (T x, T y) const { return ( exp(-(x + XNULL) * (x + XNULL) - (y + YNULL) * (y + YNULL)) );};
};

template <typename T>
class fun2_x {
    // g_x(x, y) = -2 (x + x_0) exp(-(x + x_0)^2 - (y + y_0)^2)
    public:
        CUDAMEMBER T operator() (T x, T y) const { return ( -2.0 * (x + XNULL) * exp(-(x + XNULL) * (x + XNULL) - (y + YNULL) * (y + YNULL)) );};
};

template <typename T>
class fun2_y {
    // g_y(x, y) = -2 (y + y_0) exp(-(x + x_0)^2 - (y + y_0)^2)
    public:
        CUDAMEMBER T operator() (T x, T y) const { return ( -2.0 * (y + YNULL) * exp(-(x + XNULL) * (x + XNULL) - (y + YNULL) * (y + YNULL)) );};
};


template <typename T>
class poisson {
    public:
        CUDAMEMBER T operator() (T x, T y) const { return( exp(-2.0 * (x * x + XNULL * XNULL + y * y + YNULL * YNULL)) *(-8. * XNULL * y + 8. * YNULL * x));};
};

void do_test()
{
    int Nx, My;
    cout << "Enter Nx: " << endl;
    cin >> Nx;
    cout << "Enter My: " << endl;
    cin >> My;

    //constexpr int Nx{32};
    //constexpr int My{32};
    cuda::real_t Lx{10.0};
    cuda::real_t Ly{10.0};
    cuda::real_t dx{Lx / Nx};
    cuda::real_t dy{Lx / My};
    const cuda::slab_layout_t sl(-0.5 * Lx, Lx / static_cast<cuda::real_t>(Nx), -0.5 * Ly, Ly / static_cast<cuda::real_t>(My), Nx, 0, My, 0); 

    cout << "Nx = " << Nx << ", Lx = " << Lx << ", dx = " << dx << ", dy = " << dy << endl;

    // Derivator
    derivs<cuda::real_t> der(sl);

    ofstream of;

    cout << "f_num" << endl << endl;
    cuda_array<cuda::real_t> f_num(1, My, Nx);
    cout << "fx_num" << endl << endl;
    cuda_array<cuda::real_t> fx_num(1, My, Nx);
    cout << "fy_num" << endl << endl;
    cuda_array<cuda::real_t> fy_num(1, My, Nx);

    cout << "fy_analytic" << endl << endl;
    cuda_array<cuda::real_t> fx_analytic(1, My, Nx);
    cout << "fy_analytic" << endl << endl;
    cuda_array<cuda::real_t> fy_analytic(1, My, Nx);

    cout << "g_num" << endl << endl;
    cuda_array<cuda::real_t> g_num(1, My, Nx);
    cout << "gx_num" << endl << endl;
    cuda_array<cuda::real_t> gx_num(1, My, Nx);
    cout << "gy_num" << endl << endl;
    cuda_array<cuda::real_t> gy_num(1, My, Nx);

    cout << "gy_analytic" << endl << endl;
    cuda_array<cuda::real_t> gx_analytic(1, My, Nx);
    cout << "gy_analytic" << endl << endl;
    cuda_array<cuda::real_t> gy_analytic(1, My, Nx);

    /* Numerically computad Laplace of f */
    cout << "lapl_f_num" << endl << endl;
    cuda_array<cuda::real_t> lapl_f_num(1, My, Nx);
    /* Analytic Laplacian of f*/
    cout << "lapl_f_an" << endl << endl;
    cuda_array<cuda::real_t> lapl_f_an(1, My, Nx);

    /* Reconstructed f from Laplace */
    cout << "f_rcr" << endl << endl;
    cuda_array<cuda::real_t> f_rcr(1, My, Nx);

    /* Analytic solution of the poisson bracket */
    cout << "pb_an" << endl << endl;
    cuda_array<cuda::real_t> pb_an(1, My, Nx);
    /* Numerical computed poisson bracket */
    cout << "pb_num" << endl << endl;
    cuda_array<cuda::real_t> pb_num1(1, My, Nx);
    cuda_array<cuda::real_t> pb_num2(1, My, Nx);
    cuda_array<cuda::real_t> pb_num3(1, My, Nx);


    /* Evaluate f and g */
    f_num.op_scalar_fun<fun1<cuda::real_t> >(sl, 0);
    of.open("f.dat");
    of << f_num;
    of.close();

    g_num.op_scalar_fun<fun2<cuda::real_t> >(sl, 0);
    of.open("g.dat");
    of << g_num;
    of.close();

    pb_an.op_scalar_fun<poisson<cuda::real_t> >(sl, 0);
    of.open("pb_an.dat");
    of << pb_an;
    of.close();

    /* Compute spectral derivative of f and g */
    der.d_dx1_dy1(f_num, fx_num, fy_num);
    der.d_dx1_dy1(g_num, gx_num, gy_num);
    der.d_laplace(f_num, lapl_f_num, 0);

    /* Invert Laplace equation */
    der.inv_laplace(lapl_f_num, f_rcr, 0);

    // Compare numerical derivatives to analytic expressions
    fx_analytic.op_scalar_fun<fun1_x<cuda::real_t> >(sl, 0);
    of.open("fx_an.dat");
    of << fx_analytic;
    of.close();
    
    fy_analytic.op_scalar_fun<fun1_y<cuda::real_t> >(sl, 0);
    of.open("fy_an.dat");
    of << fy_analytic;
    of.close();

    gx_analytic.op_scalar_fun<fun2_x<cuda::real_t> >(sl, 0);
    of.open("gx_an.dat");
    of << gx_analytic;
    of.close();

    gy_analytic.op_scalar_fun<fun2_y<cuda::real_t> >(sl, 0);
    of.open("gy_an.dat");
    of << gy_analytic;
    of.close();

    lapl_f_an.op_scalar_fun<fun1_xx_yy<cuda::real_t> >(sl, 0);
    of.open("lapl_f_an.dat");
    of << lapl_f_an;
    of.close();


    of.open("fx_num.dat");
    of << fx_num;
    of.close();

    of.open("fy_num.dat");
    of << fy_num;
    of.close();

    of.open("gx_num.dat");
    of << gx_num;
    of.close();

    of.open("gy_num.dat");
    of << gy_num;
    of.close();

    of.open("lapl_f_num.dat");
    of << lapl_f_num;
    of.close();


    // compute L2 error of derivatives
    cout << "diff_deriv" << endl << endl;
    cuda_darray<cuda::real_t> diff_deriv(My, Nx);
    cout << "diff_deriv, assign" << endl << endl;
    diff_deriv = (fx_num - fx_analytic) * (fx_num - fx_analytic);
    cout << "L2 norm of error in f_x is " << sqrt(diff_deriv.get_sum()) / double(Nx * My) << endl;
    cout << "diff_deriv, assign" << endl << endl;
    diff_deriv = (fy_num - fy_analytic) * (fy_num - fy_analytic);
    cout << "L2 norm of error in f_y is " << sqrt(diff_deriv.get_sum()) / double(Nx * My) << endl; 
    cout << "diff_deriv, assign" << endl << endl;
    diff_deriv = (gx_num - gx_analytic) * (gx_num - gx_analytic);
    cout << "L2 norm of error in g_x is " << sqrt(diff_deriv.get_sum()) / double(Nx * My) << endl;
    cout << "diff_deriv, assign" << endl << endl;
    diff_deriv = (gy_num - gy_analytic) * (gy_num - gy_analytic);
    cout << "L2 norm of error in g_y is " << sqrt(diff_deriv.get_sum()) / double(Nx * My) << endl;
    cout << "diff_deriv, assign" << endl << endl;
    diff_deriv = (lapl_f_num - lapl_f_an) * (lapl_f_num - lapl_f_an);
    cout << "L2 norm of error in laplace(f) is " << sqrt(diff_deriv.get_sum()) / double(Nx * My) << endl;

    /* Print L2 norm of f reconstruction... */
    diff_deriv = f_num;
    cuda::real_t f_mean = diff_deriv.get_mean();

    cout << "diff_deriv, assign" << endl << endl;
    diff_deriv = (f_rcr - (f_num - f_mean)) * (f_rcr - (f_num - f_mean));
    cout << "L2 norm of error in inv_lapla(f) is " << sqrt(diff_deriv.get_sum()) / double(Nx * My) << endl;

    /* Compute poisson bracket in three different ways:
     * 1.) {f, g} = f_x g_y - f_y g_x
     */
    pb_num1 = fx_num * gy_num - fy_num * gx_num;
    cout << "diff_deriv, assign" << endl << endl;
    diff_deriv = (pb_num1 - pb_an) * (pb_num1 - pb_an);
    cout << "Method 1: {f,g} = f_x g_y - f_y g_x" << endl;
    cout << "L2 norm of error in poisson bracket is " << sqrt(diff_deriv.get_sum()) / double(Nx * My) << endl;


    /*
     * 2.) {f, g} = (f g_y)_x - (f g_x)_y
     */

    cuda_array<cuda::real_t> tmp(f_num * gy_num);
    cuda_array<cuda::real_t> tmp_x(1, My, Nx);
    cuda_array<cuda::real_t> tmp_y(1, My, Nx);

    der.d_dx1_dy1(tmp, tmp_x, tmp_y);
    // {f, g} = (f g_y)_x + ... 
    pb_num2 = tmp_x;
    
    tmp = f_num * gx_num;
    
    der.d_dx1_dy1(tmp, tmp_x, tmp_y);

    // {f, g} = (f g_y)_x - (f g_x)_y 
    pb_num2 -= tmp_y;
    diff_deriv = (pb_num1 - pb_an) * (pb_num1 - pb_an);
    cout << "Method 2: {f,g} = (f g_y)_x - (f g_x)_y" << endl;
    cout << "L2 norm of error in poisson bracket is " << sqrt(diff_deriv.get_sum()) / double(Nx * My) << endl;

    /*
     * 3.) {f, g} = (f_x g)_y - (f_y g)_x
     */


    tmp = fx_num * g_num;
    der.d_dx1_dy1(tmp, tmp_x, tmp_y);

    // {f, g} = (f_x g)_y - ...
    pb_num3 = tmp_y;

    tmp = fy_num * g_num;
    der.d_dx1_dy1(tmp, tmp_x, tmp_y);
    // {f, g} = (f_x g)_y - (f_y g)_x
    pb_num3 -= tmp_x;
    diff_deriv = (pb_num3 - pb_an) * (pb_num3 - pb_an);
    cout << "Method 3: {f,g} = (f_x g)_y - (f_y g)_x" << endl;
    cout << "L2 norm of error in poisson bracket is " << sqrt(diff_deriv.get_sum()) / double(Nx * My) << endl;


    tmp = (pb_num1 + pb_num2 + pb_num3) / 3.0;
    diff_deriv = (tmp - pb_an) * (tmp - pb_an);
    cout << "Method 4" << endl;
    cout << "L2 norm of error in poisson bracket is " << sqrt(diff_deriv.get_sum()) / double(Nx * My) << endl;

    of.open("pb_num1.dat");   
    of << pb_num1;
    of.close();

    of.open("pb_num2.dat");   
    of << pb_num2;
    of.close();

    of.open("pb_num3.dat");   
    of << pb_num3;
    of.close();

    of.open("pb_num4.dat");   
    of << tmp;
    of.close();

    of.open("pb_an.dat");   
    of << pb_an;
    of.close();

    return;
}

int main(void)
{
    do_test();
    cudaDeviceReset();
    return(0);
}


