/*
 * Implementation of slab_bc
 */

#include "slab_bc.h"

using namespace std;

slab_bc :: slab_bc(const twodads::slab_layout_t _sl, const twodads::bvals_t<twodads::real_t> _bc, const twodads::stiff_params_t _sp) : 
    boundaries(_bc), geom(_sl), tint_params(_sp),
    myfft{new dft_library_t(get_geom(), twodads::dft_t::dft_1d)},
    my_derivs(get_geom()),
    tint(get_geom(), get_bvals(),  _sp),
    arr1(_sl, _bc, 1), arr1_x(_sl, _bc, 1), arr1_y(_sl, _bc, 1),
    arr2(_sl, _bc, 1), arr2_x(_sl, _bc, 1), arr2_y(_sl, _bc, 1),
    arr3(_sl, _bc, 1), arr3_x(_sl, _bc, 1), arr3_y(_sl, _bc, 1),
    get_field_by_name{ {test_ns::field_t::arr1,     &arr1},
                       {test_ns::field_t::arr1_x,   &arr1_x},
                       {test_ns::field_t::arr1_y,   &arr1_y},
                       {test_ns::field_t::arr2,     &arr2},
                       {test_ns::field_t::arr2_x,   &arr2_x},
                       {test_ns::field_t::arr2_y,   &arr2_y},
                       {test_ns::field_t::arr3,     &arr3},
                       {test_ns::field_t::arr3_x,   &arr3_x},
                       {test_ns::field_t::arr3_y,   &arr3_y}}
{
}


void slab_bc :: dft_r2c(const test_ns::field_t fname, const size_t tlev)
{
    cuda_arr_real* arr{get_field_by_name.at(fname)};
    if(!((*arr).is_transformed()))
    {
        (*myfft).dft_r2c((*arr).get_tlev_ptr(tlev), reinterpret_cast<twodads::cmplx_t*>((*arr).get_tlev_ptr(tlev)));
        (*arr).set_transformed(true);
    }
    else
    {
        cout << "Array is already transformed, skipping dft r2c" << endl;
    }
}


void slab_bc :: dft_c2r(const test_ns::field_t fname, const size_t tlev)
{
    cuda_arr_real* arr{get_field_by_name.at(fname)};
    if((*arr).is_transformed())
    {
        (*myfft).dft_c2r(reinterpret_cast<twodads::cmplx_t*>((*arr).get_tlev_ptr(tlev)), (*arr).get_tlev_ptr(tlev));
        utility :: normalize(*arr, tlev);
    }
    else
    {
        cout << "Array is not transformed, skipping dft c2r" << endl;
    }
}


void slab_bc :: initialize_invlaplace(const test_ns::field_t fname)
{
    cuda_arr_real* arr = get_field_by_name.at(fname);
    (*arr).apply([] LAMBDACALLER (twodads::real_t dummy, const size_t n, const size_t m, const twodads::slab_layout_t geom) -> value_t
                {
                    const value_t x{geom.get_x(n)};
                    const value_t y{geom.get_y(m)};
                    return(exp(-0.5 * (x * x + y * y)) * (-2.0 + x * x + y * y));
                    //return(-1.0 * (twodads::TWOPI) * (twodads::TWOPI) * sin(twodads::TWOPI * y));
                }, 0);
}


void slab_bc :: initialize_sine(const test_ns::field_t fname)
{
    cuda_arr_real* arr = get_field_by_name.at(fname);
    (*arr).apply([] LAMBDACALLER (twodads::real_t dummy, size_t n, size_t m, twodads::slab_layout_t geom) -> value_t
                 {
                    return(sin(twodads::TWOPI * geom.get_x(n)) + 0.0 * sin(twodads::TWOPI * geom.get_y(m)));
                 }, 0);
}


void slab_bc :: initialize_arakawa(const test_ns::field_t fname1, const test_ns::field_t fname2)
{
    cuda_arr_real* arr1 = get_field_by_name.at(fname1);
    cuda_arr_real* arr2 = get_field_by_name.at(fname2);
  
    // arr1 =-sin^2(2 pi y) sin^2(2 pi x). Dirichlet bc, f(-1, y) = f(1, y) = 0.0
    (*arr1).apply([] LAMBDACALLER (twodads::real_t dummy, size_t n, size_t m, twodads::slab_layout_t geom) -> value_t
                  {
                    value_t x{geom.get_x(n)};
                    value_t y{geom.get_y(m)};
                    return(-1.0 * sin(twodads::TWOPI * y) * sin(twodads::TWOPI * y) * sin(twodads::TWOPI * x) * sin(twodads::TWOPI * x));
                  }, 0);

    // arr2 = sin(pi x) sin (pi y). Dirichlet BC: g(-1, y) = g(1, y) = 0.0
    (*arr2).apply([] LAMBDACALLER (twodads::real_t dummy, size_t n, size_t m, twodads::slab_layout_t geom) -> value_t
                  {
                    return(sin(twodads::PI * geom.get_x(n)) * sin(twodads::PI * geom.get_y(m)));
                  }, 0);
}


void slab_bc :: initialize_derivatives(const test_ns::field_t fname1, const test_ns::field_t fname2)
{
    cuda_arr_real* arr1 = get_field_by_name.at(fname1);
    cuda_arr_real* arr2 = get_field_by_name.at(fname2);

    (*arr1).apply([] LAMBDACALLER (twodads::real_t dummy, size_t n, size_t m, twodads::slab_layout_t geom) -> value_t
            {       
                return(sin(twodads::TWOPI * geom.get_x(n)));
            }, 0);

    (*arr2).apply([] LAMBDACALLER (twodads::real_t dummy, size_t n, size_t m, twodads::slab_layout_t geom) -> value_t
            {       
                value_t y{geom.get_y(m)};
                return(exp(-50.0 * (y - 0.5) * (y - 0.5))); 
            }, 0);
}


void slab_bc :: initialize_dfttest(const test_ns::field_t fname)
{
    cuda_arr_real* arr = get_field_by_name.at(fname);
    (*arr).apply([] LAMBDACALLER (twodads::real_t dummy, size_t n, size_t m, twodads::slab_layout_t geom) -> value_t
           {       
               return(sin(twodads::TWOPI * geom.get_y(m)));
           }, 0);
}


void slab_bc :: initialize_gaussian(const test_ns::field_t fname)
{
    cuda_arr_real* arr = get_field_by_name.at(fname);
    (*arr).apply([] LAMBDACALLER (twodads::real_t dummy, size_t n, size_t m, twodads::slab_layout_t geom) -> value_t
            {
                const twodads::real_t x{geom.get_x(n)};
                const twodads::real_t y{geom.get_y(m)};
                return(exp(-0.5 * (x * x + y * y)));
            }, 0);
}


// Compute x-derivative
void slab_bc :: d_dx(const test_ns::field_t fname_src, const test_ns::field_t fname_dst,
                     const size_t d, const size_t t_src, const size_t t_dst)
{
    cuda_arr_real* arr_src{get_field_by_name.at(fname_src)};
    cuda_arr_real* arr_dst{get_field_by_name.at(fname_dst)};

    if(d == 1)
    {
        cout << "slab_bc :: computing d_dx" << endl;
        my_derivs.dx_1((*arr_src), (*arr_dst), t_src, t_dst);   
    }
    else if (d == 2)
    {
        my_derivs.dx_2((*arr_src), (*arr_dst), t_src, t_dst);
    }
}


// Compute y-derivative
void slab_bc :: d_dy(const test_ns::field_t fname_src, const test_ns::field_t fname_dst,
                     const size_t d, const size_t t_src, const size_t t_dst)
{
    cuda_arr_real* arr_src = get_field_by_name.at(fname_src);
    cuda_arr_real* arr_dst = get_field_by_name.at(fname_dst);

    if(d == 1)
    {
        my_derivs.dy_1((*arr_src), (*arr_dst), t_src, t_dst);   
    }
    else if (d == 2)
    {
        my_derivs.dy_2((*arr_src), (*arr_dst), t_src, t_dst);
    }
}


// Compute Arakawa brackets
// res = {f, g} = f_y g_x - g_y f_x
// Input:
// fname_arr_f: array where f is stored
// fname_arr_g: array where g is stored
// fname_arr_res: array where we store the result
// t_src: time level at which f and g are taken
// t_dst: time level where we store the result in res
void slab_bc :: arakawa(const test_ns::field_t fname_arr_f, const test_ns::field_t fname_arr_g, const test_ns::field_t fname_arr_res,
                        const size_t t_src, const size_t t_dst)
{
    cuda_arr_real* f_arr = get_field_by_name.at(fname_arr_f);
    cuda_arr_real* g_arr = get_field_by_name.at(fname_arr_g);
    cuda_arr_real* res_arr = get_field_by_name.at(fname_arr_res);

    my_derivs.arakawa((*f_arr), (*g_arr), (*res_arr), t_src, t_dst);
}


// Invert the laplace equation
void slab_bc :: invert_laplace(const test_ns::field_t in, const test_ns::field_t out, const size_t t_src, const size_t t_dst)
{
    cuda_arr_real* in_arr{get_field_by_name.at(in)};
    cuda_arr_real* out_arr{get_field_by_name.at(out)};
    my_derivs.invert_laplace((*in_arr), (*out_arr), 
                             (*in_arr).get_bvals().get_bc_left(), (*in_arr).get_bvals().get_bv_left(),
                             (*in_arr).get_bvals().get_bc_right(), (*in_arr).get_bvals().get_bv_right(),
                             t_src, t_dst);
}


//void slab_bc :: initialize_tint(const test_ns::field_t fname)
//{
//    cuda_arr_real* in_arr{get_field_by_name.at(fname)};
//    if(!((*in_arr).is_transformed()))
//    {
//        dft_r2c(fname, 0);
//    }
//    (*tint).initialize_field(get_field_by_name.at(fname), 0);
//    dft_c2r(fname, 0);
//}

// Integrate the field in time
//void slab_bc :: integrate(const test_ns::field_t fname)
//{
//    //cuda_arr_real* in_arr = get_field_by_name.at(in);
//    //(*tint).integrate();
//    //(*tint).update_field(in_arr);
//}


slab_bc :: ~slab_bc()
{
    delete myfft;
    cout << "Deleting slab" << endl;
}

// End of file slab_bc.cu