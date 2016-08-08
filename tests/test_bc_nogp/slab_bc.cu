/*
 * Implementation of slab_bc
 */

#include "slab_bc.h"

slab_bc :: slab_bc(const cuda::slab_layout_t _sl, const cuda::bvals_t<real_t> _bc) :
    Nx(_sl.Nx), My(_sl.My), tlevs(1), 
    boundaries(_bc), geom(_sl), 
    der(_sl),
    myfft(get_geom(), cuda::dft_t::dft_1d),
    arr1(_sl, _bc, tlevs), arr1_x(_sl, _bc, 1), arr1_y(_sl, _bc, 1),
    arr2(_sl, _bc, tlevs), arr2_x(_sl, _bc, 1), arr2_y(_sl, _bc, 1),
    arr3(_sl, _bc, tlevs), arr3_x(_sl, _bc, 1), arr3_y(_sl, _bc, 1),
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
    cout << "Creating new slab" << endl;
}


void slab_bc :: print_field(const test_ns::field_t fname) const
{
    get_field_by_name.at(fname) -> copy_device_to_host();
    get_field_by_name.at(fname) -> print();
}


void slab_bc :: print_field(const test_ns::field_t fname, const string file_name) const
{
    //get_field_by_name.at(fname) -> copy_device_to_host();
    ofstream output_file;
    output_file.open(file_name.data());
    output_file << *get_field_by_name.at(fname) << endl;
    output_file.close();
}


void slab_bc :: dft_r2c(const test_ns::field_t fname, const size_t tlev)
{
    cuda_arr_real* arr{get_field_by_name.at(fname)};
    //myfft.dft_r2c((*arr).get_array_d(tlev), (CuCmplx<cuda::real_t>*)((*arr).get_array_d(tlev)));
    myfft.dft_r2c((*arr).get_array_d(tlev), reinterpret_cast<CuCmplx<cuda::real_t>*>((*arr).get_array_d(tlev)));
}


void slab_bc :: dft_c2r(const test_ns::field_t fname, const size_t tlev)
{
    cuda_arr_real* arr{get_field_by_name.at(fname)};
    //myfft.dft_c2r((CuCmplx<cuda::real_t>*) (*arr).get_array_d(tlev), (*arr).get_array_d(tlev));
    if((*arr).is_transformed())
    {
        myfft.dft_c2r(reinterpret_cast<CuCmplx<cuda::real_t>*>((*arr).get_array_d(tlev)), (*arr).get_array_d(tlev));
        (*arr).normalize(tlev);
    }
}


void slab_bc :: initialize_invlaplace(const test_ns::field_t fname)
{
    cuda_arr_real* arr = get_field_by_name.at(fname);
    (*arr).evaluate([=] __device__ (size_t n, size_t m, cuda::slab_layout_t geom) -> value_t
            {
                value_t x{geom.x_left + (value_t(n) + 0.5) * geom.delta_x};
                value_t y{geom.y_lo + (value_t(m) + 0.5) * geom.delta_y};
                return(exp(-0.5 * (x * x + y * y)) * (-2.0 + x * x + y * y));
            }, 0);
}


void slab_bc :: initialize_sine(const test_ns::field_t fname)
{
    cuda_arr_real* arr = get_field_by_name.at(fname);
    (*arr).evaluate([=] __device__ (size_t n, size_t m, cuda::slab_layout_t geom) -> value_t
            {
                value_t x{geom.x_left + (value_t(n) + 0.5) * geom.delta_x};
                value_t y{geom.y_lo + (value_t(m) + 0.5) * geom.delta_y};
                return(sin(cuda::TWOPI * y) + 0.0 * sin(cuda::TWOPI * x));
            }, 0);
}


void slab_bc :: initialize_arakawa(const test_ns::field_t fname1, const test_ns::field_t fname2)
{
    cuda_arr_real* arr1 = get_field_by_name.at(fname1);
    cuda_arr_real* arr2 = get_field_by_name.at(fname2);
  
    // arr1 =-sin^2(2 pi y) sin^2(2 pi x). Dirichlet bc, f(-1, y) = f(1, y) = 0.0
    (*arr1).evaluate([=] __device__(size_t n, size_t m, cuda::slab_layout_t geom) -> value_t
            {
                value_t x{geom.x_left + (value_t(n) + 0.5) * geom.delta_x};
                value_t y{geom.y_lo + (value_t(m) + 0.5) * geom.delta_y};
                return(-1.0 * sin(cuda::TWOPI * y) * sin(cuda::TWOPI * y) * sin(cuda::TWOPI * x) * sin(cuda::TWOPI * x));
            }, 0);

    // arr2 = sin(pi x) sin (pi y). Dirichlet BC: g(-1, y) = g(1, y) = 0.0
    (*arr2).evaluate([=] __device__(size_t n, size_t m, cuda::slab_layout_t geom) -> value_t
            {
                value_t x{geom.x_left + (value_t(n) + 0.5) * geom.delta_x};
                value_t y{geom.y_lo + (value_t(m) + 0.5) * geom.delta_y};
                return(sin(cuda::PI * x) * sin(cuda::PI * y));
            }, 0);
}


void slab_bc :: initialize_derivatives(const test_ns::field_t fname1, const test_ns::field_t fname2)
{
    cuda_arr_real* arr1 = get_field_by_name.at(fname1);
    cuda_arr_real* arr2 = get_field_by_name.at(fname2);

    (*arr1).evaluate([=] __device__(size_t n, size_t m, cuda::slab_layout_t geom) -> value_t
            {       
                value_t x{geom.x_left + (value_t(n) + 0.5) * geom.delta_x};
                return(sin(cuda::TWOPI * x));
            }, 0);

    (*arr2).evaluate([=] __device__(size_t n, size_t m, cuda::slab_layout_t geom) -> value_t
            {       
                value_t y{geom.y_lo + (value_t(m) + 0.5) * geom.delta_y};
                return(exp(-50.0 * (y - 0.5) * (y - 0.5))); 
            }, 0);
}


void slab_bc :: initialize_dfttest(const test_ns::field_t fname)
{
    cuda_arr_real* arr = get_field_by_name.at(fname);
    (*arr).evaluate([=] __device__(size_t n, size_t m, cuda::slab_layout_t geom) -> value_t
           {       
               //value_t x{geom.x_left + (value_t(n) + 0.5) * geom.delta_x};
               value_t y{geom.y_lo + (value_t(m) + 0.0) * geom.delta_y};
               return(sin(cuda::TWOPI * y));
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
        der.dx_1((*arr_src), (*arr_dst), t_src, t_dst);   
    }
    else if (d == 2)
    {
        der.dx_2((*arr_src), (*arr_dst), t_src, t_dst);
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
        der.dy_1((*arr_src), (*arr_dst), t_src, t_dst);   
    }
    else if (d == 2)
    {
        der.dy_2((*arr_src), (*arr_dst), t_src, t_dst);
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

    der.arakawa((*f_arr), (*g_arr), (*res_arr), t_src, t_dst);
}


// Invert the laplace equation
void slab_bc :: invert_laplace(const test_ns::field_t in, const test_ns::field_t out, const size_t t_src, const size_t t_dst)
{
    cuda_arr_real* in_arr = get_field_by_name.at(in);
    cuda_arr_real* out_arr = get_field_by_name.at(out);
    der.invert_laplace((*in_arr), (*out_arr), 
                       in_arr -> get_bvals().get_bc_left(), in_arr -> get_bvals().get_bv_left(),
                       in_arr -> get_bvals().get_bc_right(), in_arr -> get_bvals().get_bv_right(),
                       t_src, t_dst);
}


slab_bc :: ~slab_bc()
{
    cout << "Deleting slab" << endl;
}
