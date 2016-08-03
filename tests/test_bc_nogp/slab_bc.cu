/*
 * Implementation of slab_bc
 */

#include "slab_bc.h"

slab_bc :: slab_bc(const cuda::slab_layout_t _sl, const cuda::bvals_t<real_t> _bc) :
    Nx(_sl.Nx), My(_sl.My), tlevs(1), 
    boundaries(_bc), geom(_sl), 
    der(_sl),
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
    init_dft();
}


void slab_bc :: print_field(const test_ns::field_t fname) const
{
    get_field_by_name.at(fname) -> copy_device_to_host();
    get_field_by_name.at(fname) -> print();
}


void slab_bc :: print_field(const test_ns::field_t fname, const string file_name) const
{
    get_field_by_name.at(fname) -> copy_device_to_host();

    ofstream output_file;
    output_file.open(file_name.data());
    output_file << *get_field_by_name.at(fname) << endl;
    output_file.close();
}


//cuda_arr_real* slab_bc :: get_array_ptr(const test_ns::field_t fname) const
//{
//    return(get_field_by_name.at(fname));
//}


void slab_bc :: init_dft()
{
    cout << "init_dft()" << endl;
	cufftResult err;
    int dft_size[2] = {0, 0};       // Size of the transformation
    int dft_onembed[2] = {0, 0};    // Input, embedded size of the transformation
    int dft_inembed[2] = {0, 0};    // Output, embedded size of the transformation
    int dist_real{0};               // Distance between two vectors inpu vectors for DFT, in units of double
    int dist_cplx{0};               // Distance between two complex input vectors for iDFT, in units of (2 * double)
    int istride{1};                 // Distance between two successive input and output elements in the least significant (that is, the innermost) dimension
    int ostride{1};
    switch(boundaries.get_bc_left())
    {
        case cuda::bc_t::bc_dirichlet:
            // fall through
        case cuda::bc_t::bc_neumann:
            // Initialize a 1d transformation
            dft_size[0] = int(My);                                  
            dft_onembed[0] = int(My / 2 + 1); 
            dist_real = int(My + geom.pad_y); 
            dist_cplx = int(My / 2 + 1); 

            // Plan the DFT, D2Z
            if ((err = cufftPlanMany(&plan_r2c, 
                                     1,             //int rank
                                     dft_size,      //int* n
                                     dft_size,      //int* inembed
                                     istride,       //int istride
                                     dist_real,     //int idist
                                     dft_onembed,   //int* onembed
                                     ostride,       //int ostride
                                     dist_cplx,     //int odist
                                     CUFFT_D2Z,     //cufftType type
                                     Nx)            //int batch
                ) != CUFFT_SUCCESS)
            {
                stringstream err_str;
                err_str << "Error planning 1d D2Z DFT: " << err << "\n";
                throw gpu_error(err_str.str());
            }
           
            // Plan the iDFT, Z2D 
            if((err = cufftPlanMany(&plan_c2r,
                                    1,              //int rank
                                    dft_size,       //int* n
                                    dft_onembed,    //int* inembed
                                    istride,        //int istride
                                    dist_cplx,      //int idist
                                    dft_size,       //int* onembed
                                    ostride,        //int ostride
                                    dist_real,      //int odist
                                    CUFFT_Z2D,      //cufftType type
                                    Nx)             //int batch
               ) != CUFFT_SUCCESS)
            {
                stringstream err_str;
                err_str << "Error planning 1d Z2D DFT: " << err << "\n";
                throw gpu_error(err_str.str());
            }
            break;

        case cuda::bc_t::bc_periodic:
            // Initialize 2d transformation
            dft_size[1] = int(My);
            dft_size[0] = int(Nx);
            dft_inembed[1] = My + geom.pad_y; 
            dft_onembed[1] = My / 2 + 1;
            istride = 1;
            ostride = 1;

            // Plan 2d r2c transformation
            //if((err = cufftPlanMany(&plan_r2c,
            //                        2,              //int rank
            //                        dft_size,       //int* n
            //                        dft_inembed,    //int* inembed
            //                        istride,        //int istride
            //                        (Nx + geom.pad_y) * (My + geom.pad_y), //int idist
            //                        dft_onembed,    //int* onembed
            //                        ostride,        //int ostride
            //                        Nx * (My / 2 + 1), //int odist
            //                        CUFFT_D2Z,      //cufftType typ
            //                        1)              //int batch
            //   ) != CUFFT_SUCCESS)
            if((err = cufftPlan2d(&plan_r2c, Nx, My, CUFFT_D2Z) )!= CUFFT_SUCCESS)
            {
                stringstream err_str;
                err_str << "Error planning 2d D2Z DFT: " << err << "\n";
                throw gpu_error(err_str.str());
            }

            // Plan 2d c2r transformation
            //if((err = cufftPlanMany(&plan_c2r,
            //                        2,
            //                        dft_size,
            //                        dft_onembed,
            //                        ostride,
            //                        Nx * (My / 2 + 1),
            //                        dft_inembed,
            //                        istride,
            //                        Nx * (My + geom.pad_y),
            //                        CUFFT_Z2D,
            //                        1)
            //  ) != CUFFT_SUCCESS)
            if((err = cufftPlan2d(&plan_c2r, Nx, My, CUFFT_Z2D) )!= CUFFT_SUCCESS)
            {
                stringstream err_str;
                err_str << "Error planning 2d Z2D DFT: " << err << "\n";
                throw gpu_error(err_str.str());
            }


            break;
    }
}


/// Perform DFT in y-direction, row-wise
void slab_bc :: dft_r2c(const test_ns::field_t fname, const size_t tlev)
{
    cuda_arr_real* arr{get_field_by_name.at(fname)};
    cufftResult err;

	// Use the CUFFT plan to transform the signal in place. 
	if ((err = cufftExecD2Z(plan_r2c, 
	                        (*arr).get_array_d(tlev),
					        (cufftDoubleComplex*) (*arr).get_array_d(tlev)) 
        ) != CUFFT_SUCCESS)
    {
        stringstream err_str;
        err_str << "Error executing D2Z DFT: " << cufftGetErrorString.at(err) << endl;
        throw gpu_error(err_str.str());
    }
    cerr << err << endl;
}


/// Perform iDFT in y-direction, row-wise. Normalize real array after transform
void slab_bc :: dft_c2r(const test_ns::field_t fname, const size_t tlev)
{
	cuda_arr_real* arr{get_field_by_name.at(fname)};
	cufftResult err;

	// Use the CUFFT plan to transform the signal in place. 
	err = cufftExecZ2D(plan_c2r,
					   (cufftDoubleComplex*) (*arr).get_array_d(tlev), 
				       (*arr).get_array_d(tlev));
	if(err != CUFFT_SUCCESS)
    {
        stringstream err_str;
        err_str << "Error planning Z2D DFT: " << cufftGetErrorString.at(err) << endl;
        throw gpu_error(err_str.str());
    }

    // Normalize
    (*arr).normalize(0);
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


// Compute spatial derivatives in x and y direction
void slab_bc :: d_dx_dy(const size_t tlev)
{
    //dft_r2c(0);
    //d_dy1(arr1, arr1_y);
    //dft_c2r(0);
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
