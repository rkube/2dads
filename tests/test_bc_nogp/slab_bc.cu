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
    get_field_by_name{ {test_ns::field_t::arr1,     &arr1},
                       {test_ns::field_t::arr1_x,   &arr1_x},
                       {test_ns::field_t::arr1_y,   &arr1_y},
                       {test_ns::field_t::arr2,     &arr2},
                       {test_ns::field_t::arr2_x,   &arr2_x},
                       {test_ns::field_t::arr2_y,   &arr2_y}}
{
    cout << "Creating new slab" << endl;
    arr1.evaluate_device(0);
    init_dft();
}


void slab_bc :: print_field(const test_ns::field_t fname) const
{
    get_field_by_name.at(fname) -> copy_device_to_host();
    get_field_by_name.at(fname) -> dump_full();
}


void slab_bc :: print_field(const test_ns::field_t fname, const string file_name) const
{
    get_field_by_name.at(fname) -> copy_device_to_host();

    ofstream output_file;
    output_file.open(file_name.data());
    output_file << *get_field_by_name.at(fname) << endl;
    output_file.close();
}


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
    switch(boundaries.bc_left)
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
    (*arr).init_inv_laplace();
}


void slab_bc :: initialize_sine(const test_ns::field_t fname)
{
    cuda_arr_real* arr = get_field_by_name.at(fname);
    (*arr).init_sine();
}


// Compute spatial derivatives in x and y direction
void slab_bc :: d_dx_dy(const size_t tlev)
{
    //dft_r2c(0);
    //d_dy1(arr1, arr1_y);
    //dft_c2r(0);
}

// Invert the laplace equation
void slab_bc :: invert_laplace(const test_ns::field_t in, const test_ns::field_t out, const size_t tlev)
{
    cuda_arr_real* in_arr = get_field_by_name.at(in);
    cuda_arr_real* out_arr = get_field_by_name.at(out);
    der.invert_laplace((*in_arr), (*out_arr), 
                       in_arr -> get_bvals().bc_left, in_arr -> get_bvals().bval_left,
                       in_arr -> get_bvals().bc_right, in_arr -> get_bvals().bval_right,
            tlev);
}


slab_bc :: ~slab_bc()
{
    cout << "Deleting slab" << endl;
}
