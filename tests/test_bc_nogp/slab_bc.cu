/*
 * Implementation of slab_bc
 */

#include "slab_bc.h"

slab_bc :: slab_bc(const cuda::slab_layout_t _sl, const cuda::bvals_t<real_t> _bc) :
    Nx(_sl.Nx), My(_sl.My), tlevs(1), 
    boundaries(_bc), geom(_sl), 
    arr1(_sl, _bc, tlevs), arr1_x(_sl, _bc, 1), arr1_y(_sl, _bc, 1)
{
    cout << "Creating new slab" << endl;
    arr1.evaluate_device(0);
}


void slab_bc :: dump_arr1()
{
    arr1.copy_device_to_host();
    arr1.dump_full();
}


void slab_bc :: dump_arr1x()
{
    arr1_x.copy_device_to_host();
    arr1_x.dump_full();
}


void slab_bc :: dump_arr1y()
{
    arr1_y.copy_device_to_host();
    arr1_y.dump_full();
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
void slab_bc :: dft_r2c(const size_t tlev)
{
    //const size_t offset = cuda::gp_offset_x * (geom.My + geom.pad_y) + cuda::gp_offset_y; 
    cufftResult err;
	
	// Use the CUFFT plan to transform the signal in place. 
	if ((err = cufftExecD2Z(plan_r2c, 
	                        arr1.get_array_d(tlev),
					        (cufftDoubleComplex*) arr1.get_array_d(tlev)) 
        ) != CUFFT_SUCCESS)
    {
        stringstream err_str;
        err_str << "Error executing D2Z DFT: " << cufftGetErrorString.at(err) << endl;
        throw gpu_error(err_str.str());
    }
}


/// Perform iDFT in y-direction, row-wise. Normalize real array after transform
void slab_bc :: dft_c2r(const size_t tlev)
{
	//const size_t offset = cuda::gp_offset_x * (My + cuda::num_gp_y) + cuda::gp_offset_y; 
	cufftResult err;

	// Use the CUFFT plan to transform the signal in place. 
	err = cufftExecZ2D(plan_c2r,
					   (cufftDoubleComplex*) arr1.get_array_d(tlev), 
				       arr1.get_array_d(tlev));
	if(err != CUFFT_SUCCESS)
    {
        stringstream err_str;
        err_str << "Error planning Z2D DFT: " << cufftGetErrorString.at(err) << endl;
        throw gpu_error(err_str.str());
    }

    // Normalize
    arr1.normalize(0);
}


// Compute spatial derivatives in x and y direction
void slab_bc :: d_dx_dy(const size_t tlev)
{
    dx_1(arr1, arr1_x, 0, geom, boundaries);
    //dft_r2c(0);
    //d_dy1(arr1, arr1_y);
    //dft_c2r(0);
}



slab_bc :: ~slab_bc()
{
    cout << "Deleting slab" << endl;
}
