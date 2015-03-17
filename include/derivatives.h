/*******************************************************************************
 *
 * Implementation of spectral derivatives
 *
 * Initial version 2015-03-16
 *
 ******************************************************************************/

#ifndef DERIVS_H
#define DERIVS_H


#include <iostream>
#include <fstream>
#include <cmath>
#include "error.h"
#include "cucmplx.h"
#include "cuda_array4.h"
#include "cuda_types.h"
#include "cufft.h"


#ifdef __CUDACC__
/*****************************************************************************
 *
 * Kernel implementation
 *
  ****************************************************************************/

//    d/dy: Compute spectral y-derivative for frequencies 0 .. My/2 - 1,
//          stored in the columns 0..My/2-1
//    Call:  d_d_dy_lo<<<grid_dy_half, block_nx21>>>(arr_in -> get_array_d(t_src), arr_out -> get_array_d(0), My, Nx21, Ly);
//    block_nx21    = dim3(cuda::blockdim_x, 1)
//    grid_dy_half  = dim3(((Nx / 2 + 1) + cuda::blockdim_nx - 1) / cuda::blockdim_nx, My / 2)


// Generate multiplicators to use for x- and y-derivatives
// kmap[index].re = kx
// kmap[index].im = ky
//
// then: theta_x_hat[index] = theta_hat[index] * complex(0.0, kmap[index].re())
//       theta_y_hat[index] = theta_hat[index] * complex(0.0, kmap[index].im())
//
template <typename T>
__global__
void gen_kmap_dx1_dy1(CuCmplx<T>* kmap, const T two_pi_Lx, const T two_pi_Ly,
                       const uint My, const uint Nx21)
{
    const uint row = blockIdx.y * blockDim.y + threadIdx.y;
    const uint col = blockIdx.x * blockDim.x + threadIdx.x;
    const uint index = row * Nx21 + col;

    CuCmplx<T> tmp(0.0, 0.0);

    if(row < My / 2)
        tmp.set_im(two_pi_Ly * double(row));
    else if (row == My / 2)
        tmp.set_im(0.0);
    else
        tmp.set_im(two_pi_Ly * (double(row) - double(My)));

    if(col < Nx21 - 1)
        tmp.set_re(two_pi_Lx * double(col));
    else
        tmp.set_re(0.0);

    if((col < Nx21) && (row < My))
        kmap[index] = tmp;
}



// Generate wave-number coefficients. These can be used for second order-derivatives
// in diffusion terms, computed in stiffk. See "Notes on FFT-based differentiation
// S.G. Johnson
template <typename T>
__global__
void gen_kmap_dx2_dy2(CuCmplx<T>* k2map, const T two_pi_Lx, const T two_pi_Ly,
                       const uint My, const uint Nx21)
{
    const uint row = blockIdx.y * blockDim.y + threadIdx.y;
    const uint col = blockIdx.x * blockDim.x + threadIdx.x;
    const uint index = row * Nx21 + col;

    CuCmplx<T> tmp(0.0, 0.0);

    if(row < My / 2 + 1)
        tmp.set_im(-1.0 * two_pi_Ly * two_pi_Ly * double(row * row));
    else
        tmp.set_im(-1.0 * two_pi_Ly * two_pi_Ly * (double(row) - double(My)) * (double(row) - double(My)));

    if(col < Nx21)
        tmp.set_re(-1.0 * two_pi_Lx * two_pi_Lx * double(col * col));

    if((col < Nx21) && (row < My))
        k2map[index] = tmp;
}

// Elementwise multiplication of
// in_arr[index] * kmap[index] for spatial derivatives
//
// kmap is coefficient wise, i.e. pass
// kmap_dx1_dy1 for 1st derivatives and
// kmap_dx2_dy2 for 2nd derivatives
//
// Load T elements of in_arr and kmap in shared memory,
// each thread processes T elements.
// see tests/test_perf/test_perf_derivs_map.cu for benchmarking


// Create mode multiplier for first derivative,
// i.e. d_x -> * i k

template <typename T>
__global__
void d_compute_dx_dy(CuCmplx<T>* in_arr, 
        CuCmplx<T>* out_x_arr, 
        CuCmplx<T>* out_y_arr,
		CuCmplx<T>* kmap, 
        int order, const uint My, const uint Nx21)
{
    //extern __shared__ CuCmplx<cuda::real_t> shmem[];
	const uint row = blockIdx.y * blockDim.y + threadIdx.y;
	const uint col = blockIdx.x * blockDim.x + threadIdx.x;
    const uint idx = row * Nx21 + col;
    
    if(row < My && col < Nx21)
    {
        if(order == 1)
        {
            out_x_arr[idx] = in_arr[idx] * CuCmplx<T>(0.0, kmap[idx].re());
            out_y_arr[idx] = in_arr[idx] * CuCmplx<T>(0.0, kmap[idx].im());
        } else if (order == 2)
        {
            out_x_arr[idx] = in_arr[idx] * kmap[idx].re();
            out_y_arr[idx] = in_arr[idx] * kmap[idx].im();
        }
    }
}


template <typename T, int S>
__global__
void d_compute_dx_dy(CuCmplx<T>* in_arr, CuCmplx<T>* out_x_arr, CuCmplx<T>* out_y_arr,
		CuCmplx<T>* kmap, int order, const uint My, const uint Nx21)
{
	extern __shared__ CuCmplx<T> shmem[];
	const uint row_offset = (blockIdx.y * blockDim.y + threadIdx.y) * Nx21;
	uint col = blockIdx.x * blockDim.x + threadIdx.x;

	const uint offset_s = 2 * threadIdx.x;
	uint s;

//#pragma unroll 1
	for(s = 0; s < S; s++)
	{
		if(col < Nx21)
		{
			shmem[offset_s + 2 * s    ] = in_arr[row_offset + col];
			shmem[offset_s + 2 * s + 1] = kmap[row_offset + col];
		}
		col++;
	}
	col = blockIdx.x * blockDim.x + threadIdx.x;

//#pragma unroll 4
	for(s = 0; s < S; s++)
	{
		if(col < Nx21)
		{
            if (order == 1)
            {
                out_x_arr[row_offset + col] = shmem[offset_s + 2 * s] * CuCmplx<T>(0.0, shmem[offset_s + 2 * s + 1].re());
                out_y_arr[row_offset + col] = shmem[offset_s + 2 * s] * CuCmplx<T>(0.0, shmem[offset_s + 2 * s + 1].im());
            }
            else if (order == 2)
            {
                out_x_arr[row_offset + col] = shmem[offset_s + 2 * s] * CuCmplx<T>(shmem[offset_s + 2 * s + 1].re(), 0.0);
                out_y_arr[row_offset + col] = shmem[offset_s + 2 * s] * CuCmplx<T>(shmem[offset_s + 2 * s + 1].im(), 0.0);
            }
		}
		col++;
	}
}


#endif

template <typename T>
class derivs
{
    public:
        derivs(const cuda::slab_layout_t);
        ~derivs();

        /// @brief Compute first order x- and y-derivatives
        void d_dx1_dy1(cuda_array<T>&, cuda_array<T>&, cuda_array<T>&);
        inline void d_dx1_dy1(const cuda_array<CuCmplx<T> >&,  cuda_array<CuCmplx<T> >&, cuda_array<CuCmplx<T> >&, const uint);
        /// @brief Compute second order x- and y-derivatives
        void d_dx2_dy2(cuda_array<T>&, cuda_array<T>&, cuda_array<T>&);
        inline void d_dx2_dy2(cuda_array<CuCmplx<T> >&,  cuda_array<CuCmplx<T> >&, cuda_array<CuCmplx<T> >&, const uint);
        /// @brief Compute Laplacian
        void d_laplace(cuda_array<T>*, cuda_array<T>*, const uint);


    private:
        const unsigned int Nx;
        const unsigned int My;
        const T Lx;
        const T Ly;

        dim3 grid_my_nx21;
        dim3 block_my_nx21;

        cuda_array<CuCmplx<T> > kmap_dx1_dy1;
        cuda_array<CuCmplx<T> > kmap_dx2_dy2;

        cufftHandle plan_r2c;
        cufftHandle plan_c2r;

        void init_dft();

        void dft_r2c(T* in, CuCmplx<T>* out);
        void dft_c2r(CuCmplx<T>* in, T* out);
};


template <typename T>
derivs<T> :: derivs(const cuda::slab_layout_t sl) :
    Nx(sl.Nx), My(sl.My),
    Lx(T(sl.Nx) * T(sl.delta_x)),
    Ly(T(sl.My) * T(sl.delta_y)),
    kmap_dx1_dy1(1, My, Nx / 2 + 1),
    kmap_dx2_dy2(1, My, Nx / 2 + 1)
{
    init_dft();
    // Generate first and second derivative map;
    gen_kmap_dx1_dy1<<<kmap_dx1_dy1.get_grid(), kmap_dx1_dy1.get_block()>>>(kmap_dx1_dy1.get_array_d(), cuda::TWOPI / Lx,
                                                                            cuda::TWOPI / Ly, My, Nx / 2 + 1);
    gen_kmap_dx2_dy2<<<kmap_dx2_dy2.get_grid(), kmap_dx2_dy2.get_block()>>>(kmap_dx2_dy2.get_array_d(), cuda::TWOPI / Lx,
                                                                            cuda::TWOPI / Ly, My, Nx / 2 + 1);
    gpuStatus();
}


template<>
void derivs<cuda::real_t> :: init_dft()
{
    cufftResult err;
    err = cufftPlan2d(&plan_r2c, My, Nx, CUFFT_D2Z);
    if(err != CUFFT_SUCCESS)
    {
        stringstream err_str;
        err_str << "Error planning D2Z DFT: " << err << endl;
        throw gpu_error(err_str.str());
    }

    err = cufftPlan2d(&plan_c2r, My, Nx, CUFFT_Z2D);
    if (err != CUFFT_SUCCESS)
    {
        stringstream err_str;
        err_str << "Error planning D2Z DFT: " << err << endl;
        throw gpu_error(err_str.str());
    }
}


template <>
void derivs<float> :: init_dft()
{
    cufftResult err;
    err = cufftPlan2d(&plan_r2c, My, Nx, CUFFT_R2C);
    if(err != CUFFT_SUCCESS)
    {
        stringstream err_str;
        err_str << "Error planning D2Z DFT: " << err << endl;
        throw gpu_error(err_str.str());
    }

    err = cufftPlan2d(&plan_c2r, My, Nx, CUFFT_C2R);
    if (err != CUFFT_SUCCESS)
    {
        stringstream err_str;
        err_str << "Error planning D2Z DFT: " << err << endl;
        throw gpu_error(err_str.str());
    }
}



template <typename T>
derivs<T> :: ~derivs()
{
    cufftDestroy(plan_r2c);
    cufftDestroy(plan_c2r);
}


template <>
void derivs<cuda::real_t> :: dft_r2c(cuda::real_t* in, CuCmplx<cuda::real_t>* out)
{
    // compute dft of each field 
    static cufftResult err;
    err = cufftExecD2Z(plan_r2c, in, (cufftDoubleComplex*) out);
    if(err != CUFFT_SUCCESS)
    {
        stringstream err_str;
        err_str << "Error planning D2Z DFT: " << err << endl;
        throw gpu_error(err_str.str());
    }
}


template <>
void derivs<float> :: dft_r2c(float* in, CuCmplx<float>* out)
{
    // compute dft of each field 
    static cufftResult err;
    err = cufftExecR2C(plan_r2c, in, (cufftComplex*) out);
    if(err != CUFFT_SUCCESS)
    {
        stringstream err_str;
        err_str << "Error planning D2Z DFT: " << err << endl;
        throw gpu_error(err_str.str());
    }
}


template <>
void derivs<cuda::real_t> :: dft_c2r(cuda::cmplx_t* in, cuda::real_t* out)
{
    static cufftResult err;
    err = cufftExecZ2D(plan_c2r, (cufftDoubleComplex*) in, out);
    if(err != CUFFT_SUCCESS)
    {
        stringstream err_str;
        err_str << "Error planning D2Z DFT: " << err << endl;
        throw gpu_error(err_str.str());
    }
}


template <>
void derivs<float> :: dft_c2r(CuCmplx<float>* in, float* out)
{
    static cufftResult err;
    err = cufftExecC2R(plan_c2r, (cufftComplex*) in, out);
    if(err != CUFFT_SUCCESS)
    {
        stringstream err_str;
        err_str << "Error planning D2Z DFT: " << err << endl;
        throw gpu_error(err_str.str());
    }
}

// Compute spectral derivatives of in
template <typename T>
void derivs<T> :: d_dx1_dy1(cuda_array<T>& in,
                            cuda_array<T>& out_x,
                            cuda_array<T>& out_y)
{
    cuda_array<CuCmplx<T> > in_hat(1, My, Nx / 2 + 1);
    cuda_array<CuCmplx<T> > out_x_hat(1, My, Nx / 2 + 1);
    cuda_array<CuCmplx<T> > out_y_hat(1, My, Nx / 2 + 1);
  
    dft_r2c(in.get_array_d(), in_hat.get_array_d());
   
    d_dx1_dy1(in_hat, out_x_hat, out_y_hat, 0);

    dft_c2r(out_x_hat.get_array_d(), out_x.get_array_d());
    dft_c2r(out_y_hat.get_array_d(), out_y.get_array_d());

    out_x.normalize();
    out_y.normalize();
}


// Compute spectral derivatives of in
template <typename T>
void derivs<T> :: d_dx2_dy2(cuda_array<T>& in,
                            cuda_array<T>& out_x,
                            cuda_array<T>& out_y)
{
    cuda_array<CuCmplx<T> > in_hat(1, My, Nx / 2 + 1);
    cuda_array<CuCmplx<T> > out_x_hat(1, My, Nx / 2 + 1);
    cuda_array<CuCmplx<T> > out_y_hat(1, My, Nx / 2 + 1);
  
    dft_r2c(in.get_array_d(), in_hat.get_array_d());
   
    d_dx2_dy2(in_hat, out_x_hat, out_y_hat, 0);

    dft_c2r(out_x_hat.get_array_d(), out_x.get_array_d());
    dft_c2r(out_y_hat.get_array_d(), out_y.get_array_d());

    out_x.normalize();
    out_y.normalize();
}

// Call derivation kernel
template <typename T>
inline void derivs<T> :: d_dx1_dy1(const cuda_array<CuCmplx<T> >& in_hat,
        cuda_array<CuCmplx<T> >& out_x_hat,
        cuda_array<CuCmplx<T> >& out_y_hat,
        const uint t_src)
{
    //int elem_per_thread{4};
    //size_t shmem_size = 2 * elem_per_thread * block_my_nx21.x * sizeof(CuCmplx<T>);

    d_compute_dx_dy<<<in_hat.get_grid(), in_hat.get_block()>>>(in_hat.get_array_d(t_src),
            out_x_hat.get_array_d(),
            out_y_hat.get_array_d(),
            kmap_dx1_dy1.get_array_d(), 
            1, My, Nx / 2 + 1);
    //d_compute_dx_dy<elem_per_thread><<<in_hat.get_grid(), in_hat.get_block(), shmem_size>>>
    //    (in_hat.get_array_d(t_src), out_x_hat.get_array_d(), out_y_hat.get_array_d(),
    //     kmap_dx1_dy1.get_array_d(), 1, My, Nx / 2 + 1);
}


template <typename T>
inline void derivs<T> :: d_dx2_dy2(cuda_array<CuCmplx<T> >& in_hat,
        cuda_array<CuCmplx<T> >& out_x_hat,
        cuda_array<CuCmplx<T> >& out_y_hat,
        const uint t_src)
{
    //int elem_per_thread{4};
    //size_t shmem_size = 2 * elem_per_thread * block_my_nx21.x * sizeof(CuCmplx<T>);
    
    d_compute_dx_dy<<<in_hat.get_grid(), in_hat.get_block()>>>(in_hat.get_array_d(t_src),
            out_x_hat.get_array_d(),
            out_y_hat.get_array_d(),
            kmap_dx2_dy2.get_array_d(), 
            2, My, Nx / 2 + 1);

//    d_compute_dx_dy<elem_per_thread><<<in_hat.get_grid(), in_hat.get_block(), shmem_size>>>
//        (in_hat.get_array_d(t_src), out_x_hat.get_array_d(), out_y_hat.get_array_d(),
//         kmap1_dx_dy1.get_array_d(), 2, My, Nx / 2 + 1);
}

#endif
