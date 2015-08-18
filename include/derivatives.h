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
//
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
// invert two dimensional laplace equation.
// In spectral space, 
//                              / 4 pi^2 ((ky/Ly)^2 + (kx/Lx)^2 )  for ky, kx  <= N/2
// phi(ky, kx) = omega(ky, kx)  
//                              \ 4 pi^2 (((ky-My)/Ly)^2 + (kx/Lx)^2) for ky > N/2 and kx <= N/2
//
// and phi(0,0) = 0 (to avoid division by zero)
// Divide into 4 sectors:
//
//            Nx/2 + 1
//         ^<------>|
//  My/2+1 |        |
//         |   I    |
//         |        |
//         v        |
//         ==========
//         ^        |
//         |        |
//  My/2-1 |  II    |
//         |        |
//         v<------>|
//           Nx/2 + 1
//
// 
// sector I    : ky <= My/2, kx < Nx/2   BS = (cuda_blockdim_nx, 1), GS = (((Nx / 2 + 1) + cuda_blockdim_nx - 1) / cuda_blockdim_nx, My / 2 + 1)
// sector II   : ky >  My/2, kx < Nx/2   BS = (cuda_blockdim_nx, 1), GS = (((Nx / 2 - 1) + cuda_blockdim_nx - 1) / cuda_blockdim_nx, My / 2 - 1)

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


template <typename T>
__global__
void d_compute_laplace(CuCmplx<T>* in_arr, CuCmplx<T>* out_arr, CuCmplx<T>* kmap, const uint My, const uint Nx21)
{
	const uint row = blockIdx.y * blockDim.y + threadIdx.y;
	const uint col = blockIdx.x * blockDim.x + threadIdx.x;
    const uint idx = row * Nx21 + col;
    
    if(row < My && col < Nx21)
        out_arr[idx] = in_arr[idx] * (kmap[idx].re() + kmap[idx].im());
}


template <typename T>
__global__
void d_inv_laplace(CuCmplx<T>* in_arr, CuCmplx<T>* out_arr, CuCmplx<T>* kmap, const uint My, const uint Nx21)
{
	const uint row = blockIdx.y * blockDim.y + threadIdx.y;
	const uint col = blockIdx.x * blockDim.x + threadIdx.x;
    const uint idx = row * Nx21 + col;
    
    if(row < My && col < Nx21)
        out_arr[idx] = in_arr[idx] / (kmap[idx].re() + kmap[idx].im());
}


template <typename T>
__global__
void d_inv_laplace_zero(CuCmplx<T>* out)
{
    out[0] = cuda::cmplx_t(0.0, 0.0);
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


// Class to be used to calculate spectral derivatives on the slab, defined by
// slab_layout.
//
// typename T is either double or float
//
//
template <typename T>
class derivs
{
    public:
        derivs(const cuda::slab_layout_t);
        ~derivs();

        /// @brief Compute first order x- and y-derivatives
        /// @detailed Allocates memory for Fourier coefficients.
        /// @detailed If spectral representation is available, use
        /// @d_dx1_dy1 where they are passed as arguments instead
        void d_dx1_dy1(cuda_array<T>&, cuda_array<T>&, cuda_array<T>&);
        void d_dx1_dy1(const cuda_array<CuCmplx<T> >&,  cuda_array<CuCmplx<T> >&, cuda_array<CuCmplx<T> >&, const uint);
        void d_dx1_dy1(const cuda_array<CuCmplx<T> >*,  cuda_array<CuCmplx<T> >*, cuda_array<CuCmplx<T> >*, const uint);
        /// @brief Compute second order x- and y-derivatives
        void d_dx2_dy2(cuda_array<T>&, cuda_array<T>&, cuda_array<T>&);
        void d_dx2_dy2(cuda_array<CuCmplx<T> >&,  cuda_array<CuCmplx<T> >&, cuda_array<CuCmplx<T> >&, const uint);
        /// @brief Compute Laplacian
        void d_laplace(cuda_array<T>&, cuda_array<T>&, const uint);
        void d_laplace(cuda_array<CuCmplx<T> >&, cuda_array<CuCmplx<T> >&, const uint);
        void d_laplace(cuda_array<CuCmplx<T> >*, cuda_array<CuCmplx<T> >*, const uint);
        /// @brief Invert Laplace equation
        void inv_laplace(cuda_array<T>&, cuda_array<T>&, const uint);
        void inv_laplace(cuda_array<CuCmplx<T> >&, cuda_array<CuCmplx<T> >&, const uint);
        void inv_laplace(cuda_array<CuCmplx<T> >*, cuda_array<CuCmplx<T> >*, const uint);

        void dft_r2c(T* in, CuCmplx<T>* out);
        void dft_c2r(CuCmplx<T>* in, T* out);

    private:
        const unsigned int Nx;
        const unsigned int My;
        const T Lx;
        const T Ly;
        const T dx;
        const T dy;

        dim3 grid_my_nx21;
        dim3 block_my_nx21;

        cuda_array<CuCmplx<T> > kmap_dx1_dy1;
        cuda_array<CuCmplx<T> > kmap_dx2_dy2;

        cufftHandle plan_r2c;
        cufftHandle plan_c2r;

        void init_dft();
};


#ifdef __CUDACC__

template <typename T>
derivs<T> :: derivs(const cuda::slab_layout_t sl) :
    Nx(sl.Nx), My(sl.My),
    Lx(T(sl.Nx) * T(sl.delta_x)),
    Ly(T(sl.My) * T(sl.delta_y)),
    dx(T(sl.delta_x)), dy(T(sl.delta_y)),
    kmap_dx1_dy1(1, My, Nx / 2 + 1),
    kmap_dx2_dy2(1, My, Nx / 2 + 1)
{
    init_dft();
    // Generate first and second derivative map;
    gen_kmap_dx1_dy1<<<kmap_dx1_dy1.get_grid(), kmap_dx1_dy1.get_block()>>>(kmap_dx1_dy1.get_array_d(), cuda::TWOPI / Lx,
                                                                            cuda::TWOPI / Ly, My, Nx / 2 + 1);
    gen_kmap_dx2_dy2<<<kmap_dx2_dy2.get_grid(), kmap_dx2_dy2.get_block()>>>(kmap_dx2_dy2.get_array_d(), cuda::TWOPI / Lx,
                                                                            cuda::TWOPI / Ly, My, Nx / 2 + 1);
    //ostream of;
    //of.open("k2map.dat");
    //of << kmap_dx2_dy2 << endl;
    //of.close()
    gpuStatus();
}


// Generic template for DFT planning, should not be called. But
// standard is double precision
template <typename T> inline void plan_dft_r2c(cufftHandle& plan, int My, int Nx, cufftResult& err, const T dummy)
{
    //err = cufftPlan2d(&plan_r2c, My, Nx, CUFFT_D2Z);
}

template <typename T> inline void plan_dft_c2r(cufftHandle& plan, int My, int Nx, cufftResult& err, const T dummy)
{
    //err = cufftPlan2d(&plan_r2c, My, Nx, CUFFT_Z2D);
}


// DFT r2c planning for double precision
template <> inline void plan_dft_r2c<cuda::real_t>(cufftHandle& plan_r2c, int My, int Nx, cufftResult& err, const cuda::real_t dummy)
{
    err = cufftPlan2d(&plan_r2c, My, Nx, CUFFT_D2Z);
}

// DFT c2r planning for double precision
template <> inline void plan_dft_c2r<cuda::real_t>(cufftHandle& plan_c2r, int My, int Nx, cufftResult& err, const cuda::real_t dummy)
{
    err = cufftPlan2d(&plan_c2r, My, Nx, CUFFT_Z2D);
}

// DFT r2c planning for single precision
template <> inline void plan_dft_r2c<float>(cufftHandle& plan_r2c, int My, int Nx, cufftResult& err, const float dummy)
{
    err = cufftPlan2d(&plan_r2c, My, Nx, CUFFT_R2C);
}

// DFT c2r planning for single precision
template <> inline void plan_dft_c2r<float>(cufftHandle& plan_c2r, int My, int Nx, cufftResult& err, const float dummy)
{
    err = cufftPlan2d(&plan_c2r, My, Nx, CUFFT_C2R);
}


// Calls the explicit function templates specizalizations above
// Create dummy variable, so that we can use type inference to call the
// correct planning routine, i.e. CUFFT_R2C/C2R for float and
// CUFFT_D2Z, Z2D for double
template<typename T>
void derivs<T> :: init_dft()
{
    cufftResult err;
    T dummy{0.0};
    plan_dft_r2c(plan_r2c, My, Nx, err, dummy);
    if(err != CUFFT_SUCCESS)
    {
        stringstream err_str;
        err_str << "Error planning D2Z DFT: " << err << endl;
        throw gpu_error(err_str.str());
    }

    plan_dft_c2r(plan_c2r, My, Nx, err, dummy);
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

// Generic function template for r2c transformation
template <typename T>
inline void call_dft_r2c(cufftHandle& plan, T* in, CuCmplx<T>* out, cufftResult& err, T dummy)
{
    err = cufftExecD2Z(plan, in, (cufftDoubleComplex*) out);
}

// Inline template specialization, called from derivs::dft_r2c for T = cuda::real_t
template <>
inline void call_dft_r2c<cuda::real_t>(cufftHandle& plan, cuda::real_t* in, CuCmplx<cuda::real_t>* out, cufftResult& err, cuda::real_t dummy)
{
    err = cufftExecD2Z(plan, in, (cufftDoubleComplex*) out);
}

// Inline template specialization, called from derivs::dft_r2c for T = float
template <>
inline void call_dft_r2c<float>(cufftHandle& plan, float* in, CuCmplx<float>* out, cufftResult& err, float dummy)
{
    err = cufftExecR2C(plan, in, (cufftComplex*) out);
}


template <typename T>
void derivs<T> :: dft_r2c(T* in, CuCmplx<T>* out)
{
#ifdef DEBUG
    cerr << "derivs<T> :: dft_r2c " << endl;
    cerr << "\treal data at " << in << "\tcomplex data at " << out << endl;
#endif
    // compute dft of each field 
    cufftResult err;
    T dummy{0.0};
    call_dft_r2c(plan_r2c, in, out, err, dummy);
    if(err != CUFFT_SUCCESS)
    {
        stringstream err_str;
        err_str << "Error planning D2Z DFT: " << cufftGetErrorString.at(err) << endl;
        throw gpu_error(err_str.str());
    }

    // Verify Parseval's theorem
#ifdef DEBUG
    T* in_h = new T[My * Nx];
    CuCmplx<T>* out_h = new CuCmplx<T>[My * (Nx / 2 + 1)];

    gpuErrchk(cudaMemcpy(in_h, in, sizeof(T) * My * Nx, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(out_h, out, sizeof(CuCmplx<T>) * My * (Nx / 2 + 1), cudaMemcpyDeviceToHost));
    T sum_r2{0.0};
    T sum_c2{0.0};

    int idx;
    for(uint m = 0; m < My; m++)
    {
        for(uint n = 0; n < Nx; n++)
        {
            sum_r2 += in_h[(m * Nx) + n] * in_h[(m * Nx) + n];
            // Take care of hermitian symmetry
            if(n < Nx / 2 + 1)
                idx = m * (Nx / 2 + 1) + n;
            else
                idx = ((My - m) % My) * (Nx / 2 + 1) + (Nx - n);

            sum_c2 += out_h[idx].abs() * out_h[idx].abs();
        }
    }
    sum_c2 = sum_c2 / T(My * Nx);
    T rel_err;

    // check if sums are not zero and are good numbers
    bool good_sum = true;
    if(std::isnan(sum_r2))
        good_sum = false;
    if(std::isinf(sum_r2))
        good_sum = false;
    if(std::isnan(sum_c2))
        good_sum = false;
    if(std::isinf(sum_c2))
        good_sum = false;

    if(sum_r2 > 1e-10 && sum_c2 > 1e-10)
       rel_err = abs(sum_r2 - sum_c2) / abs(sum_r2);
    else
        rel_err = 0.0;

    if(rel_err > 1e-8)
        good_sum = false;


    // In case, we did not take the dft of zero, checkt he relative error
    //if(abs(sum_r2) > 1e-10 && abs(sum_c2) > 1e-10)
    //{

    //    T rel_err = abs(sum_r2 - sum_c2) / abs(sum_r2);
    //    if(rel_err > 1e-8)
    //    {
    if(!good_sum)
    {
        cerr << "!!!!!!!!!!!!!!! dft_r2c: sum_r2 = " << sum_r2 << "\tsum_c2 / (N*M)= " << sum_c2 << "\tRel. err: " << rel_err << endl;
        ofstream of_r, of_c;
        of_r.open("derivs_arr_r.dat", ios::trunc);
        of_c.open("derivs_arr_c.dat", ios::trunc);

        for(uint m = 0; m < My; m++)
        {
            for(uint n = 0; n < Nx; n++)
            {
                of_r << in_h[(m * Nx) + n] << "\t";
                if(n < Nx / 2 + 1)
                    of_c << out_h[m * (Nx / 2 + 1) + n] << "\t";
            }
            of_r << "\n";
            of_c << "\n";
        }
        of_r.close();
        of_c.close();

        stringstream err_msg;
        err_msg << "Error in dft_r2c" << endl;
        err_msg << "real data at " << in << "\tcomplex data at " << out << endl;
        err_msg << "Rel. error in parsevals theorem: " << rel_err << endl;

        throw assert_error(err_msg.str());
    }
    else
    {
        cerr << "                dft_r2c: sum_r2 = " << sum_r2 << "\tsum_c2 / (N*M)= " << sum_c2 << "\tRel. err: " << rel_err << endl;
    }
    //assert(rel_err < 1e-3);
    delete [] in_h;
    delete [] out_h;
#endif // DEBUG

}

// Generic function template for r2c transformation
template <typename T>
inline void call_dft_c2r(cufftHandle& plan, CuCmplx<T>* in, T* out, cufftResult& err)
{
    err = cufftExecZ2D(plan, in, (cufftDoubleComplex*) out);
    if(err != CUFFT_SUCCESS)
    {
        stringstream err_str;
        err_str << "Error executing cufftExecZ2D: " << cufftGetErrorString.at(err) << endl;
        throw gpu_error(err_str.str());
    }
}

// Inline template specialization, called from derivs::dft_r2c for T = cuda::real_t
template <>
inline void call_dft_c2r<cuda::real_t>(cufftHandle& plan, CuCmplx<cuda::real_t>* in, cuda::real_t* out, cufftResult& err)
{
    err = cufftExecZ2D(plan, (cufftDoubleComplex*) in, out);
    if(err != CUFFT_SUCCESS)
    {
        stringstream err_str;
        err_str << "Error executing cufftExecZ2D: " << cufftGetErrorString.at(err) << endl;
        throw gpu_error(err_str.str());
    }
}

// Inline template specialization, called from derivs::dft_r2c for T = float
template <>
inline void call_dft_c2r<float>(cufftHandle& plan, CuCmplx<float>* in, float* out, cufftResult& err)
{
    err = cufftExecC2R(plan,(cufftComplex*) in, out);
    if(err != CUFFT_SUCCESS)
    {
        stringstream err_str;
        err_str << "Error executing cufftExecZ2D: " << cufftGetErrorString.at(err) << endl;
        throw gpu_error(err_str.str());
    }
}



template <typename T>
void derivs<T> :: dft_c2r(CuCmplx<T>* in, T* out)
{
#ifdef DEBUG
    cerr << "derivs<T> :: dft_c2r " << endl;
    cerr << "\tcomplex data at " << in << "\treal data at " << out << endl;
#endif
    // compute dft of each field 
    cufftResult err;
    call_dft_c2r(plan_c2r, in, out, err);
    if(err != CUFFT_SUCCESS)
    {
        stringstream err_str;
        err_str << "Error planning D2Z DFT: " << cufftGetErrorString.at(err) << endl;
        throw gpu_error(err_str.str());
    }
    // Verify Parseval's theorem
#ifdef DEBUG
    CuCmplx<T>* in_h = new CuCmplx<T>[My * (Nx / 2 + 1)];
    T* out_h = new T[My * Nx];

    gpuErrchk(cudaMemcpy(in_h, in, sizeof(CuCmplx<T>) * My * (Nx / 2 + 1), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(out_h, out, sizeof(T) * My * Nx, cudaMemcpyDeviceToHost));
    T sum_r2{0.0};
    T sum_c2{0.0};

    int idx;
    for(uint m = 0; m < My; m++)
    {
        for(uint n = 0; n < Nx; n++)
        {
            sum_r2 += out_h[(m * Nx) + n] * out_h[(m * Nx) + n];
            // Take care of hermitian symmetry
            if(n < Nx / 2 + 1)
                idx = m * (Nx / 2 + 1) + n;
            else
                idx = ((My - m) % My) * (Nx / 2 + 1) + (Nx - n);

            sum_c2 += in_h[idx].abs() * in_h[idx].abs();
        }
    }

    // Account for normalization. Do two divisions, to avoid division overflow:
    // const unsigned int Nx, My;  <- max value is 256*256*256*256-1 !
    sum_r2 = sum_r2 / T(Nx * My);
    sum_r2 = sum_r2 / T(Nx * My);
    sum_c2 = sum_c2 / T(Nx * My);
    T rel_err;

    // check if sums are not zero and are good numbers
    bool good_sum = true;
    if(std::isnan(sum_r2))
        good_sum = false;
    if(std::isinf(sum_r2))
        good_sum = false;
    if(std::isnan(sum_c2))
        good_sum = false;
    if(std::isinf(sum_c2))
        good_sum = false;

    if(sum_r2 > 1e-10 && sum_c2 > 1e-10)
       rel_err = abs(sum_r2 - sum_c2) / abs(sum_r2);
    else
        rel_err = 0.0;

    if(rel_err > 1e-8)
        good_sum = false;

    // check relative error in case we didn't take DFT of zero
    //if(abs(sum_r2) > 1e-10 && abs(sum_c2) > 1e-10)
    //{
    //    T rel_err = abs(sum_r2 - sum_c2) / abs(sum_r2);
    //    if(rel_err > 1e-8)
    if(!good_sum)
    {
        cerr << "!!!!!!!!!!!!!!! dft_c2r: sum_r2 = " << sum_r2 << "\tsum_c2 / (N*M)= " << sum_c2 << "\tRel. err: " << rel_err << endl;
        ofstream of_r, of_c;
        of_r.open("derivs_arr_r.dat", ios::trunc);
        of_c.open("derivs_arr_c.dat", ios::trunc);

        for(uint m = 0; m < My; m++)
        {
            for(uint n = 0; n < Nx; n++)
            {
                of_r << out_h[(m * Nx) + n] << "\t";
                if(n < Nx / 2 + 1)
                    of_c << in_h[m * (Nx / 2 + 1) + n] << "\t";
            }
            of_r << "\n";
            of_c << "\n";
        }
        of_r.close();
        of_c.close();

        stringstream err_msg;
        err_msg << "Error in dft_c2r" << endl;
        err_msg << "real data at " << in << "\tcomplex data at " << out << endl;
        err_msg << "Rel. error in parsevals theorem: " << rel_err << endl;
        throw assert_error(err_msg.str());
    }
    else
    {
        cerr << "                dft_c2r: sum_r2 = " << sum_r2 << "\tsum_c2 / (N*M)= " << sum_c2 << "\tRel. err: " << rel_err << endl;
    }

    delete [] in_h;
    delete [] out_h;
#endif // DEBUG
}


// Compute spectral derivatives of in
template <typename T>
void derivs<T> :: d_dx1_dy1(cuda_array<T>& in,
                            cuda_array<T>& out_x,
                            cuda_array<T>& out_y)
{
    static cuda_array<CuCmplx<T> > in_hat(1, My, Nx / 2 + 1);
    static cuda_array<CuCmplx<T> > out_x_hat(1, My, Nx / 2 + 1);
    static cuda_array<CuCmplx<T> > out_y_hat(1, My, Nx / 2 + 1);
  
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
    static cuda_array<CuCmplx<T> > in_hat(1, My, Nx / 2 + 1);
    static cuda_array<CuCmplx<T> > out_x_hat(1, My, Nx / 2 + 1);
    static cuda_array<CuCmplx<T> > out_y_hat(1, My, Nx / 2 + 1);
  
    dft_r2c(in.get_array_d(), in_hat.get_array_d());
   
    d_dx2_dy2(in_hat, out_x_hat, out_y_hat, 0);

    dft_c2r(out_x_hat.get_array_d(), out_x.get_array_d());
    dft_c2r(out_y_hat.get_array_d(), out_y.get_array_d());

    out_x.normalize();
    out_y.normalize();
}

// Call derivation kernel
template <typename T>
void derivs<T> :: d_dx1_dy1(const cuda_array<CuCmplx<T> >& in_hat,
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

// Call derivation kernel
template <typename T>
void derivs<T> :: d_dx1_dy1(const cuda_array<CuCmplx<T> >* in_hat,
        cuda_array<CuCmplx<T> >* out_x_hat,
        cuda_array<CuCmplx<T> >* out_y_hat,
        const uint t_src)
{
    //int elem_per_thread{4};
    //size_t shmem_size = 2 * elem_per_thread * block_my_nx21.x * sizeof(CuCmplx<T>);

    d_compute_dx_dy<<<in_hat -> get_grid(), in_hat -> get_block()>>>(in_hat -> get_array_d(t_src),
            out_x_hat -> get_array_d(),
            out_y_hat -> get_array_d(),
            kmap_dx1_dy1.get_array_d(), 
            1, My, Nx / 2 + 1);
    //d_compute_dx_dy<elem_per_thread><<<in_hat.get_grid(), in_hat.get_block(), shmem_size>>>
    //    (in_hat.get_array_d(t_src), out_x_hat.get_array_d(), out_y_hat.get_array_d(),
    //     kmap_dx1_dy1.get_array_d(), 1, My, Nx / 2 + 1);
}


template <typename T>
void derivs<T> :: d_dx2_dy2(cuda_array<CuCmplx<T> >& in_hat,
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


template <typename T>
void derivs<T> :: d_laplace(cuda_array<T>& in, cuda_array<T>& out, const uint t_src)
{
    static cuda_array<CuCmplx<T> > in_hat(1, My, Nx / 2 + 1);
    static cuda_array<CuCmplx<T> > out_hat(1, My, Nx / 2 + 1);
  
    dft_r2c(in.get_array_d(t_src), in_hat.get_array_d());
 
    d_laplace(in_hat, out_hat, t_src);    

    dft_c2r(out_hat.get_array_d(), out.get_array_d());

    out.normalize();
}

template <typename T>
void derivs<T> :: d_laplace(cuda_array<CuCmplx<T> >* in, cuda_array<CuCmplx<T> >* out, const uint t_src)
{
    d_compute_laplace<<<in -> get_grid(), in -> get_block()>>>(in -> get_array_d(t_src), 
                                                               out -> get_array_d(), 
                                                               kmap_dx2_dy2.get_array_d(), My, Nx / 2 + 1);
    d_inv_laplace_zero<<<1, 1>>>(out -> get_array_d());
}


template <typename T>
void derivs<T> :: d_laplace(cuda_array<CuCmplx<T> >& in, cuda_array<CuCmplx<T> >& out, const uint t_src)
{
    d_compute_laplace<<<in.get_grid(), in.get_block()>>>(in.get_array_d(t_src), 
                                                         out.get_array_d(), 
                                                         kmap_dx2_dy2.get_array_d(), My, Nx / 2 + 1);
    gpuStatus();
    d_inv_laplace_zero<<<1, 1>>>(out.get_array_d());
    gpuStatus();
}

template <typename T>
void derivs<T> :: inv_laplace(cuda_array<T>& in, cuda_array<T>& out, const uint t_src)
{
    cuda_array<CuCmplx<T> > in_hat(1, My, Nx / 2 + 1);
    cuda_array<CuCmplx<T> > out_hat(1, My, Nx / 2 + 1);

    dft_r2c(in.get_array_d(t_src), in_hat.get_array_d());

    d_inv_laplace<<<in_hat.get_grid(), in_hat.get_block()>>>(in_hat.get_array_d(), 
                                                             out_hat.get_array_d(), 
                                                             kmap_dx2_dy2.get_array_d(), My, Nx / 2 + 1);
    gpuStatus();
    d_inv_laplace_zero<<<1, 1>>>(out_hat.get_array_d());
    gpuStatus();

    dft_c2r(out_hat.get_array_d(), out.get_array_d());
    out.normalize();
}


template <typename T>
void derivs<T> :: inv_laplace(cuda_array<CuCmplx<T> >* in, cuda_array<CuCmplx<T> >* out, const uint t_src)
{
    //cout << "derivs :: inv_laplace(cuda_array<..>* in, cuda_array<..>* out,...)" << endl;
    //cout << "derivs::inv_laplace... t_src = " << t_src << endl;
    //cout << "in_arr at " << in -> get_array_d(t_src) << "\tout_arr at " << out -> get_array_d() << endl;

    d_inv_laplace<<<in -> get_grid(), in -> get_block()>>>(in -> get_array_d(t_src), 
                                                           out -> get_array_d(), 
                                                           kmap_dx2_dy2.get_array_d(), My, Nx / 2 + 1);
    gpuStatus();
    d_inv_laplace_zero<<<1, 1>>>(out -> get_array_d());
    gpuStatus();
}


template <typename T>
void derivs<T> :: inv_laplace(cuda_array<CuCmplx<T> >& in, cuda_array<CuCmplx<T> >& out, const uint t_src)
{
    //cout << "derivs :: inv_laplace(cuda_array<..>& in, cuda_array<..>& out,...)" << endl;
    //cout << "derivs::inv_laplace... t_src = " << t_src << endl;
    //cout << "in_arr at " << in.get_array_d(t_src) << "\tout_arr at " << out.get_array_d() << endl;

    d_inv_laplace<<<in.get_grid(), in.get_block()>>>(in.get_array_d(t_src), 
                                                     out.get_array_d(), 
                                                     kmap_dx2_dy2.get_array_d(), My, Nx / 2 + 1);
    gpuStatus();
    d_inv_laplace_zero<<<1, 1>>>(out.get_array_d());
    gpuStatus();
}





#endif // CUDACC
#endif // DERIVS_H
