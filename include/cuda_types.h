#ifndef CUDA_TYPES
#define CUDA_TYPES

#include <cuda.h>
#include "cufft.h"
#include "cucmplx.h"
#include <map>

#ifdef __CUDACC__
#define CUDAMEMBER __host__ __device__
#else
#define CUDAMEMBER
#endif

namespace cuda
{
    using real_t = double;
    constexpr unsigned int blockdim_col{32}; ///< Block dimension for consecutive elements (y-direction)
    constexpr unsigned int blockdim_row{16}; ///< Block dimension for non-consecutive elements (x-direction)

    constexpr unsigned int blockdim_nx_max{1024};
    constexpr unsigned int blockdim_my_max{1024};

    constexpr unsigned int griddim_nx_max{1024};
    constexpr unsigned int griddim_my_max{1024};

    constexpr size_t num_pad_y{2};                      ///< Number of rows to pad for in-place DFT

#ifdef _CUFFT_H_
    const std::map<cufftResult, std::string> cufftGetErrorString
    {
        {CUFFT_SUCCESS, std::string("CUFFT_SUCCESS")},
        {CUFFT_INVALID_PLAN, std::string("CUFFT_INVALID_PLAN")},
        {CUFFT_ALLOC_FAILED, std::string("CUFFT_ALLOC_FAILED")},
        {CUFFT_INVALID_TYPE, std::string("CUFFT_INVALID_TYPE")},
        {CUFFT_INVALID_VALUE, std::string("CUFFT_INVALID_VALUE")},
        {CUFFT_INTERNAL_ERROR, std::string("CUFFT_INTERNAL_ERROR")},
        {CUFFT_EXEC_FAILED, std::string("CUFFT_EXEC_FAILED")},
        {CUFFT_SETUP_FAILED, std::string("CUFFT_SETUP_FAILED")},
        {CUFFT_INVALID_SIZE, std::string("CUFFT_INVALID_SIZE")},
        {CUFFT_UNALIGNED_DATA, std::string("CUFFT_UNALIGNED_DATA")}
    };
#endif // _CUFFT_H_

#ifdef __CUDACC__
    struct thread_idx
    {
        static __device__ size_t get_col() {return(blockIdx.x * blockDim.x + threadIdx.x);}
        static __device__ size_t get_row() {return(blockIdx.y * blockDim.y + threadIdx.y);}

        __device__ thread_idx(){}
    };

    __constant__ const real_t ss3_alpha_d[3][3] = {{1., 0., 0.}, {2., -0.5, 0.}, {3., -1.5, 1./3.}};
    __constant__ const real_t ss3_beta_d[3][3]  = {{1., 0., 0.}, {2., -1. , 0.}, {3., -3.,  1.  }};
#endif //__CUDACC

//    //constexpr real_t ss3_alpha_r[3][4] = {{1.0, 1.0, 0.0, 0.0}, {1.5, 2.0, -0.5, 0.0}, {11./6., 3., -1.5, 1./3.}}; ///< Coefficients for implicit part in time integration
//    constexpr real_t ss3_alpha_r[3][3] = {{1., 0., 0.}, {2., -0.5, 0.}, {3., -1.5, 1./3.}}; ///< Coefficients for implicit part in time integration
//    constexpr real_t ss3_beta_r[3][3] =  {{1., 0., 0.}, {2., -1. , 0.}, {3., -3.,  1.   }}; ///< Coefficients for explicit part in time integration
};

#endif //CUDA_TYPES
