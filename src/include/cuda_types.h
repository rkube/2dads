#ifndef CUDA_TYPES
#define CUDA_TYPES

#include <cuda.h>
#include "cufft.h"
#include "cucmplx.h"
#include <map>


namespace cuda
{
    using real_t = double;
    constexpr unsigned int blockdim_col{16}; ///< Block dimension for consecutive elements (y-direction)
    constexpr unsigned int blockdim_row{16}; ///< Block dimension for non-consecutive elements (x-direction)

    constexpr size_t elem_per_thread{8};

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

//#ifdef __CUDACC__
#if defined(__clang__) && defined(__CUDA__) && defined(__CUDA_ARCH__)
    struct thread_idx
    {
        static __device__ size_t get_col() {return(blockIdx.x * blockDim.x + threadIdx.x);}
        static __device__ size_t get_row() {return(blockIdx.y * blockDim.y + threadIdx.y);}

        __device__ thread_idx(){}
    };

    __constant__ const real_t ss3_alpha_d[3][3] = {{1., 0., 0.}, {2., -0.5, 0.}, {3., -1.5, 1./3.}};
    __constant__ const real_t ss3_beta_d[3][3]  = {{1., 0., 0.}, {2., -1. , 0.}, {3., -3.,  1.  }};
#endif //__CUDACC
};

#endif //CUDA_TYPES
