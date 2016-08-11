/*
 * cuda_array_bc.h
 *
 *  Created on: May 13, 2016
 *      Author: ralph
 *
 * Array type with variable padding in x- and y-direction.
 * Knows about boundary conditions, routines operating on this datatype are to compute
 * them on the fly
 *
 * Memory Layout
 *
 * rows: 0...My-1 ... My-1 + pad_y
 * cols: 0...Nx-1 ... Nx-1 + pad_x
 *
 *     0                        My-1 ... My - 1 + pad_y
 * Nx - 1 |--------- ... ------|    |
 *        |--------- ... ------|    |
 * ...
 *  0     |--------- ... ------|    |
 *
 * idx = n * (My + pad_y) + m
 *
 * columns (m, y-direction) are consecutive in memory
 *
 *
 * Mapping of CUDA threads on the array:
 *
 * Columns: 0..My - 1 + pad_y -> col = blockIdx.x * blockDim.x + threadIdx.x
 * Rows:    0..Nx - 1 + pad_x -> row = blockIdx.y * blockDim.y + threadIdx.y
 *
 * dimBlock = (blocksize_row, blocksize_col)
 * dimGrid = (My + pad_y) / blocksize_row, (My + pad_y) / blocksize_col
 *
 * Ghost points are to be computed on the fly, not stored in memory
 * They can be access by the address object
 *
 *
 */

#ifndef cuda_array_bc_H_
#define cuda_array_bc_H_

#include <iostream>
#include <iomanip>
#include <map>
#include <functional>
#include <sstream>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "bounds.h"
#include "address.h"
#include "error.h"
#include "cuda_types.h"
#include "allocators.h"


// Error checking macro for cuda calls
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess)
    {
        std::stringstream err_str;
        err_str << "GPUassert: " << cudaGetErrorString(code) << "\t file: " << file << ", line: " << line << "\n";
        throw gpu_error(err_str.str());
    }
}


// Verify last kernel launch
#define gpuStatus() { gpuVerifyLaunch(__FILE__, __LINE__); }
inline void gpuVerifyLaunch(const char* file, int line)
{
     cudaThreadSynchronize();
     cudaError_t error = cudaGetLastError();
     if(error != cudaSuccess)
     {
         std::stringstream err_str;
        err_str << "GPUassert: " << cudaGetErrorString(error) << "\t file: " << file << ", line: " << line << "\n";
        throw gpu_error(err_str.str());
     }
}


/// Device function to compute column and row
//__device__ inline size_t d_get_col() {
//    return (blockIdx.x * blockDim.x + threadIdx.x);
//}
//
//
//__device__ inline size_t d_get_row() {
//    return (blockIdx.y * blockDim.y + threadIdx.y);
//}


namespace device
{
#ifdef __CUDACC__
    /// Return true if row and column are within geom(excluding padded rows/cols)
    /// Return false if row or column is outside the geometry
    __device__ inline bool good_idx(size_t row, size_t col, const cuda::slab_layout_t geom)
    {
        return((row < geom.get_nx()) && (col < geom.get_my()));
    }


    template <typename T>
    __global__
    void kernel_set_tlev_ptr(T* data, T** tlev_ptr, const cuda::slab_layout_t geom)
    {
        for(size_t t = 0; t < geom.get_tlevs(); t++)
        {
            tlev_ptr[t] = &data[t * geom.get_nelem_per_t()];
            printf("I am kernel set_tlev_ptr: tlev_ptr[%u] at %p\n", t, tlev_ptr[t]);
        }
    }


    /// Apply the lambda op_func (with type given by template parameter O)
    /// op_func(T, size_t, size_t, slab_layout_t)
    template <typename T, typename O>
    __global__
    void kernel_apply(T* array_d_t, O device_func, const cuda::slab_layout_t geom) 
    {
        const size_t col{cuda :: thread_idx :: get_col()};
        const size_t row{cuda :: thread_idx :: get_row()};
        const size_t index{row * (geom.get_my() + geom.get_pad_y()) + col};

        if (good_idx(row, col, geom))
            array_d_t[index] = device_func(array_d_t[index], row, col, geom);
    }


    /// Perform element-wise arithmetic operation lhs[idx] = op(lhs[idx], rhs[idx])
    template<typename T, typename O>
    __global__
    void kernel_elementwise(T* lhs, T* rhs, O device_func, const cuda::slab_layout_t geom)
    {
        const size_t col{cuda :: thread_idx :: get_col()};
        const size_t row{cuda :: thread_idx :: get_row()};
        const size_t index{row * (geom.get_my() + geom.get_pad_y()) + col};

        if(good_idx(row, col, geom))
            lhs[index] = device_func(lhs[index], rhs[index]);
    }


    // For accessing elements in GPU kernels and interpolating ghost points
    template <typename T>
    __global__
    void kernel_init_address(address_t<T>** my_address, 
            const cuda::slab_layout_t geom, 
            const cuda::bvals_t<T> bvals)
    {
        *my_address = new address_t<T>(geom, bvals);
        printf("kernel_init_address: address_t at %p\n", *my_address);
    }


    template <typename T>
    __global__
    void kernel_use_address(const T* u, address_t<T>** address_u, const cuda::slab_layout_t geom)
    {
        printf("kernel_use_address: address_t at %p\n", address_u);
        const int col{static_cast<int>(cuda :: thread_idx :: get_col())};
        for(int n = 0; n < geom.get_nx(); n++)
        {
            for(int m = 0; m < geom.get_my(); m++)
            {
                printf("n=%d, m=%d: val = %f\n", n, m, (**address_u)(u, n, m));
            }
        }   
    }

    template <typename T>
    __global__
    void kernel_free_address(address_t<T>** my_address)
    {
        printf("kernel_free_address: address_t at %p\n", *my_address);
        delete *my_address;
    }


    template <typename T>
    __global__
    void kernel_advance_tptr(T** tlev_ptr, const size_t tlevs)
    {
        T* tmp{tlev_ptr[tlevs - 1]};
        for(size_t t = tlevs - 1; t > 0; t--)
        {
            tlev_ptr[t] = tlev_ptr[t - 1];
        }
        tlev_ptr[0] = tmp;
    }

    /// Reduction kernel, taken from cuda_darray.h
    // Perform reduction of in_data, stored in column-major order
    // Use stride_size = 1, offset_size = Nx for row-wise reduction (threads in one block reduce one row, i.e. consecutive elements of in_data)
    // row-wise reduction:
    // stride_size = 1
    // offset_size = Nx
    // blocksize = (Nx, 1)
    // gridsize = (1, My)
    //
    // column-wise reduction:
    // stride_size = My
    // offset_size = 1
    // blocksize = (My, 1)
    // gridsize = (1, Nx)
    template <typename T, typename O>
    __global__ void kernel_reduce(const T* __restrict__ in_data, 
                                  T* __restrict__ out_data, 
                                  O op_func, 
                                  const size_t stride_size, const size_t offset_size, const size_t Nx, const size_t My)
    {
        extern __shared__ T sdata[];

        const size_t tid = threadIdx.x;
        const size_t idx_data = tid * stride_size + blockIdx.y * offset_size;
        const size_t idx_out = blockIdx.y;
        if(idx_data < Nx * My)
        {
            sdata[tid] = in_data[idx_data];
            // reduction in shared memory
            __syncthreads();
            for(size_t s = 1; s < blockDim.x; s *= 2)
            {
                if(tid % (2*s) == 0)
                {
                    sdata[tid] = op_func(sdata[tid], sdata[tid + s]);
                }
                __syncthreads();
            }
            // write result for this block to global mem
            if (tid == 0)
            {
                //printf("threadIdx = %d: out_data[%d] = %f\n", threadIdx.x, row, sdata[0]);
                out_data[idx_out] = sdata[0];
            }
        }
    }
#endif // __CUDACC_
}

template <typename T, template <typename> class allocator> class cuda_array_bc_nogp;

// Host functions
namespace host
{
    template <typename T, typename O>
    void host_apply(T* data_ptr, O host_func, const cuda::slab_layout_t geom)
    {
        size_t index{0};
        for(size_t n = 0; n < geom.get_nx(); n++)
        {
            for(size_t m = 0; m < geom.get_my(); m++)
            {
                index = n * (geom.get_my() + geom.get_pad_y()) + m;
                data_ptr[index] = host_func(data_ptr[index], n, m, geom);
            }

        }
    }

    template <typename T, typename O>
    void host_elementwise(T* lhs, T* rhs, O host_func, const cuda::slab_layout_t geom)
    {
        size_t index{0};
        for(size_t n = 0; n < geom.get_nx(); n++)
        {   
            for(size_t m = 0; m < geom.get_my(); m++)
            {
                index = n * (geom.get_my() + geom.get_pad_y()) + m;
                lhs[index] = host_func(lhs[index], rhs[index]);
            }
        }
    }

}


namespace utility
{
    template <typename T>
    cuda_array_bc_nogp<T, allocator_host> create_host_vector(cuda_array_bc_nogp<T, allocator_device>& src)
    {
        std::cout << "Creating host vector\n";
        cuda_array_bc_nogp<T, allocator_host> res (src.get_geom(), src.get_bvals());
        std::cout << "Copying " << src.get_geom().get_tlevs() << "(" <<  res.get_geom().get_tlevs() << ") * " << src.get_geom().get_nelem_per_t() <<  "*" << sizeof(T) << " bytes" << std::endl;
        for(size_t t = 0; t < src.get_tlevs(); t++)
        {
            gpuErrchk(cudaMemcpy(res.get_tlev_ptr(t), src.get_tlev_ptr(t), src.get_geom().get_nelem_per_t() * sizeof(T), cudaMemcpyDeviceToHost));
        }

        return(res);
    }


    template <typename T>
    void update_host_vector(cuda_array_bc_nogp<T, allocator_host>& dst, cuda_array_bc_nogp<T, allocator_device>& src)
    {
        assert(dst.get_geom() == src.get_geom());
        for(size_t t = 0; t < src.get_tlevs(); t++)
        {
            gpuErrchk(cudaMemcpy(dst.get_tlev_ptr(t), src.get_tlev_ptr(t), src.get_geom().get_nelem_per_t() * sizeof(T), cudaMemcpyDeviceToHost));
        }
    }


    template <typename T>
    void print(cuda_array_bc_nogp<T, allocator_host>& vec, const size_t tlev, std::ostream& os)
    {
        address_t<T>* address = vec.get_address_ptr();
        for(size_t n = 0; n < vec.get_geom().get_nx(); n++)
        {
            for(size_t m = 0; m < vec.get_geom().get_my(); m++)
            {
                os << std::setw(cuda::io_w) << std::setprecision(cuda::io_p) << std::fixed << (*address)(vec.get_tlev_ptr(tlev), n, m) << "\t";
            }
            os << std::endl;
        }
    }
}


namespace detail 
{
    
    /// Initialize data_tlev_ptr:
    /// data_tlev_ptr[0] = data[0]
    /// data_tlev_ptr[1] = data[0] + nelem 
    /// ...
    template <typename T>
    inline void impl_set_data_tlev_ptr(T* data, T** data_tlev_ptr, const cuda::slab_layout_t sl, allocator_device<T>)
    {
        device :: kernel_set_tlev_ptr<<<1, 1>>>(data, data_tlev_ptr, sl);
    }


    template <typename T>
    inline void impl_set_data_tlev_ptr(T* data, T** data_tlev_ptr, const cuda::slab_layout_t sl, allocator_host<T>)
    {
        for(size_t t = 0; t < sl.get_tlevs(); t++)
        {
            data_tlev_ptr[t] = data + t * sl.get_nelem_per_t();
            std::cout << "data_tlev_ptr[" << t << "] at " << data_tlev_ptr[t] << std::endl;
        }
    }


    // Initialize ghost point interpolator
    // The next four functions are a bit messy:
    // The device implementation uses address_2ptr, an address_t<T>**, while the host implementation uses
    // an address_t<T>*.
    template <typename T>
    inline address_t<T>* impl_init_address(address_t<T>** &address_2ptr, address_t<T>* &address_ptr, const cuda::slab_layout_t geom, const cuda::bvals_t<T> bvals, allocator_device<T>)
    {
        gpuErrchk(cudaMalloc(&address_2ptr, sizeof(address_t<T>**)));
        device :: kernel_init_address<<<1, 1>>>(address_2ptr, geom, bvals);
        //std::cout << "impl_init_address(device): address_ptr = " << address_ptr << std::endl;
        //std::cout << "impl_init_address(device): address_2ptr = " << address_2ptr << std::endl;
        return(address_ptr);
    }


    template <typename T>
    inline void impl_delete_address(address_t<T>** &address_2ptr, address_t<T>* &address_ptr, allocator_device<T>)
    {
        //std::cout << "impl_delete_address(device), address_ptr -> " << address_ptr << std::endl;
        //std::cout << "impl_delete_address(device), address_2ptr -> " << address_2ptr << std::endl;
        device :: kernel_free_address<<<1, 1>>>(address_2ptr);
    }

    
    template <typename T>
    inline void impl_init_address(address_t<T>** &address_2ptr, address_t<T>* &address_ptr, const cuda::slab_layout_t geom, const cuda::bvals_t<T> bvals, allocator_host<T>)
    {
        address_ptr = new address_t<T>(geom, bvals);
        //std::cerr << "impl_init_address(host): address_ptr -> " << address_ptr << std::endl;
        //std::cerr << "impl_init_address(host): address_2ptr -> " << address_2ptr << std::endl;
    }

    template <typename T>
    inline void impl_delete_address(address_t<T>** &address_2ptr, address_t<T>* &address_ptr, allocator_host<T>)
    {
        //std::cerr << "impl_delete_address(host): address_ptr -> " << address_ptr << std::endl;
        //std::cerr << "impl_delete_address(host): address_2ptr -> " << address_2ptr << std::endl;
        delete address_ptr;        
    }


    // Get data_tlev_ptr for a given time level   
    // Returns a device-pointer 
    template <typename T>
    inline T* impl_get_data_tlev_ptr(T** data_tlev_ptr, const size_t tlev, const size_t tlevs, allocator_device<T>)
    {
        T** data_tlev_ptr_hostcopy = new T*[tlevs];
        gpuErrchk(cudaMemcpy(data_tlev_ptr_hostcopy, data_tlev_ptr, tlevs * sizeof(T*), cudaMemcpyDeviceToHost));
        return data_tlev_ptr_hostcopy[tlev];
    }

    template <typename T>
    inline T* impl_get_data_tlev_ptr(T** data_tlev_ptr, const size_t tlev, const size_t tlevs, allocator_host<T>)
    {
        return(data_tlev_ptr[tlev]);
    }


    template <typename T>
    inline void impl_initialize(T* data_ptr, const cuda::slab_layout_t geom, const dim3 grid, const dim3 block, allocator_device<T>)
    {
        device :: kernel_apply<<<grid, block>>>(data_ptr, 
                                      [=] __device__ (T value, size_t n, size_t m, cuda::slab_layout_t geom) -> T {return(T(0.0));},
                                      geom);
    }


    template <typename T>
    void impl_initialize(T* data_ptr, const cuda::slab_layout_t geom, const dim3 grid, const dim3 block, allocator_host<T>)
    {
        host :: host_apply(data_ptr, [=] (T value, size_t n, size_t m, cuda::slab_layout_t geom) -> T {return(T(0.0));}, geom);
    }


    template <typename T, typename F>
    inline void impl_apply(T* data_ptr, F myfunc, const cuda::slab_layout_t geom, const dim3 grid, const dim3 block, allocator_device<T>)
    {
        device :: kernel_apply<<<grid, block>>>(data_ptr, myfunc, geom);   
    }


    template <typename T, typename F>
    inline void impl_apply(T* data_ptr, F myfunc, const cuda::slab_layout_t geom, const dim3 grid, const dim3 block, allocator_host<T>)
    {
        host :: host_apply(data_ptr, myfunc, geom);
    }


    template <typename T>
    inline void impl_normalize_1d(T* data_ptr, const cuda::slab_layout_t& geom, const dim3 grid, const dim3 block, allocator_host<T>)
    {
        host :: host_apply(data_ptr, [=] (T value, const size_t n, const size_t m, const cuda::slab_layout_t geom) -> T
                                     { return(value / geom.get_my()); },
                           geom);
    }


    template <typename T>
    inline void impl_normalize_2d(T* data_ptr, const cuda::slab_layout_t& geom, const dim3 grid, const dim3 block, allocator_host<T>)
    {
        host :: host_apply(data_ptr, [=] (T value, const size_t n, const size_t m, const cuda::slab_layout_t geom) -> T
                                     { return(value / (geom.get_nx() * geom.get_my())); },
                           geom);
    }

    template <typename T>
    inline void impl_normalize_1d(T* data_ptr, const cuda::slab_layout_t& geom, const dim3 grid, const dim3 block, allocator_device<T>)
    {
        device :: kernel_apply<<<grid, block>>>(data_ptr, 
                                                [=] __device__ (T data, size_t n, size_t m, cuda::slab_layout_t geom) -> T {return(data / T(geom.get_my()));},
                                                geom);
    }


    template <typename T>
    inline void impl_normalize_2d(T* data_ptr, const cuda::slab_layout_t& geom, const dim3 grid, const dim3 block, allocator_device<T>)
    {
        device :: kernel_apply<<<grid, block>>>(data_ptr, 
                                                [=] __device__ (T in, size_t n, size_t m, cuda::slab_layout_t geom) -> T
                                                {return(in / T(geom.get_nx() * geom.get_my()));},
                                                geom);
    }


    template <typename T>
    inline void impl_advance(T** tlev_ptr, const size_t tlevs, allocator_host<T>)
    {
        T* tmp{tlev_ptr[tlevs - 1]};
        for(size_t t = tlevs - 1; t > 0; t--)
        {
            tlev_ptr[t] = tlev_ptr[t - 1];
        }
        tlev_ptr[0] = tmp;
    }

    template <typename T>
    inline void impl_advance(T** tlev_ptr, const size_t tlevs, allocator_device<T>)
    {
        device :: kernel_advance_tptr<<<1, 1>>>(tlev_ptr, tlevs);
    }


    template <typename T>
    void impl_op_plus_equal(T* lhs, T* rhs, const cuda::slab_layout_t geom, const dim3 grid, const dim3 block, allocator_host<T>)
    {
        host :: host_elementwise(lhs, rhs, [=] (T lhs, T rhs) -> T {return(lhs + rhs);}, geom);
    }
 

    template <typename T>
    void impl_op_plus_equal(T* lhs, T* rhs, const cuda::slab_layout_t geom, const dim3 grid, const dim3 block, allocator_device<T>)
    {
        device :: kernel_elementwise<<<grid, block>>>(lhs, rhs, [=] __device__ (T lhs, T rhs) -> T {return(lhs + rhs);}, geom);
    }


    template <typename T>
    void impl_op_minus_equal(T* lhs, T* rhs, const cuda::slab_layout_t geom, const dim3 grid, const dim3 block, allocator_host<T>)
    {
        host :: host_elementwise(lhs, rhs, [=] (T lhs, T rhs) -> T {return(lhs - rhs);}, geom);
    }
 

    template <typename T>
    void impl_op_minus_equal(T* lhs, T* rhs, const cuda::slab_layout_t geom, const dim3 grid, const dim3 block, allocator_device<T>)
    {
        host :: host_elementwise(lhs, rhs, [=] (T lhs, T rhs) -> T {return(lhs - rhs);}, geom);
    }
 

    template <typename T>
    void impl_op_mult_equal(T* lhs, T* rhs, const cuda::slab_layout_t geom, const dim3 grid, const dim3 block, allocator_host<T>)
    {
        host :: host_elementwise(lhs, rhs, [=] (T lhs, T rhs) -> T {return(lhs * rhs);}, geom);
    }
 

    template <typename T>
    void impl_op_mult_equal(T* lhs, T* rhs, const cuda::slab_layout_t geom, const dim3 grid, const dim3 block, allocator_device<T>)
    {
        device :: kernel_elementwise(lhs, rhs, [=] (T lhs, T rhs) -> T {return(lhs * rhs);}, geom);
    }
 

    template <typename T>
    T impl_reduce(T* data_ptr, const cuda::slab_layout_t geom, const dim3 grid, const dim3 block, allocator_host<T>)
    {
        T tmp{0.0};
        for(size_t n = 0; n < geom.get_nx(); n++)
        {
            for(size_t m = 0; m < geom.get_my(); m++)
            {
                tmp += abs(data_ptr[n * (geom.get_my() + geom.get_pad_y()) + m] * data_ptr[n * (geom.get_my() + geom.get_pad_y()) + m]);
            }
        }
        return tmp;
    }


    template <typename T>
    T impl_reduce(T* data_ptr, const cuda::slab_layout_t geom, const dim3 grid, const dim3 block, allocator_device<T>)
    {
        // Configuration for reduction kernel
        const size_t shmem_size_row = geom.get_nx() * sizeof(T);
        const dim3 blocksize_row(static_cast<int>(geom.get_nx()), 1, 1);
        const dim3 gridsize_row(1, static_cast<int>(geom.get_my()), 1);

        T rval{0.0};

        // temporary value profile
        //T* h_tmp_profile(new T[Nx]);
        T* d_rval_ptr{nullptr};
        T* d_tmp_profile{nullptr};
        T* device_copy{nullptr};

        // Result from 1d->0d reduction on device
        gpuErrchk(cudaMalloc((void**) &d_rval_ptr, sizeof(T)));
        // Result from 2d->1d reduction on device
        gpuErrchk(cudaMalloc((void**) &d_tmp_profile, geom.get_nx() * sizeof(T)));
        // Copy data to non-strided memory layout
        gpuErrchk(cudaMalloc((void**) &device_copy, geom.get_nx() * geom.get_my() * sizeof(T)));

        // Geometry of the temporary array, no padding
        cuda::slab_layout_t tmp_geom{geom.get_xleft(), geom.get_deltax(), geom.get_ylo(), geom.get_deltay(),
                                     geom.get_nx(), 0, geom.get_my(), 0, geom.get_grid(), geom.get_tlevs()};

        // Create device copy column-wise, ignore padding
        for(size_t n = 0; n < geom.get_nx(); n++)
        {
            gpuErrchk(cudaMemcpy((void*) (device_copy + n * geom.get_my()),
                                 (void*) (data_ptr + n * (geom.get_my() + geom.get_pad_y())), 
                                 geom.get_my() * sizeof(T), 
                                 cudaMemcpyDeviceToDevice));
        }

        // Take the square of the absolute value
        device :: kernel_apply<<<grid, block>>>(device_copy,
                                                [=] __device__ (T in, size_t n, size_t m, cuda::slab_layout_t geom ) -> T 
                                                {return(abs(in) * abs(in));}, 
                                                tmp_geom);
        //T* tmp_arr(new T[Nx * My]);
        //gpuErrchk(cudaMemcpy(tmp_arr, device_copy.get(), get_nx() * get_my() * sizeof(T), cudaMemcpyDeviceToHost));
        //for(size_t n = 0; n < get_nx(); n++)
        //{
        //    for(size_t m = 0; m < get_my(); m++)
        //    {
        //        cout << tmp_arr[n * get_my() + m] << "\t";
        //    }
        //    cout << endl;
        //}
        //delete [] tmp_arr;
        // Perform 2d -> 1d reduction
        device :: kernel_reduce<<<gridsize_row, blocksize_row, shmem_size_row>>>(device_copy, d_tmp_profile, 
                                                                       [=] __device__ (T op1, T op2) -> T {return(op1 + op2);},
                                                                       1, geom.get_nx(), geom.get_nx(), geom.get_my());
        //gpuErrchk(cudaMemcpy(h_tmp_profile, d_tmp_profile.get(), get_nx() * sizeof(T), cudaMemcpyDeviceToHost));
        //for(size_t n = 0; n < Nx; n++)
        //{
        //    cout << n << ": " << h_tmp_profile[n] << endl;
        //}
        // Perform 1d -> 0d reduction
        device :: kernel_reduce<<<1, geom.get_nx(), shmem_size_row>>>(d_tmp_profile, d_rval_ptr, 
                                                       [=] __device__ (T op1, T op2) -> T {return(op1 + op2);},
                                                       1, geom.get_nx(), geom.get_nx(), 1);
        gpuErrchk(cudaMemcpy(&rval, (void*) d_rval_ptr, sizeof(T), cudaMemcpyDeviceToHost));

        cudaFree(device_copy);
        cudaFree(d_tmp_profile);
        cudaFree(d_rval_ptr);

        return(sqrt(rval / static_cast<T>(geom.get_nx() * geom.get_my())));
    }
}



template <typename T, template <typename> class allocator>
class cuda_array_bc_nogp{
public:

    // T* pointers
    using allocator_type = typename my_allocator_traits<T, allocator> :: allocator_type;
    using deleter_type = typename my_allocator_traits<T, allocator> :: deleter_type;
    using ptr_type = std::unique_ptr<T, deleter_type>;

    // T** pointers
    using p_allocator_type = typename my_allocator_traits<T*, allocator> :: allocator_type;
    using p_deleter_type = typename my_allocator_traits<T*, allocator> :: deleter_type;
    using pptr_type = std::unique_ptr<T*, p_deleter_type>;

	cuda_array_bc_nogp(const cuda::slab_layout_t, const cuda::bvals_t<T>);
    cuda_array_bc_nogp(const cuda_array_bc_nogp<T, allocator>* rhs);
    cuda_array_bc_nogp(const cuda_array_bc_nogp<T, allocator>& rhs);

	~cuda_array_bc_nogp();

    /// Evaluate the function F on the grid
    template <typename F> inline void apply(F, const size_t);

    /// Initialize all elements to zero. Making this private results in compile error:
    /// /home/rku000/source/2dads/include/cuda_array_bc_nogp.h(414): error: An explicit __device__ lambda 
    //cannot be defined in a member function that has private or protected access within its class ("cuda_array_bc_nogp")
    inline void initialize();
    inline void initialize(const size_t);

    cuda_array_bc_nogp<T, allocator>& operator=(const cuda_array_bc_nogp<T, allocator>&);

    cuda_array_bc_nogp<T, allocator>& operator+=(const cuda_array_bc_nogp<T, allocator>&);
    cuda_array_bc_nogp<T, allocator>& operator-=(const cuda_array_bc_nogp<T, allocator>&);
    cuda_array_bc_nogp<T, allocator>& operator*=(const cuda_array_bc_nogp<T, allocator>&);
    cuda_array_bc_nogp<T, allocator>& operator/=(const cuda_array_bc_nogp<T, allocator>&);

    cuda_array_bc_nogp<T, allocator> operator+(const cuda_array_bc_nogp<T, allocator>&) const;  
    cuda_array_bc_nogp<T, allocator> operator-(const cuda_array_bc_nogp<T, allocator>&) const;  
    cuda_array_bc_nogp<T, allocator> operator*(const cuda_array_bc_nogp<T, allocator>&) const;  
    cuda_array_bc_nogp<T, allocator> operator/(const cuda_array_bc_nogp<T, allocator>&) const;  

	inline void normalize(const size_t);

	///@brief Copy data from t_src to t_dst
	inline void copy(const size_t t_dst, const size_t t_src);
	///@brief Copy data from src, t_src to t_dst
    inline void copy(size_t t_dst, const cuda_array_bc_nogp<T, allocator>& src, size_t t_src);
	///@brief Move data from t_src to t_dst, zero out t_src
	inline void move(const size_t t_dst, const size_t t_src);

	// Advance time levels
	inline void advance();

    T L2(const size_t);

	// Access to private members
	inline size_t get_nx() const {return(get_geom().get_nx());};
	inline size_t get_my() const {return(get_geom().get_my());};
	inline size_t get_tlevs() const {return(get_geom().get_tlevs());};
    inline cuda::slab_layout_t get_geom() const {return(geom);};
    inline cuda::bvals_t<T> get_bvals() const {return(boundaries);};
    // We are working with 2 pointer levels, since we instantiate an address object 
    // in a cuda kernel in the constructor. That way, we can just pass this
    // pointer to all cuda kernels that need an address object.
    // Call a separate kernel in the destructor to delete it.
    // Unfortunately, this means we have to use 2 pointer levels also in cpu functions.
    inline address_t<T>** get_address_2ptr() const {return(address_2ptr);};
    inline address_t<T>* get_address_ptr() const {return(address_ptr);};

	inline dim3 get_grid() const {return grid;};
	inline dim3 get_block() const {return block;};

	// smart pointer to device data, entire array
	inline T* get_data() const {return data.get();};
	// Pointer to array of pointers, corresponding to time levels
	inline T** get_tlev_ptr() const {return data_tlev_ptr.get();};
	// Pointer to device data at time level t
    inline T* get_tlev_ptr(const size_t) const;

	// Check bounds
	inline void check_bounds(size_t t, size_t n, size_t m) const {array_bounds(t, n, m);};
	inline void check_bounds(size_t n, size_t m) const {array_bounds(n, m);};

    // Set true if transformed
    inline bool is_transformed() const {return(transformed);};
    inline bool set_transformed(bool val) 
    {
        transformed = val; 
        return(transformed);
    };

private:
    const bounds array_bounds;
	const cuda::bvals_t<T> boundaries;
    const cuda::slab_layout_t geom;
    bool transformed;

    allocator_type my_alloc;
    p_allocator_type my_palloc;

    // The cuda implementation uses this one. address_t is instantiated once on the device
    address_t<T>** address_2ptr;
    // The host implementation uses this one.
    address_t<T>* address_ptr;

	// block and grid for access without ghost points, use these normally
	const dim3 block;
	const dim3 grid;

    // Size of shared memory bank
    const size_t shmem_size_col;   
	// Array data is on device
	// Pointer to device data
	ptr_type data;
	// Pointer to each time stage. Pointer to array of pointers on device
	pptr_type data_tlev_ptr;
};


template <typename T, template<typename> class allocator>
cuda_array_bc_nogp<T, allocator> :: cuda_array_bc_nogp (const cuda::slab_layout_t _geom, const cuda::bvals_t<T> bvals) : 
        array_bounds(get_tlevs(), get_nx(), get_my()),
        boundaries(bvals), 
        geom(_geom), 
        transformed{false},
        address_2ptr{nullptr},
        address_ptr{nullptr},
        block(dim3(cuda::blockdim_row, cuda::blockdim_col)),
		grid(dim3(((get_my() + geom.get_pad_y()) + cuda::blockdim_row - 1) / cuda::blockdim_row, 
                  ((get_nx() + geom.get_pad_x()) + cuda::blockdim_col - 1) / cuda::blockdim_col)),
        shmem_size_col(get_nx() * sizeof(T)),
        data(my_alloc.allocate(get_tlevs() * get_geom().get_nelem_per_t())),
		data_tlev_ptr(my_palloc.allocate(get_tlevs()))
{
    //cout << "cuda_array_bc<allocator> ::cuda_array_bc<allocator>\t";
    //cout << "Nx = " << Nx << ", pad_x = " << geom.pad_x << ", My = " << My << ", pad_y = " << geom.pad_y << endl;
    //cout << "block = ( " << block.x << ", " << block.y << ")" << endl;
    //cout << "grid = ( " << grid.x << ", " << grid.y << ")" << endl;
    //cout << geom << endl;

    // Set the pointer in array_tlev_ptr to data[0], data[0] + get_nelem_per_t(), data[0] + 2 * get_nelem_per_t() ...
    detail :: impl_set_data_tlev_ptr(get_data(), get_tlev_ptr(), get_geom(), allocator_type{});
   
    // Initialize the address object
    detail :: impl_init_address(address_2ptr, address_ptr, get_geom(), get_bvals(), allocator_type{});
    
    std::cout << "cuda_array_bc_nogp:: address_ptr -> " << address_ptr << std::endl;
    std::cout << "cuda_array_bc_nogp:: address_2ptr -> " << address_2ptr << std::endl;

    initialize();
}


template <typename T, template <typename> class allocator>
cuda_array_bc_nogp<T, allocator> :: cuda_array_bc_nogp(const cuda_array_bc_nogp<T, allocator>* rhs) :
    cuda_array_bc_nogp(rhs -> get_geom(), rhs -> get_bvals()) 
{
    my_palloc.copy(get_tlev_ptr(), rhs -> get_tlev_ptr(), get_tlevs(), rhs -> get_tlev_ptr());
};


template <typename T, template <typename> class allocator>
cuda_array_bc_nogp<T, allocator> :: cuda_array_bc_nogp(const cuda_array_bc_nogp<T, allocator>& rhs) :
    cuda_array_bc_nogp(rhs.get_geom(), rhs.get_bvals()) 
{
    my_alloc.copy(rhs.get_data(), rhs.get_data() + get_tlevs() * get_geom().get_nelem_per_t(), get_data());
    my_palloc.copy(rhs.get_tlev_ptr(), rhs.get_tlev_ptr() + get_tlevs(), get_tlev_ptr());
};


template <typename T, template <typename> class allocator>
cuda_array_bc_nogp<T, allocator> :: ~cuda_array_bc_nogp()
{
    //std::cout << "~cuda_array: address_ptr = " << address_ptr << std::endl;
    //std::cout << "~cuda_array: address_2ptr = " << address_2ptr << std::endl;
    detail :: impl_delete_address(address_2ptr, address_ptr, allocator_type{});
}


template <typename T, template <typename> class allocator>
inline T* cuda_array_bc_nogp<T, allocator> :: get_tlev_ptr(const size_t tlev) const
{
    return(detail :: impl_get_data_tlev_ptr(get_tlev_ptr(), tlev, get_geom().get_tlevs(), allocator_type{}));
}


template <typename T, template <typename> class allocator>
template <typename F>
inline void cuda_array_bc_nogp<T, allocator> :: apply(F myfunc, const size_t tlev)
{
    check_bounds(tlev, 0, 0);
    detail :: impl_apply(get_tlev_ptr(tlev), myfunc, get_geom(), get_grid(), get_block(), allocator_type{});
}

template <typename T, template <typename> class allocator>
inline void cuda_array_bc_nogp<T, allocator> :: initialize(const size_t tlev)
{
    detail :: impl_initialize(get_tlev_ptr(tlev), get_geom(), get_grid(), get_block(), allocator_type{});
}

template <typename T, template <typename> class allocator>
inline void cuda_array_bc_nogp<T, allocator> :: initialize()
{
    for(size_t t = 0; t < get_geom().get_tlevs(); t++)
    {
        initialize(t);
    }
}


template <typename T, template <typename> class allocator>
cuda_array_bc_nogp<T, allocator>& cuda_array_bc_nogp<T, allocator> :: operator= (const cuda_array_bc_nogp<T, allocator>& rhs)
{
    // Check for self-assignment
    if(this == &rhs)
        return(*this);

    check_bounds(rhs.get_tlevs(), rhs.get_nx(), rhs.get_my());
    for(size_t t = 0; t < get_tlevs(); t++)
    {
        my_alloc.copy(get_tlev_ptr(t), get_tlev_ptr(t) + get_geom().get_nelem_per_t(), rhs.get_tlev_ptr(t));
    }
    return (*this);
}


template <typename T, template <typename> class allocator>
cuda_array_bc_nogp<T, allocator>& cuda_array_bc_nogp<T, allocator> :: operator+=(const cuda_array_bc_nogp<T, allocator>& rhs) 
{
    check_bounds(rhs.get_tlevs(), rhs.get_nx(), rhs.get_my());
    detail :: impl_op_plus_equal(get_tlev_ptr(0), rhs.get_tlev_ptr(0), get_geom(), get_block(), get_grid(), allocator_type{});
    return *this;
}


template <typename T, template <typename> class allocator>
cuda_array_bc_nogp<T, allocator>& cuda_array_bc_nogp<T, allocator> :: operator-=(const cuda_array_bc_nogp<T, allocator>& rhs) 
{
    check_bounds(rhs.get_tlevs(), rhs.get_nx(), rhs.get_my());
    std::cerr << "Not implemented yet" << std::endl;
    //kernel_op1_arr<<<get_grid(), get_block()>>>(get_array_d(0), rhs.get_array_d(0),
    //                                            [=] __device__ (T lhs, T rhs) -> T  
    //                                            {
    //                                                return(lhs - rhs);
    //                                            }, get_geom());
    return *this;
}


template <typename T, template <typename> class allocator>
cuda_array_bc_nogp<T, allocator>& cuda_array_bc_nogp<T, allocator> :: operator*=(const cuda_array_bc_nogp<T, allocator>& rhs) 
{
    check_bounds(rhs.get_tlevs(), rhs.get_nx(), rhs.get_my());
    std::cerr << "Not implemented yet" << std::endl;
    //kernel_op1_arr<<<get_grid(), get_block()>>>(get_array_d(0), rhs.get_array_d(0),
    //                                            [=] __device__ (T lhs, T rhs) -> T
    //                                            {
    //                                                return(lhs * rhs);
    //                                            }, get_geom());
    return *this;
}


template <typename T, template <typename> class allocator>
cuda_array_bc_nogp<T, allocator>& cuda_array_bc_nogp<T, allocator> :: operator/=(const cuda_array_bc_nogp<T, allocator>& rhs) 
{
    check_bounds(rhs.get_tlevs(), rhs.get_nx(), rhs.get_my());
    std::cerr << "Not implemented yet" << std::endl;
    //kernel_op1_arr<<<get_grid(), get_block()>>>(get_array_d(0), rhs.get_array_d(0),
    //                                            [=] __device__ (T lhs, T rhs) -> T
    //                                            {
    //                                                return(lhs / rhs);
    //                                            }, get_geom());
    return *this;
}


//template <typename T, template <typename> class allocator>
//cuda_array_bc_nogp<T, allocator> cuda_array_bc_nogp<T, allocator> :: operator+(const cuda_array_bc_nogp<T, allocator>& rhs) const
//{
//    cuda_array_bc_nogp<T, allocator> result(this);
//    result += rhs;
//    return(result);
//}
//
//
//template <typename T, template <typename> class allocator>
//cuda_array_bc_nogp<T, allocator> cuda_array_bc_nogp<T, allocator> :: operator-(const cuda_array_bc_nogp<T, allocator>& rhs) const
//{
//    cuda_array_bc_nogp<T, allocator> result(this);
//    result -= rhs;
//    return(result);
//}
//
//
//template <typename T, template <typename> class allocator>
//cuda_array_bc_nogp<T, allocator> cuda_array_bc_nogp<T, allocator> :: operator*(const cuda_array_bc_nogp<T, allocator>& rhs) const
//{
//    cuda_array_bc_nogp<T, allocator> result(this);
//    result *= rhs;
//    return(result);
//}
//
//
//template <typename T, template <typename> class allocator>
//cuda_array_bc_nogp<T, allocator> cuda_array_bc_nogp<T, allocator> :: operator/(const cuda_array_bc_nogp<T, allocator>& rhs) const
//{
//    cuda_array_bc_nogp<T, allocator> result(this);
//    result /= rhs;
//    return(result);
//}


template <typename T, template <typename> class allocator>
inline void cuda_array_bc_nogp<T, allocator> :: normalize(const size_t tlev)
{
    // If we made a 1d DFT normalize by My. Otherwise nomalize by Nx * My
    switch (boundaries.get_bc_left())
    {
        case cuda::bc_t::bc_dirichlet:
            // fall through
        case cuda::bc_t::bc_neumann:
            detail :: impl_normalize_1d(get_tlev_ptr(tlev), get_geom(), get_grid(), get_block(), allocator_type{});
            break;

        case cuda::bc_t::bc_periodic:
            detail :: impl_normalize_2d(get_tlev_ptr(tlev), get_geom(), get_grid(), get_block(), allocator_type{});
            break;
    }
}


template <typename T, template <typename> class allocator>
inline void cuda_array_bc_nogp<T, allocator> :: copy(const size_t t_dst, const size_t t_src)
{
    check_bounds(t_dst, 0, 0);
    check_bounds(t_src, 0, 0);
    my_alloc.copy(get_tlev_ptr(t_src), get_tlev_ptr(t_src) + get_geom().get_nelem_per_t(), get_tlev_ptr(t_dst));    

}


template <typename T, template <typename> class allocator>
inline void cuda_array_bc_nogp<T, allocator> :: copy(const size_t t_dst, const cuda_array_bc_nogp<T, allocator>& src, const size_t t_src)
{
    check_bounds(t_dst, 0, 0);
    src.check_bounds(t_src, 0, 0);
    assert(get_geom() == src.get_geom());
    //cout << "copying array data t_dst = " << t_dst << ", t_src = " << t_src << endl;
    my_alloc.copy(src.get_tlev_ptr(t_src), src.get_tlev_ptr() + src.get_geom().get_nelem_per_t(), get_tlev_ptr(t_dst));
}


template <typename T, template <typename> class allocator>
inline void cuda_array_bc_nogp<T, allocator> :: move(const size_t t_dst, const size_t t_src)
{
    my_alloc.copy(get_tlev_ptr(t_src), get_tlev_ptr(t_src) + get_geom().get_nelem_per_t(), get_tlev_ptr(t_dst));
    detail :: impl_initialize(get_tlev_ptr(t_src), get_geom(), get_grid(), get_block(), allocator_type{});
}


template <typename T, template <typename> class allocator>
inline void cuda_array_bc_nogp<T, allocator> :: advance()
{
    // Advance tlev_ptr and zero out first time step
    detail :: impl_advance(get_tlev_ptr(), get_tlevs(), allocator_type{});
    initialize(0);
}


// Computes the L2 norm
// Problem: when T is CuCmplx<T>, we return a CuCmplx<T>, when T is double, we return double.
// Anyway, the Lambda calls abs(in), which does not give the correct result when T is CuCmplx<T>.
//
// For now, just use this routine if we have doubles!
template <typename T, template <typename> class allocator>
T cuda_array_bc_nogp<T, allocator> :: L2(const size_t tlev)
{
    return(detail :: impl_reduce(get_tlev_ptr(tlev), get_geom(), get_grid(), get_block(), allocator_type{}));
}


#endif /* cuda_array_bc_H_ */
