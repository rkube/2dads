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
#include <fstream>
#include <map>
#include <functional>
#include <sstream>

//#include <memory>
#include "2dads_types.h"
#include "bounds.h"
#include "address.h"
#include "error.h"
#include "allocators.h"


#ifdef __CUDACC__
#include "cuda_types.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#endif // __CUDACC__


#ifndef __CUDACC__
struct dim3{
    int x;
    int y;
    int z;
};

#define LAMBDACALLER

#endif //ifndef __CUDACC

#ifdef __CUDACC__

#define LAMBDACALLER __device__

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
#endif //__CUDACC__


namespace device
{
#ifdef __CUDACC__
    /// Return true if row and column are within geom(excluding padded rows/cols)
    /// Return false if row or column is outside the geometry
    __device__ inline bool good_idx(size_t row, size_t col, const twodads::slab_layout_t geom)
    {
        return((row < geom.get_nx()) && (col < geom.get_my()));
    }


    template <typename T>
    __global__
    void kernel_set_tlev_ptr(T* data, T** tlev_ptr, const size_t tlevs, const twodads::slab_layout_t geom)
    {
        for(size_t t = 0; t < tlevs; t++)
        {
            tlev_ptr[t] = &data[t * geom.get_nelem_per_t()];
        }
    }


    /// Apply the lambda op_func (with type given by template parameter O)
    /// op_func(T, size_t, size_t, slab_layout_t)
    template <typename T, typename O>
    __global__
    void kernel_apply(T* array_d_t, O device_func, const twodads::slab_layout_t geom) 
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
    void kernel_elementwise(T* lhs, T* rhs, O device_func, const twodads::slab_layout_t geom)
    {
        const size_t col{cuda :: thread_idx :: get_col()};
        const size_t row{cuda :: thread_idx :: get_row()};
        const size_t index{row * (geom.get_my() + geom.get_pad_y()) + col};

        if(good_idx(row, col, geom))
        {
            lhs[index] = device_func(lhs[index], rhs[index]);
        }
    }


    // For accessing elements in GPU kernels and interpolating ghost points
    template <typename T>
    __global__
    void kernel_init_address(address_t<T>** my_address, 
            const twodads::slab_layout_t geom, 
            const twodads::bvals_t<T> bvals)
    {
        *my_address = new address_t<T>(geom, bvals);
    }

    template <typename T>
    __global__
    void kernel_free_address(address_t<T>** my_address)
    {
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
    void host_apply(T* data_ptr, O host_func, const twodads::slab_layout_t& geom)
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
    void host_elementwise(T* lhs, T* rhs, O host_func, const twodads::slab_layout_t& geom)
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




namespace detail 
{
    
    /// Initialize data_tlev_ptr:
    /// data_tlev_ptr[0] = data[0]
    /// data_tlev_ptr[1] = data[0] + nelem 
    /// ...
#ifdef __CUDACC__
    template <typename T>
    inline void impl_set_data_tlev_ptr(T* data, T** data_tlev_ptr, const size_t tlevs, const twodads::slab_layout_t sl, allocator_device<T>)
    {
        device :: kernel_set_tlev_ptr<<<1, 1>>>(data, data_tlev_ptr, tlevs, sl);
        gpuErrchk(cudaPeekAtLastError());
    }

    // Initialize ghost point interpolator
    // The next four functions are a bit messy:
    // The device implementation uses address_2ptr, an address_t<T>**, while the host implementation uses
    // an address_t<T>*.
    template <typename T>
    inline address_t<T>* impl_init_address(address_t<T>** &address_2ptr, address_t<T>* &address_ptr, const twodads::slab_layout_t geom, const twodads::bvals_t<T> bvals, allocator_device<T>)
    {
        gpuErrchk(cudaMalloc(&address_2ptr, sizeof(address_t<T>**)));
        device :: kernel_init_address<<<1, 1>>>(address_2ptr, geom, bvals);
        gpuErrchk(cudaPeekAtLastError());
        return(address_ptr);
    }


    template <typename T>
    inline void impl_delete_address(address_t<T>** &address_2ptr, address_t<T>* &address_ptr, allocator_device<T>)
    {
        device :: kernel_free_address<<<1, 1>>>(address_2ptr);
        gpuErrchk(cudaPeekAtLastError());
    }

     // Get data_tlev_ptr for a given time level   
    // Returns a device-pointer 
    template <typename T>
    inline T* impl_get_data_tlev_ptr(T** data_tlev_ptr, const size_t tlev, const size_t tlevs, allocator_device<T>)
    {
        T** data_tlev_ptr_hostcopy = new T*[tlevs];
        gpuErrchk(cudaMemcpy(data_tlev_ptr_hostcopy, data_tlev_ptr, tlevs * sizeof(T*), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaPeekAtLastError());
        return data_tlev_ptr_hostcopy[tlev];
    }

    template <typename T>
    inline void impl_initialize(T* data_ptr, const twodads::slab_layout_t geom, const dim3 grid, const dim3 block, allocator_device<T>)
    {
        device :: kernel_apply<<<grid, block>>>(data_ptr, 
                                      [] __device__ (T value, size_t n, size_t m, twodads::slab_layout_t geom) -> T {return(T(0.0));},
                                      geom);
        gpuErrchk(cudaPeekAtLastError());
    }

    template <typename T, typename F>
    inline void impl_apply(T* data_ptr, F myfunc, const twodads::slab_layout_t geom, const dim3 grid, const dim3 block, allocator_device<T>)
    {   
        device :: kernel_apply<<<grid, block>>>(data_ptr, myfunc, geom);   
        gpuErrchk(cudaPeekAtLastError());
    }

    template <typename T, typename F>
    inline void impl_elementwise(T* x, T* rhs, F myfunc, const twodads::slab_layout_t& geom, const dim3& grid, const dim3& block, allocator_device<T>)
    {
        device :: kernel_elementwise<<<grid, block>>>(x, rhs, myfunc, geom);
        gpuErrchk(cudaPeekAtLastError());
    }


    template <typename T>
    inline void impl_advance(T** tlev_ptr, const size_t tlevs, allocator_device<T>)
    {
        device :: kernel_advance_tptr<<<1, 1>>>(tlev_ptr, tlevs);
        gpuErrchk(cudaPeekAtLastError());
    }


    template <typename T>
    void impl_op_plus_equal(T* lhs, T* rhs, const twodads::slab_layout_t geom, const dim3 grid, const dim3 block, allocator_device<T>)
    {
        device :: kernel_elementwise<<<grid, block>>>(lhs, rhs, [] __device__ (T a, T b) -> T {return(a + b);}, geom);
        gpuErrchk(cudaPeekAtLastError());
    }

    template <typename T>
    void impl_op_plus_equal_scalar(T* lhs, T rhs, const twodads::slab_layout_t geom, const dim3 grid, const dim3 block, allocator_device<T>)
    {
        // Let lambda capture local variables by value [=], thus T rhs from argument list of 
        // this function is passed into the lambda
        device :: kernel_apply<<<grid, block>>>(lhs, [=] __device__ (T a, size_t row, size_t col, twodads::slab_layout_t geom) -> T {return(a + rhs);}, geom);
        gpuErrchk(cudaPeekAtLastError());    
    }

    template <typename T>
    void impl_op_minus_equal(T* lhs, T* rhs, const twodads::slab_layout_t geom, const dim3 grid, const dim3 block, allocator_device<T>)
    {
        device :: kernel_elementwise<<<grid, block>>>(lhs, rhs, [] __device__ (T a, T b) -> T {return(a - b);}, geom);
        gpuErrchk(cudaPeekAtLastError());
    }


    template <typename T>
    void impl_op_minus_equal_scalar(T* lhs, T rhs, const twodads::slab_layout_t geom, const dim3 grid, const dim3 block, allocator_device<T>)
    {
        // Lambda captures local rhs by value
        device :: kernel_apply<<<grid, block>>>(lhs, [=] __device__ (T a, size_t row, size_t col, twodads::slab_layout_t geom) -> T {return(a - rhs);}, geom);
        gpuErrchk(cudaPeekAtLastError());    
    }


    template <typename T>
    void impl_op_mult_equal(T* lhs, T* rhs, const twodads::slab_layout_t geom, const dim3 grid, const dim3 block, allocator_device<T>)
    {
        device :: kernel_elementwise(lhs, rhs, [] (T a, T b) -> T {return(a * b);}, geom);
        gpuErrchk(cudaPeekAtLastError());
    }


    template <typename T>
    void impl_op_mult_equal_scalar(T* lhs, T rhs, const twodads::slab_layout_t geom, const dim3 grid, const dim3 block, allocator_device<T>)
    {
        // Lambda captures local rhs by value
        device :: kernel_apply<<<grid, block>>>(lhs, [=] __device__ (T a, size_t row, size_t col, twodads::slab_layout_t geom) -> T {return(a * rhs);}, geom);
        gpuErrchk(cudaPeekAtLastError());    
    }
#endif //__CUDACC__

    template <typename T>
    inline void impl_set_data_tlev_ptr(T* data, T** data_tlev_ptr, const size_t tlevs, const twodads::slab_layout_t& sl, allocator_host<T>)
    {
        for(size_t t = 0; t < tlevs; t++)
        {
            data_tlev_ptr[t] = data + t * sl.get_nelem_per_t();
        }
    }

    
    template <typename T>
    inline void impl_init_address(address_t<T>** &address_2ptr, address_t<T>* &address_ptr, const twodads::slab_layout_t& geom, const twodads::bvals_t<T>& bvals, allocator_host<T>)
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
        if(address_ptr != nullptr)
            delete address_ptr;        
        address_ptr = nullptr;
    }


    template <typename T>
    inline T* impl_get_data_tlev_ptr(T** data_tlev_ptr, const size_t tidx, const size_t tlevs, allocator_host<T>)
    {
        return(data_tlev_ptr[tidx]);
    }


    template <typename T>
    void impl_initialize(T* data_ptr, const twodads::slab_layout_t& geom, const dim3& grid, const dim3& block, allocator_host<T>)
    {
        host :: host_apply(data_ptr, [] (T value, size_t n, size_t m, twodads::slab_layout_t geom) -> T {return(T(0.0));}, geom);
    }


    template <typename T, typename F>
    inline void impl_apply(T* data_ptr, F myfunc, const twodads::slab_layout_t& geom, const dim3& grid, const dim3& block, allocator_host<T>)
    {
        host :: host_apply(data_ptr, myfunc, geom);
    }


    template <typename T, typename F>
    inline void impl_elementwise(T* lhs, T* rhs, F myfunc, const twodads::slab_layout_t& geom, const dim3& grid, const dim3& block, allocator_host<T>)
    {
        host :: host_elementwise(lhs, rhs, myfunc, geom);
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
    void impl_op_plus_equal(T* lhs, T* rhs, const twodads::slab_layout_t& geom, const dim3& grid, const dim3& block, allocator_host<T>)
    {
        host :: host_elementwise(lhs, rhs, [] (T a, T b) -> T {return(a + b);}, geom);
    }
 

    template <typename T>
    void impl_op_plus_equal_scalar(T* lhs, T rhs, const twodads::slab_layout_t& geom, const dim3& grid, const dim3& block, allocator_host<T>)
    {
        // Lambda captures local rhs
        host :: host_apply(lhs, [=] (T value, size_t n, size_t m, twodads::slab_layout_t geom){return (value + rhs);}, geom);
    }


    template <typename T>
    void impl_op_minus_equal(T* lhs, T* rhs, const twodads::slab_layout_t& geom, const dim3& grid, const dim3& block, allocator_host<T>)
    {
        host :: host_elementwise(lhs, rhs, [] (T a, T b) -> T {return(a - b);}, geom);
    }
 

    template <typename T>
    void impl_op_minus_equal_scalar(T* lhs, T rhs, const twodads::slab_layout_t& geom, const dim3& grid, const dim3& block, allocator_host<T>)
    {
        // Lambda captures local rhs
        host :: host_apply(lhs, [=] (T value, size_t n, size_t m, twodads::slab_layout_t geom){return (value - rhs);}, geom);
    }

    template <typename T>
    void impl_op_mult_equal(T* lhs, T* rhs, const twodads::slab_layout_t& geom, const dim3& grid, const dim3& block, allocator_host<T>)
    {
        host :: host_elementwise(lhs, rhs, [] (T a, T b) -> T {return(a * b);}, geom);
    } 


    template <typename T>
    void impl_op_mult_equal_scalar(T* lhs, T rhs, const twodads::slab_layout_t& geom, const dim3& grid, const dim3& block, allocator_host<T>)
    {
        // Lambda captures local rhs
        host :: host_apply(lhs, [=](T value, size_t n, size_t m, twodads::slab_layout_t geom){return (value * rhs);}, geom);
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

	cuda_array_bc_nogp(const twodads::slab_layout_t, const twodads::bvals_t<T>, size_t _tlevs);
    cuda_array_bc_nogp(const cuda_array_bc_nogp<T, allocator>* rhs);
    cuda_array_bc_nogp(const cuda_array_bc_nogp<T, allocator>& rhs);

	~cuda_array_bc_nogp()
    {
        detail :: impl_delete_address(address_2ptr, address_ptr, allocator_type{});
    };

    /// Evaluate the function F on the grid
    template <typename F> inline void apply(F myfunc, const size_t tidx)
    {
        check_bounds(tidx + 1, 0, 0);
        detail :: impl_apply(get_tlev_ptr(tidx), myfunc, get_geom(), get_grid(), get_block(), allocator_type{});   
    }


    template<typename F> inline void elementwise(F myfunc, const cuda_array_bc_nogp<T, allocator>& rhs,
                                                 const size_t tidx_rhs, const size_t tidx_lhs)
    {
        check_bounds(tidx_rhs + 1, 0, 0);
        check_bounds(tidx_lhs + 1, 0, 0);
        assert(rhs.get_geom() == get_geom());
        detail :: impl_elementwise(myfunc, get_tlev_ptr(tidx_lhs), rhs.get_tlev_ptr(tidx_rhs), get_geom(), get_grid(), get_block(), allocator_type{});
    }

    template<typename F> inline void elementwise(F myfunc, const size_t tidx_lhs, const size_t tidx_rhs)
    {
        check_bounds(tidx_rhs + 1, 0, 0);
        check_bounds(tidx_lhs + 1, 0, 0);
        detail :: impl_elementwise(get_tlev_ptr(tidx_lhs), get_tlev_ptr(tidx_rhs), myfunc, get_geom(), get_grid(), get_block(), allocator_type{});   
    }
    /// Initialize all elements to zero. Making this private results in compile error:
    /// /home/rku000/source/2dads/include/cuda_array_bc_nogp.h(414): error: An explicit __device__ lambda 
    //cannot be defined in a member function that has private or protected access within its class ("cuda_array_bc_nogp")
    inline void initialize()
    {
        for(size_t t = 0; t < get_tlevs(); t++)
        {
            initialize(t);
        }
    }

    inline void initialize(const size_t tidx)
    {
        detail :: impl_initialize(get_tlev_ptr(tidx), get_geom(), get_grid(), get_block(), allocator_type{});
    }

    cuda_array_bc_nogp<T, allocator>& operator=(const cuda_array_bc_nogp<T, allocator>& rhs)
    {
        // Check for self-assignment
        if(this == &rhs)
            return(*this);

        check_bounds(rhs.get_tlevs(), rhs.get_nx(), rhs.get_my());
        for(size_t t = 0; t < get_tlevs(); t++)
        {
            my_alloc.copy(rhs.get_tlev_ptr(t), rhs.get_tlev_ptr(t) + rhs.get_geom().get_nelem_per_t(), get_tlev_ptr(t));
            set_transformed(t, rhs.is_transformed(t));
        }
        return (*this);
    }


    cuda_array_bc_nogp<T, allocator>& operator+=(const cuda_array_bc_nogp<T, allocator>& rhs)
    {
        check_bounds(rhs.get_tlevs(), rhs.get_nx(), rhs.get_my());
        detail :: impl_op_plus_equal(get_tlev_ptr(0), rhs.get_tlev_ptr(0), get_geom(), get_grid(), get_block(), allocator_type{});
        return *this;
    }

    cuda_array_bc_nogp<T, allocator>& operator+=(const T rhs)
    {
        detail :: impl_op_plus_equal_scalar(get_tlev_ptr(0), rhs, get_geom(), get_grid(), get_block(), allocator_type{});
        return *this;
    }


    cuda_array_bc_nogp<T, allocator>& operator-=(const cuda_array_bc_nogp<T, allocator>& rhs)
    {
        check_bounds(rhs.get_tlevs(), rhs.get_nx(), rhs.get_my());
        detail :: impl_op_minus_equal(get_tlev_ptr(0), rhs.get_tlev_ptr(0), get_geom(), get_grid(), get_block(), allocator_type{});
        return *this;
    }
    

    cuda_array_bc_nogp<T, allocator>& operator-=(const T rhs)
    {
        detail :: impl_op_plus_equal_scalar(get_tlev_ptr(0), rhs, get_geom(), get_grid(), get_block(), allocator_type{});
        return *this;
    }
    
    cuda_array_bc_nogp<T, allocator>& operator*=(const cuda_array_bc_nogp<T, allocator>& rhs)
    {
        check_bounds(rhs.get_tlevs(), rhs.get_nx(), rhs.get_my());
        detail :: impl_op_mult_equal(get_tlev_ptr(0), rhs.get_tlev_ptr(0), get_geom(), get_grid(), get_block(), allocator_type{});
        return *this;
    }
    

    cuda_array_bc_nogp<T, allocator>& operator*=(const T rhs)
    {
        detail :: impl_op_mult_equal_scalar(get_tlev_ptr(0), rhs, get_geom(), get_grid(), get_block(), allocator_type{});
        return *this;
    }


    cuda_array_bc_nogp<T, allocator> operator+(const cuda_array_bc_nogp<T, allocator>& rhs) const
    {
        cuda_array_bc_nogp<T, allocator> result(this);
        result += rhs;
        return(result);
    }

    cuda_array_bc_nogp<T, allocator> operator-(const cuda_array_bc_nogp<T, allocator>& rhs) const
    {
        cuda_array_bc_nogp<T, allocator> result(this);
        result -= rhs;
        return(result);
    }

    cuda_array_bc_nogp<T, allocator> operator*(const cuda_array_bc_nogp<T, allocator>& rhs) const
    {
        cuda_array_bc_nogp<T, allocator> result(this);
        result *= rhs;
        return(result);
    }  
      

	//inline void normalize(const size_t);

	///@brief Copy data from t_src to t_dst
	inline void copy(const size_t tidx_dst, const size_t tidx_src)
    {
        check_bounds(tidx_dst + 1, 0, 0);
        check_bounds(tidx_src + 1, 0, 0);
        my_alloc.copy(get_tlev_ptr(tidx_src), get_tlev_ptr(tidx_src) + get_geom().get_nelem_per_t(), get_tlev_ptr(tidx_dst));
    }

	///@brief Copy data from src, t_src to t_dst
    inline void copy(size_t tidx_dst, const cuda_array_bc_nogp<T, allocator>& src, size_t tidx_src)
    {
        check_bounds(tidx_dst + 1, 0, 0);
        src.check_bounds(tidx_src + 1, 0, 0);
        assert(get_geom() == src.get_geom());
        my_alloc.copy(src.get_tlev_ptr(tidx_src), src.get_tlev_ptr(tidx_src) + src.get_geom().get_nelem_per_t(), get_tlev_ptr(tidx_dst));
    }

	///@brief Move data from t_src to t_dst, zero out t_src
	inline void move(const size_t tidx_dst, const size_t tidx_src)
    {
        check_bounds(tidx_dst + 1, 0, 0);
        check_bounds(tidx_src + 1, 0, 0);
        my_alloc.copy(get_tlev_ptr(tidx_src), get_tlev_ptr(tidx_src) + get_geom().get_nelem_per_t(), get_tlev_ptr(tidx_dst));
        detail :: impl_initialize(get_tlev_ptr(tidx_src), get_geom(), get_grid(), get_block(), allocator_type{});
    }

	// Advance time levels
	inline void advance()
    {
        detail :: impl_advance(get_tlev_ptr(), get_tlevs(), allocator_type{});
        initialize(0);
    }

	// Access to private members
	inline size_t get_nx() const {return(get_geom().get_nx());};
	inline size_t get_my() const {return(get_geom().get_my());};
	inline size_t get_tlevs() const {return(tlevs);};
    inline twodads::slab_layout_t get_geom() const {return(geom);};
    inline twodads::bvals_t<T> get_bvals() const {return(boundaries);};
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
    inline T* get_tlev_ptr(const size_t tidx) const
    {
        check_bounds(tidx + 1, 0, 0);
        return(detail :: impl_get_data_tlev_ptr(get_tlev_ptr(), tidx, get_tlevs(), allocator_type{}));   
    };

	// Check bounds
	inline void check_bounds(size_t t, size_t n, size_t m) const {array_bounds(t, n, m);};
	inline void check_bounds(size_t n, size_t m) const {array_bounds(n, m);};

    // Set true if transformed
    inline bool is_transformed(const size_t tlev) const {return(transformed[tlev]);};
    inline bool set_transformed(const size_t tlev, const bool val) 
    {
        transformed[tlev] = val; 
        return(transformed[tlev]);
    };

private:
	const twodads::bvals_t<T> boundaries;
    const twodads::slab_layout_t geom;
    const size_t tlevs;
    const bounds array_bounds;
    std::vector<bool> transformed;

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
cuda_array_bc_nogp<T, allocator> :: cuda_array_bc_nogp (const twodads::slab_layout_t _geom, const twodads::bvals_t<T> _bvals, size_t _tlevs) : 
        boundaries(_bvals), 
        geom(_geom), 
        tlevs(_tlevs),
        array_bounds(get_tlevs(), get_nx(), get_my()),
        transformed{std::vector<bool>(get_tlevs(), 0)},
        address_2ptr{nullptr},
        address_ptr{nullptr},
#ifdef __CUDACC__
        block(dim3(cuda::blockdim_row, cuda::blockdim_col)),
		grid(dim3(((get_my() + geom.get_pad_y()) + cuda::blockdim_row - 1) / cuda::blockdim_row, 
                  ((get_nx() + geom.get_pad_x()) + cuda::blockdim_col - 1) / cuda::blockdim_col)),
#endif //__CUDACC__
#ifndef __CUDACC__
        block{0,0,0},
        grid{0,0,0},
#endif
        shmem_size_col(get_nx() * sizeof(T)),
        data(my_alloc.allocate(get_tlevs() * get_geom().get_nelem_per_t())),
		data_tlev_ptr(my_palloc.allocate(get_tlevs()))
{
    // Set the pointer in array_tlev_ptr to data[0], data[0] + get_nelem_per_t(), data[0] + 2 * get_nelem_per_t() ...
    detail :: impl_set_data_tlev_ptr(get_data(), get_tlev_ptr(), get_tlevs(), get_geom(), allocator_type{});
   
    // Initialize the address object
    detail :: impl_init_address(address_2ptr, address_ptr, get_geom(), get_bvals(), allocator_type{});
    initialize();
}


template <typename T, template <typename> class allocator>
cuda_array_bc_nogp<T, allocator> :: cuda_array_bc_nogp(const cuda_array_bc_nogp<T, allocator>* rhs) :
    cuda_array_bc_nogp(rhs -> get_geom(), rhs -> get_bvals(), rhs -> get_tlevs()) 
{
    my_alloc.copy(rhs -> get_data(), rhs -> get_data() + get_tlevs() * get_geom().get_nelem_per_t(), get_data());
    my_palloc.copy(rhs -> get_tlev_ptr(), rhs -> get_tlev_ptr() + get_tlevs(), get_tlev_ptr());
};


template <typename T, template <typename> class allocator>
cuda_array_bc_nogp<T, allocator> :: cuda_array_bc_nogp(const cuda_array_bc_nogp<T, allocator>& rhs) :
    cuda_array_bc_nogp(rhs.get_geom(), rhs.get_bvals(), rhs.get_tlevs()) 
{
    my_alloc.copy(rhs.get_data(), rhs.get_data() + get_tlevs() * get_geom().get_nelem_per_t(), get_data());
    my_palloc.copy(rhs.get_tlev_ptr(), rhs.get_tlev_ptr() + get_tlevs(), get_tlev_ptr());
};


#endif /* cuda_array_bc_H_ */