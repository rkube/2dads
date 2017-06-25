//
//   Array type with variable padding in x- and y-direction.
//   Knows about boundary conditions, routines operating on this datatype are to compute
//   them on the fly
//
//   Memory Layout
//  
//   rows: 0...My-1 ... My-1 + pad_y
//   cols: 0...Nx-1 ... Nx-1 + pad_x
//  
//       0                        My-1 ... My - 1 + pad_y
//   Nx - 1 |--------- ... ------|          |
//          |--------- ... ------|          |
//   ...
//    0     |--------- ... ------|          |
//  
//   idx = n * (My + pad_y) + m
//  
//   columns (m, y-direction) are consecutive in memory
//  
//  
//   Mapping of CUDA threads on the array:
//  
//   Columns: 0..My - 1 + pad_y -> col = blockIdx.x * blockDim.x + threadIdx.x
//   Rows:    0..Nx - 1 + pad_x -> row = blockIdx.y * blockDim.y + threadIdx.y
//  
//   dimBlock = (blocksize_row, blocksize_col)
//   dimGrid = (My + pad_y) / blocksize_row, (My + pad_y) / blocksize_col
//  
//   Ghost points are to be computed on the fly, not stored in memory
//   They can be access by the address object

#ifndef cuda_array_bc_H_
#define cuda_array_bc_H_


#include <iostream>
#include <iomanip>
#include <fstream>
#include <map>
#include <functional>
#include <sstream>

#include "2dads_types.h"
#include "bounds.h"
#include "address.h"
#include "error.h"
#include "allocators.h"


#if defined(DEVICE)
#warning cuda_array_bc_nogp: compiling for device
#endif // DEVICE

#if defined(HOST)
#warning cuda_array_bc_nogp: compiling for host
#endif //HOST

#include "cuda_types.h"
#include <cuda.h>
#include <cuda_runtime_api.h>

#define LAMBDACALLER __host__ __device__

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


namespace device
{
    // Return true if row and column are within geom(include padded columns if is_transformed is true)
    // Return false if row or column is outside the geometry
    __device__ inline bool good_idx(const size_t row, const size_t col, const twodads::slab_layout_t geom, const bool is_transformed)
    {
        return((row < geom.get_nx()) && (col < (is_transformed ? geom.get_my() + geom.get_pad_y() : geom.get_my())));
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


    // Apply the lambda op_func (with type given by template parameter O)
    // op_func(T, size_t, size_t, slab_layout_t)
    template <typename T, typename O>
    __global__
    void kernel_apply_single(T* array_d_t, O device_func, const twodads::slab_layout_t geom, const bool is_transformed) 
    {
        const size_t col{cuda :: thread_idx :: get_col()};
        const size_t row{cuda :: thread_idx :: get_row()};
        const size_t index{row * (geom.get_my() + geom.get_pad_y()) + col};

        if (good_idx(row, col, geom, is_transformed))
            array_d_t[index] = device_func(array_d_t[index], row, col, geom);
    }


    template<typename T, typename O, size_t ELEMS>
    __global__
    void kernel_apply_unroll(T* array_d_t, O device_func, const twodads::slab_layout_t geom, const bool is_transformed) 
    {
        const size_t row{cuda :: thread_idx :: get_row()};

        const size_t col_0{cuda :: thread_idx :: get_col() * ELEMS};
        const size_t index_0{row * (geom.get_my() + geom.get_pad_y()) + col_0};

        for(size_t n = 0; n < ELEMS; n++)
        {
            if (good_idx(row, col_0 + n, geom, is_transformed))
                array_d_t[index_0 + n] = device_func(array_d_t[index_0 + n], row, col_0 + n, geom);
        }
    }


    // Perform element-wise arithmetic operation lhs[idx] = op(lhs[idx], rhs[idx])
    template<typename T, typename O>
    __global__
    void kernel_elementwise(T* lhs, T* rhs, O device_func, const twodads::slab_layout_t geom, const bool is_transformed)
    {
        const size_t col{cuda :: thread_idx :: get_col()};
        const size_t row{cuda :: thread_idx :: get_row()};
        const size_t index{row * (geom.get_my() + geom.get_pad_y()) + col};

        if(good_idx(row, col, geom, is_transformed))
        {
            lhs[index] = device_func(lhs[index], rhs[index]);
        }
    }


    template<typename T, typename O, size_t ELEMS>
    __global__
    void kernel_elementwise_unroll(T* lhs, T* rhs, O device_func, const twodads::slab_layout_t geom, const bool is_transformed)
    {
        const size_t col_0{cuda :: thread_idx :: get_col() * ELEMS};
        const size_t row{cuda :: thread_idx :: get_row()};
        const size_t index_0{row * (geom.get_my() + geom.get_pad_y()) + col_0};

        for(size_t n = 0; n < ELEMS; n++)
        {
            if(good_idx(row, col_0 + n, geom, is_transformed))
            {
                lhs[index_0 + n] = device_func(lhs[index_0 + n], rhs[index_0 + n]);
            }
        }
    }
    

    // For accessing elements in GPU kernels and interpolating ghost points
    template <typename T>
    __global__
    void kernel_init_address(address_t<T>** my_address, 
            const twodads::slab_layout_t geom, 
            const twodads::bvals_t<T> bvals)
    {
        //*my_address = (address_t<T>*) malloc(sizeof(address_t<T>));
        //*my_address = new(*my_address) address_t<T>(geom, bvals);

        *my_address = new address_t<T>(geom, bvals);
    }

    template <typename T>
    __global__
    void kernel_free_address(address_t<T>** my_address)
    {
        // Call the destructor explicitly. clang doesn't like delete.
        //(*my_address) -> ~address_t<T>();
        free(*my_address);
        //delete *my_address;
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
}

template <typename T, template <typename> class allocator> class cuda_array_bc_nogp;

namespace detail 
{
    
    // Initialize data_tlev_ptr:
    // data_tlev_ptr[0] = data[0]
    // data_tlev_ptr[1] = data[0] + nelem 
    // ...

    template <typename T>
    inline void impl_set_data_tlev_ptr(T* data, T** data_tlev_ptr, const size_t tlevs, const twodads::slab_layout_t& geom, allocator_device<T>)
    {
        #if defined(DEVICE)
        device :: kernel_set_tlev_ptr<<<1, 1>>>(data, data_tlev_ptr, tlevs, geom);
        #endif
        gpuErrchk(cudaPeekAtLastError());
    }

    // Initialize ghost point interpolator
    // The next four functions are a bit messy:
    // The device implementation uses address_2ptr, an address_t<T>**, while the host implementation uses
    // an address_t<T>*.
    template <typename T>
    inline address_t<T>* impl_init_address(address_t<T>** &address_2ptr, address_t<T>* &address_ptr, const twodads::slab_layout_t& geom, const twodads::bvals_t<T>& bvals, allocator_device<T>)
    {
        gpuErrchk(cudaMalloc(&address_2ptr, sizeof(address_t<T>**)));
        #if defined(DEVICE)
        device :: kernel_init_address<<<1, 1>>>(address_2ptr, geom, bvals);
        #endif
        gpuErrchk(cudaPeekAtLastError());
        return(address_ptr);
    }


    template <typename T>
    inline void impl_delete_address(address_t<T>** &address_2ptr, address_t<T>* &address_ptr, allocator_device<T>)
    {
        #if defined(DEVICE)
        device :: kernel_free_address<<<1, 1>>>(address_2ptr);
        #endif
        gpuErrchk(cudaPeekAtLastError());
    }

    // Get data_tlev_ptr for a given time level   
    // Returns a device-pointer 
    template <typename T>
    inline T* impl_get_data_tlev_ptr(T** data_tlev_ptr, const size_t tidx, const size_t tlevs, allocator_device<T>)
    {
        T** data_tlev_ptr_hostcopy = new T*[tlevs];
        gpuErrchk(cudaMemcpy(data_tlev_ptr_hostcopy, data_tlev_ptr, tlevs * sizeof(T*), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaPeekAtLastError());
        return data_tlev_ptr_hostcopy[tidx];
    }


    template <typename T, typename F>
    inline void impl_apply(T* data_ptr, F myfunc, const twodads::slab_layout_t& geom, const bool transformed, const dim3& grid_unroll, const dim3& block, allocator_device<T>)
    {
        #if defined(DEVICE)
        device :: kernel_apply_unroll<T, F, cuda::elem_per_thread><<<grid_unroll, block>>>(data_ptr, myfunc, geom, transformed);   
        #endif
        gpuErrchk(cudaPeekAtLastError());
    }

    template <typename T, typename F>
    inline void impl_elementwise(T* x, T* rhs, F myfunc, const twodads::slab_layout_t& geom, const bool transformed, const dim3& grid_unroll, const dim3& block, allocator_device<T>)
    {
        #if defined (DEVICE)
        device :: kernel_elementwise_unroll<T, F, cuda::elem_per_thread><<<grid_unroll, block>>>(x, rhs, myfunc, geom, transformed);
        #endif
        gpuErrchk(cudaPeekAtLastError());
    }


    template <typename T>
    inline void impl_advance(T** tlev_ptr, const size_t tlevs, allocator_device<T>)
    {
        #if defined(DEVICE)
        device :: kernel_advance_tptr<<<1, 1>>>(tlev_ptr, tlevs);
        #endif
        gpuErrchk(cudaPeekAtLastError());
    }


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
    }


    template <typename T>
    inline void impl_delete_address(address_t<T>** &address_2ptr, address_t<T>* &address_ptr, allocator_host<T>)
    {
        if(address_ptr != nullptr)
            delete address_ptr;        
        address_ptr = nullptr;
    }


    template <typename T>
    inline T* impl_get_data_tlev_ptr(T** data_tlev_ptr, const size_t tidx, const size_t tlevs, allocator_host<T>)
    {
        return(data_tlev_ptr[tidx]);
    }


    template <typename T, typename F>
    void impl_apply(T* data_ptr, F host_func, const twodads::slab_layout_t& geom, const bool is_transformed, const dim3& grid, const dim3& block, allocator_host<T>)
    {
        size_t index{0};
        size_t m{0};

        // Loop over the padded elements if the array is transformed
        const size_t nelem_m{is_transformed ? geom.get_my() + geom.get_pad_y() : geom.get_my()};
        
        const size_t my_plus_pad{geom.get_my() + geom.get_pad_y()};
#pragma omp parallel for private(index, m)
        for(size_t n = 0; n < geom.get_nx(); n++)
        {
        // nelem_m is determined at runtime. To vectorize the loop
        // determine the number of total iterations when handling 4 elements
        // per iteration. The remaining elements are done sequentially
            for(m = 0; m < nelem_m - (nelem_m % 4); m += 4)
            {
                index = n * my_plus_pad + m;
                data_ptr[index] = host_func(data_ptr[index], n, m, geom);
                data_ptr[index + 1] = host_func(data_ptr[index + 1], n, m + 1, geom);
                data_ptr[index + 2] = host_func(data_ptr[index + 2], n, m + 2, geom);
                data_ptr[index + 3] = host_func(data_ptr[index + 3], n, m + 3, geom);
            }
            for(; m < nelem_m; m++)
            {
                index = n * (my_plus_pad) + m;
                data_ptr[index] = host_func(data_ptr[index], n, m, geom);
            }

        }
    }


    template <typename T, typename F>
    void impl_elementwise(T* lhs, T* rhs, F host_func, const twodads::slab_layout_t& geom, const bool is_transformed, const dim3& grid, const dim3& block, allocator_host<T>)
    {
        size_t index{0};
        size_t m{0};
        size_t n{0};

        // Iterate over the padding elements the array is transformed
        // Skip the padding elements if the array is not transformed
        const size_t nelem_m{is_transformed ? geom.get_my() + geom.get_pad_y() : geom.get_my()};
        
        // Use the padded array size to compute indices, whether array is transformed or not.
        const size_t my_plus_pad{geom.get_my() + geom.get_pad_y()};
#pragma omp parallel for private(index, m)
        for(n = 0; n < geom.get_nx(); n++)
        {   
            // Loop vectorization scheme follows host_apply (above)
            for(m = 0; m < nelem_m - (nelem_m % 4); m += 4)
            {
                index = n * my_plus_pad + m;
                lhs[index] = host_func(lhs[index], rhs[index]);
                lhs[index + 1] = host_func(lhs[index + 1], rhs[index + 1]);
                lhs[index + 2] = host_func(lhs[index + 2], rhs[index + 2]);
                lhs[index + 3] = host_func(lhs[index + 3], rhs[index + 3]);
            }
            for(; m < nelem_m; m++)
            {
                index = n * my_plus_pad + m;
                lhs[index] = host_func(lhs[index], rhs[index]);
            }
        }
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
        tmp = nullptr;
    }
}

/**
  .. cpp:class:: template <typename T, template <typename> class allocator> cuda_array_bc_npgp

   Basic 2d vector used in 2dads.
   
   It can store the data of fields, at several time steps.
   It interpolates values of ghost cells by bval_interpolators.


   Memory Layout

   The class maps a two-dimensional array to memory space.
   Data positions are given in rows and columns, where columns are along
   the x-direction and rows are along the y-direction:

   .. math:: 
      n = 0\ldots N_x - 1, N_x, \ldots N_x + \mathrm{pad}_x - 1
   




   Okay, lets  leave math mode
*/

/*

  Columns are stored consecutively in memory. To traverse the array in memory define

   .. math:: 
      idx = n * (M_y + pad_y) + m

*/

/*
       0                        My-1 ... My - 1 + pad_y
   Nx - 1 |--------- ... ------|          |
          |--------- ... ------|          |
   ...
    0     |--------- ... ------|          |
  
   idx = n * (My + pad_y) + m
  
   columns (m, y-direction) are consecutive in memory
  
  
   Mapping of CUDA threads on the array:
  
   Columns: 0..My - 1 + pad_y -> col = blockIdx.x * blockDim.x + threadIdx.x
   Rows:    0..Nx - 1 + pad_x -> row = blockIdx.y * blockDim.y + threadIdx.y
  
   dimBlock = (blocksize_row, blocksize_col)
   dimGrid = (My + pad_y) / blocksize_row, (My + pad_y) / blocksize_col
  
   Ghost points are to be computed on the fly, not stored in memory
   They can be access by the address object

 */


template <typename T, template <typename> class allocator>
class cuda_array_bc_nogp{
public:

    /**
     .. cpp:type:: allocator_type = my_allocator_traits<T, allocator> :: allocator_type

        Declaration of a type alias for the used memory allocator
    */
    using allocator_type = typename my_allocator_traits<T, allocator> :: allocator_type;

    /**
     .. cpp:type:: deleter_type = my_allocator_traits<T, allocator> :: deleter_type

        Declaration of a type alias for the used deleter
    */
    using deleter_type = typename my_allocator_traits<T, allocator> :: deleter_type;

    /**
     .. cpp:type:: ptr_type = std::unique_ptr<T, deleter_type>

       Type alias of the internally used pointers
    */
    using ptr_type = std::unique_ptr<T, deleter_type>;

    // T** pointers
    using p_allocator_type = typename my_allocator_traits<T*, allocator> :: allocator_type;
    using p_deleter_type = typename my_allocator_traits<T*, allocator> :: deleter_type;
    using pptr_type = std::unique_ptr<T*, p_deleter_type>;

    /**
     .. cpp:function:: cuda_array_bc_nogp(const twodads::slab_layout_t, const twodads::bvals_t<T>, size_t _tlevs)

      Default constructor. Takes information on slab layout and boundary conditions. 
      Stores data for _tlevs time levels
    */

	cuda_array_bc_nogp(const twodads::slab_layout_t, const twodads::bvals_t<T>, size_t _tlevs);
    cuda_array_bc_nogp(const cuda_array_bc_nogp<T, allocator>* rhs);
    cuda_array_bc_nogp(const cuda_array_bc_nogp<T, allocator>& rhs);

    /**
     .. cpp:function:: cuda_array_bc_nogp::~cuda_array_bc_nogp()

     Free all allocated resources
     */
	~cuda_array_bc_nogp()
    {
        detail :: impl_delete_address(address_2ptr, address_ptr, allocator_type{});
    };

    /**
     .. cpp:function:: template <typename F> inline void cuda_array_bc_nogp::apply(F myfunc, const size_t tidx)

      Apply F on all array elements at tidx
   
      ======  ====================================================
      Input   Description
      ======  ====================================================
      myfunc  F, functor taking 2 T as input
      tidx    const size_t - Time index on which myfunc is applied
      ======  ====================================================
    */
    template <typename F> inline void apply(F myfunc, const size_t tidx)
    {
        check_bounds(tidx + 1, 0, 0);
        detail :: impl_apply(get_tlev_ptr(tidx), myfunc, get_geom(), is_transformed(tidx), get_grid_unroll(), get_block(), allocator_type{});   
    }

    /**
     .. cpp:function:: template <typename F> inline void cuda_array_bc_nogp::elementwise(F myfunc, const cuda_array_bc_nogp<T, allocator>& rhs, const size_t tidx_rhs, const siz_t tidx_lhs)

       Evaluates myfunc(l, r) elementwise on elements l, r of arrays lhs rhs. 
       Stores result in lhs.

       Input:

       ========  ==================================================
       Input     Description
       ========  ==================================================
       myfunc    callable, takes two T as input, returns T
       rhs       const cuda_array_bc_nogp<T, allocator>&, RHS array
       tidx_rhs  const size_t, time index of RHS array
       tidx_lhs  const size_t, time index of LHS array
       ========  ==================================================

    */
    template<typename F> inline void elementwise(F myfunc, const cuda_array_bc_nogp<T, allocator>& rhs,
                                                 const size_t tidx_rhs, const size_t tidx_lhs)
    {
        check_bounds(tidx_rhs + 1, 0, 0);
        check_bounds(tidx_lhs + 1, 0, 0);
        assert(rhs.get_geom() == get_geom());
        assert(is_transformed(tidx_lhs) == rhs.is_transformed(tidx_rhs));

        detail :: impl_elementwise(get_tlev_ptr(tidx_lhs), rhs.get_tlev_ptr(tidx_rhs), myfunc, get_geom(), is_transformed(tidx_lhs) | rhs.is_transformed(tidx_rhs), get_grid(), get_block(), allocator_type{});
    }

    /**
     .. cpp:function:: template <typename F> inline void cuda_array_bc_nogp::elementwise(F myfunc, const size_t tidx_lhs, const siz_t tidx_rhs)

       Evaluates myfunc(l1, l2) elementwise on elements l1, l2 of arrays lhs on time indices t1 and t2. 
       Stores result in lhs at tidx t1.

       ========  =========================================
       Input     Description
       ========  =========================================
       myfunc    callable, takes two T as input, returns T
       tidx_lhs  const size_t, time index t1 for array
       tidx_rhs  const size_t, time index t2 for array
       ========  =========================================

    */
    template<typename F> inline void elementwise(F myfunc, const size_t tidx_lhs, const size_t tidx_rhs)
    {
        check_bounds(tidx_rhs + 1, 0, 0);
        check_bounds(tidx_lhs + 1, 0, 0);
        detail :: impl_elementwise(get_tlev_ptr(tidx_lhs), get_tlev_ptr(tidx_rhs), myfunc, get_geom(), is_transformed(tidx_lhs) | is_transformed(tidx_rhs), get_grid(), get_block(), allocator_type{});   
    }
       

	/**
     .. cpp:function:: inline void cuda_array_bc_nogp::copy(const size_t tidx_dst, const size_t tidx_src)
      
       Copy data from tidx_src to tidx_dst. 

       ========  =======================================
       Input     Description
       ========  =======================================
       tidx_dst  const size_t, time index of destination
       tidx_src  const size_t, time index of source
       ========  =======================================

     */
	inline void copy(const size_t tidx_dst, const size_t tidx_src)
    {
        check_bounds(tidx_dst + 1, 0, 0);
        check_bounds(tidx_src + 1, 0, 0);
        my_alloc.copy(get_tlev_ptr(tidx_src), get_tlev_ptr(tidx_src) + get_geom().get_nelem_per_t(), get_tlev_ptr(tidx_dst));
        
        set_transformed(tidx_dst, is_transformed(tidx_src));
    }

	/**
     .. cpp:function:: inline void cuda_array_bc_nogp::copy(size_t tidx_dst, const cuda_array_bc_nogp<T, allocator>& src, size_t tidx_src)

        Copy data from array rhs at tidx_src to tidx_dst.

        ========  ==========================================================
        Input     Description
        ========  ========================================================== 
        rhs       const cuda_array_bc_nogp<T, allocator>& rhs: source array
        tidx_dst  const size_t, time index of Destination
        tidx_src  const size_t, time index of source
        ========  ========================================================== 
     
     */
    inline void copy(size_t tidx_dst, const cuda_array_bc_nogp<T, allocator>& src, size_t tidx_src)
    {
        check_bounds(tidx_dst + 1, 0, 0);
        src.check_bounds(tidx_src + 1, 0, 0);
        assert(get_geom() == src.get_geom());
        my_alloc.copy(src.get_tlev_ptr(tidx_src), src.get_tlev_ptr(tidx_src) + src.get_geom().get_nelem_per_t(), get_tlev_ptr(tidx_dst));

        set_transformed(tidx_dst, src.is_transformed(tidx_src));
    }

	// Move data from t_src to t_dst, zero out t_src
    /**
     .. cpp:function: inline void cuda_array_bc_nogp::move(const size_t tidx_dst, const size_t tidx_src)

     Move data from tidx_src to tidx_dst, zero out data at tidx_src

     ======== =======================================
     Input    Description
     ======== =======================================
     tidx_dst const size_t, time index of destination
     tidx_src const size_t, time index of source
     ======== =======================================

     - const size_t tidx_dst: Destination time index

     - const size_t tidx_src: Source time index

     */
	inline void move(const size_t tidx_dst, const size_t tidx_src)
    {
        check_bounds(tidx_dst + 1, 0, 0);
        check_bounds(tidx_src + 1, 0, 0);
        my_alloc.copy(get_tlev_ptr(tidx_src), get_tlev_ptr(tidx_src) + get_geom().get_nelem_per_t(), get_tlev_ptr(tidx_dst));
        apply([] LAMBDACALLER (T dummy, const size_t n, const size_t m, twodads::slab_layout_t geom) -> T {return(0.0);}, 0);
    }

    /**
     .. cpp:function:: inline void cuda_array_bc_nogp::advance()

     Advance data from tidx -> tidx + 1. Zero out data at tidx 0, discard data at last
     
     */
	inline void advance()
    {
        detail :: impl_advance(get_tlev_ptr(), get_tlevs(), allocator_type{});
        apply([] LAMBDACALLER (T dummy, const size_t n, const size_t m, twodads::slab_layout_t geom) -> T {return(0.0);}, 0);
        
        for(size_t tidx = get_tlevs() - 1; tidx > 0; tidx--)
            set_transformed(tidx, is_transformed(tidx - 1));
        set_transformed(0, false);
    }

    /**
     .. cpp:function:: inline size_t cuda_array_bc_nogp :: get_nx() const

     Returns number of discretization points along x-direction.

    */
	inline size_t get_nx() const {return(get_geom().get_nx());};

    /**
     .. cpp:function:: inline size_t cuda_array_bc_nogp::get_my() const

     Returns number of discretization points along y-direction.
     
     */
	inline size_t get_my() const {return(get_geom().get_my());};


    /**
     .. cpp:function:: inline size_t cuda_array_bc_nogp::get_tlevs() const

     Returns the number of time levels

     */
	inline size_t get_tlevs() const {return(tlevs);};

    /**
     .. cpp:function:: inline twodads::slab_layout_t cuda_array_bc_nogp::get_geom() const

     Returns the layout of the array
     
     */
    inline twodads::slab_layout_t get_geom() const {return(geom);};

    /**
     .. cpp:function:: template <typename T> inline twodads::bvals_t<T> cuda_array_bc_nogp::get_bvals() const

     Returns the boundary values of the array
     */
    inline twodads::bvals_t<T> get_bvals() const {return(boundaries);};
    // We are working with 2 pointer levels, since we instantiate an address object 
    // in a cuda kernel in the constructor. That way, we can just pass this
    // pointer to all cuda kernels that need an address object.
    // Call a separate kernel in the destructor to delete it.
    // Unfortunately, this means we have to use 2 pointer levels also in cpu functions.
    inline address_t<T>** get_address_2ptr() const {return(address_2ptr);};
    inline address_t<T>* get_address_ptr() const {return(address_ptr);};

    /**
     .. cpp:function:: inline dim3 get_grid() const

     Returns grid layout for CUDA kernels.

    */
	inline dim3 get_grid() const {return grid;};

    /**
     .. cpp:function:: inline dim3 get_grid_unroll() const

     Return grid layout for CUDA kernels that operate with manual loop-unrolling.

    */
    inline dim3 get_grid_unroll() const {return grid_unroll;};

    /**
     .. cpp:function:: inline dim3 get_block() const

     Return block size for CUDA kernels.

    */
	inline dim3 get_block() const {return block;};

    /**
     .. cpp:function:: template <typename T> inline T* cuda_array_bc_nogp::get_data() const

     Returns pointer to the data array.
     */
	inline T* get_data() const {return data.get();};

    /*
     .. cpp:function:: template <typename T> inline T** cuda_array_bc_nogp::get_tlev_ptr() const

     Retuns pointer to pointer at tlev data.
     */
	inline T** get_tlev_ptr() const {return data_tlev_ptr.get();};

	// Pointer to device data at time level t

    /*
     .. cpp:function:: template <typename T> inline T* cuda_array_bc_nogp::get_tlev_ptr(const size_t tidx) const

     Returns pointer to data at time level tidx.

     =====  =============================
     Input  Description
     =====  =============================
     tidx   const size_t, time level tidx
     =====  =============================

    */

    inline T* get_tlev_ptr(const size_t tidx) const
    {
        check_bounds(tidx + 1, 0, 0);
        return(detail :: impl_get_data_tlev_ptr(get_tlev_ptr(), tidx, get_tlevs(), allocator_type{}));   
    };

    // Set true if transformed
    
    /*
     .. cpp:function:: inline bool cuda_array_bc_nogp<T, allocator<T>> :: is_transformed(const size_t tidx)

     Returns true if array is transformed, false if data is in configuation space

     ====== =======================================================
     Input   Description
     ======  ======================================================
     tidx    Time index where to check whether array is transformed
     ======  ======================================================

    */
    inline bool is_transformed(const size_t tidx) const {check_bounds(tidx + 1, 0, 0); return(transformed[tidx]);};
    inline bool set_transformed(const size_t tidx, const bool val) 
    {
        check_bounds(tidx + 1, 0, 0);
        transformed[tidx] = val; 
        return(transformed[tidx]);
    };

private:
	const twodads::bvals_t<T> boundaries;
    const twodads::slab_layout_t geom;
    const size_t tlevs;
    const bounds check_bounds;
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
    const dim3 grid_unroll;

    // Size of shared memory bank
    const size_t shmem_size_col;   
	// Array data is on device
	// Pointer to device data
	ptr_type data;
	// Pointer to each time stage. Pointer to array of pointers on device
	pptr_type data_tlev_ptr;
};


template <typename T, template<typename> class allocator>
cuda_array_bc_nogp<T, allocator> :: cuda_array_bc_nogp (const twodads::slab_layout_t _geom, const twodads::bvals_t<T> _bvals, const size_t _tlevs) : 
        boundaries(_bvals), 
        geom(_geom), 
        tlevs(_tlevs),
        check_bounds(get_tlevs(), get_nx(), get_my()),
        transformed{std::vector<bool>(get_tlevs(), 0)},
        address_2ptr{nullptr},
        address_ptr{nullptr},
        block(dim3(cuda::blockdim_row, cuda::blockdim_col)),
		grid(dim3(((get_my() + get_geom().get_pad_y()) + cuda::blockdim_row - 1) / cuda::blockdim_row, 
                  ((get_nx() + get_geom().get_pad_x()) + cuda::blockdim_col - 1) / cuda::blockdim_col)),
        grid_unroll(grid.x / cuda :: elem_per_thread, grid.y),
        shmem_size_col(get_nx() * sizeof(T)),
        data(my_alloc.allocate(get_tlevs() * get_geom().get_nelem_per_t())),
		data_tlev_ptr(my_palloc.allocate(get_tlevs()))
{
    //std::cout << "block3: x = " << block.x << ", y = " << block.y << ", z = " << block.z << std::endl;
    //std::cout << "grid3: x = " << grid.x << ", y = " << grid.y << ", z = " << grid.z << std::endl;
    //std::cout << "grid_unrolled: x = " << grid_unroll.x << ", y= " << grid_unroll.y << " y = " << grid_unroll.z << std::endl;
    // Set the pointer in array_tlev_ptr to data[0], data[0] + get_nelem_per_t(), data[0] + 2 * get_nelem_per_t() ...
    detail :: impl_set_data_tlev_ptr(get_data(), get_tlev_ptr(), get_tlevs(), get_geom(), allocator_type{});
    
    // Initialize the address object
    detail :: impl_init_address(address_2ptr, address_ptr, get_geom(), get_bvals(), allocator_type{});
    for(size_t tidx = 0; tidx < tlevs; tidx++)
    {
        // Also initialize the padded elements
        set_transformed(tidx, true);
        apply([] LAMBDACALLER (T dummy, const size_t n, const size_t m, twodads::slab_layout_t geom) -> T {return(0.0);}, tidx);
        set_transformed(tidx, false);
    }
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


#endif // cuda_array_bc_H_ 
