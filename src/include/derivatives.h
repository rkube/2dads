#ifndef DERIVATIVES_H
#define DERIVATIVES_H


#include "cuda_array_bc_nogp.h"
#include "cucmplx.h"
#include "error.h"
#include "dft_type.h"
#include "solvers.h"
#include "utility.h"

#include <iostream>
#include <cassert>
#include <sstream>
#include <fstream>

#ifdef __CUDACC__
#include "cuda_types.h"
#include <cusolverSp.h>
#include <cublas_v2.h>
#endif //__CUDACC__


enum class direction {x, y};

namespace device
{
#ifdef __CUDACC__
// Apply three point stencil to points within the domain, rows 1..Nx-2
template <typename T, typename O>
__global__
void kernel_threepoint_center(const T* u, address_t<T>** address_u,
                              T* result, O stencil_func, const twodads::slab_layout_t geom)
{
    const int col{static_cast<int>(cuda :: thread_idx :: get_col())};
    const int row{static_cast<int>(cuda :: thread_idx :: get_row())};
    const size_t index{row * (geom.get_my() + geom.get_pad_y()) + col};
    const T inv_dx{1.0 / geom.get_deltax()};
    const T inv_dx2{inv_dx * inv_dx};

    if(row > 0 && row < static_cast<int>(geom.get_nx() - 1) && col >= 0 && col < static_cast<int>(geom.get_my()))
    {
        result[index] = stencil_func((**address_u).get_elem(u, row - 1, col),
                                     (**address_u).get_elem(u, row    , col),
                                     (**address_u).get_elem(u, row + 1, col),
                                     inv_dx, inv_dx2);
    }
}


// Apply three point stencil at a single row. Use address_t<T>.operator() for element access
template <typename T, typename O>
__global__
void kernel_threepoint_single_row(const T* u, address_t<T>** address_u,
                                  T* result, O stencil_func, 
                                  const twodads::slab_layout_t geom, const int row)
{
    const int col{static_cast<int>(cuda :: thread_idx :: get_col())};
    const size_t index{row * (geom.get_my() + geom.get_pad_y()) + col};
    const T inv_dx{1.0 / geom.get_deltax()};
    const T inv_dx2{inv_dx * inv_dx};

    if(col >= 0 && col < static_cast<int>(geom.get_my()))
    {
        result[index] = stencil_func((**address_u)(u, row - 1, col),
                                     (**address_u)(u, row    , col),
                                     (**address_u)(u, row + 1, col),
                                     inv_dx, inv_dx2);
    }
}


// T* u is the data pointed to by a cuda_array u, address_u its address object
// T* u is the data pointed to by a cuda_array v, address_v its address object
// Assume that u and v have the same geometry
template <typename T>
__global__
void kernel_arakawa_center(const T* u, address_t<T>** address_u, 
                           const T* v, address_t<T>** address_v, 
                           T* result, const twodads::slab_layout_t geom)
{
    const int col{static_cast<int>(cuda :: thread_idx :: get_col())};
    const int row{static_cast<int>(cuda :: thread_idx :: get_row())};
    const size_t index{row * (geom.get_my() + geom.get_pad_y()) + col}; 

    const T inv_dx_dy{-1.0 / (12.0 * geom.get_deltax() * geom.get_deltay())};
    // This checks whether we are at an inside point when calling this kernel with a thread layout
    // that covers the entire grid

    if(row > 0 && row < static_cast<int>(geom.get_nx() - 1) && col > 0 && col < static_cast<int>(geom.get_my() - 1))
    {
        //printf("threadIdx.x = %d, blockIdx.x = %d, blockDim.x = %d, threadIdx.y = %d, blockIdx.y = %d, blockDim.y = %d,row = %d, col = %d, Nx = %d, My = %d\n", 
        //        threadIdx.x, blockIdx.x, blockDim.x, threadIdx.y, blockIdx.y, blockDim.y,
        //        row, col, static_cast<int>(geom.get_nx()), static_cast<int>(geom.get_my()));
        result[index] = 
        ((((**address_u).get_elem(u, row    , col - 1) + 
           (**address_u).get_elem(u, row + 1, col - 1) - 
           (**address_u).get_elem(u, row    , col + 1) - 
           (**address_u).get_elem(u, row + 1, col + 1))
          *
          ((**address_v).get_elem(v, row + 1, col    ) + 
           (**address_v).get_elem(v, row    , col    )))
         -
         (((**address_u).get_elem(u, row - 1, col - 1) +
           (**address_u).get_elem(u, row    , col - 1) -
           (**address_u).get_elem(u, row - 1, col + 1) -
           (**address_u).get_elem(u, row    , col + 1))
          *
          ((**address_v).get_elem(v, row    , col    ) +
           (**address_v).get_elem(v, row - 1, col    )))
         +
         (((**address_u).get_elem(u, row + 1, col    ) +
           (**address_u).get_elem(u, row + 1, col + 1) -
           (**address_u).get_elem(u, row - 1, col    ) -
           (**address_u).get_elem(u, row - 1, col + 1))
          *
          ((**address_v).get_elem(v, row    , col + 1) +
           (**address_v).get_elem(v, row    , col    )))
         -
         (((**address_u).get_elem(u, row + 1, col - 1) +
           (**address_u).get_elem(u, row + 1, col    ) -
           (**address_u).get_elem(u, row - 1, col - 1) -
           (**address_u).get_elem(u, row - 1, col    ))
          *
          ((**address_v).get_elem(v, row    , col    ) +
           (**address_v).get_elem(v, row    , col - 1)))
         +
         (((**address_u).get_elem(u, row + 1, col    ) -
           (**address_u).get_elem(u, row    , col + 1))
          *
          ((**address_v).get_elem(v, row + 1, col + 1) +
           (**address_v).get_elem(v, row    , col    )))
         
         -
         (((**address_u).get_elem(u, row    , col - 1) -
           (**address_u).get_elem(u, row - 1, col    ))
          *
          ((**address_v).get_elem(v, row    , col    ) +
           (**address_v).get_elem(v, row - 1, col - 1)))

         +
         (((**address_u).get_elem(u, row    , col + 1) -
           (**address_u).get_elem(u, row - 1, col    ))
          *
          ((**address_v).get_elem(v, row - 1, col + 1) +
           (**address_v).get_elem(v, row    , col    )))
         -
         (((**address_u).get_elem(u, row + 1, col    ) -
           (**address_u).get_elem(u, row    , col - 1))
          *
          ((**address_v).get_elem(v, row    , col    ) +
           (**address_v).get_elem(v, row + 1, col - 1)))
         )
         * inv_dx_dy;
    }
};


// Kernel operates on elements with n = 0, m = 0..My-1. Extrapolate left ghost points (n = -1) of u and v on the fly
// address_u and address_v provide operator() which wrap the index and interpolate to ghost points
// when n = -1 or n = Nx.
template <typename T> 
__global__
void kernel_arakawa_single_row(const T* u, address_t<T>** address_u,
                               const T* v, address_t<T>** address_v,
                               T* result, const twodads::slab_layout_t geom,
                               const int row)
{
    // Use int for col and row to pass them into address<T>::operator()
    const int col{static_cast<int>(cuda :: thread_idx :: get_col())};
    const size_t index{row * (geom.get_my() + geom.get_pad_y()) + col}; 

    const T inv_dx_dy{1.0 / (12.0 * geom.get_deltax() * geom.get_deltay())};

    if(col < static_cast<int>(geom.get_my()))
    {
        result[index] =  
        (
        (((**address_u)(u, row    , col - 1) + 
          (**address_u)(u, row + 1, col - 1) - 
          (**address_u)(u, row    , col + 1) - 
          (**address_u)(u, row + 1, col + 1))
        *
         ((**address_v)(v, row + 1, col    ) + 
          (**address_v)(v, row    , col    ))
        -
        (((**address_u)(u, row - 1, col - 1) + 
          (**address_u)(u, row    , col - 1) - 
          (**address_u)(u, row - 1, col + 1) - 
          (**address_u)(u, row    , col + 1))
        *
         ((**address_v)(v, row    , col    ) + 
          (**address_v)(v, row - 1, col    )))
        +
        (((**address_u)(u, row + 1, col    ) + 
          (**address_u)(u, row + 1, col + 1) - 
          (**address_u)(u, row - 1, col    ) - 
          (**address_u)(u, row - 1, col + 1))
        *
         ((**address_v)(v, row    , col + 1) + 
          (**address_v)(v, row    , col    )))
        -
        (((**address_u)(u, row + 1, col - 1) + 
          (**address_u)(u, row + 1, col    ) - 
          (**address_u)(u, row - 1, col - 1) - 
          (**address_u)(u, row - 1, col    ))
        *
         ((**address_v)(v, row    , col    ) + 
          (**address_v)(v, row    , col - 1)))
        +
         ((**address_u)(u, row + 1, col    ) - 
          (**address_u)(u, row    , col + 1)) 
        * 
         ((**address_v)(v, row + 1, col + 1) + 
          (**address_v)(v, row    , col    ))
        -
         ((**address_u)(u, row    , col - 1) - 
          (**address_u)(u, row - 1, col    ))
        *
         ((**address_v)(v, row    , col    ) + 
          (**address_v)(v, row - 1, col - 1)) 
        +
         ((**address_u)(u, row    , col + 1) - 
          (**address_u)(u, row - 1, col    ))
        *
         ((**address_v)(v, row - 1, col + 1) + 
          (**address_v)(v, row    , col    ))
        -
         ((**address_u)(u, row + 1, col    ) - 
          (**address_u)(u, row    , col - 1))
        *
         ((**address_v)(v, row    , col    ) + 
          (**address_v)(v, row + 1, col - 1)))
        ) * inv_dx_dy;
    }
}


// Kernel operates on elements with n = 0..Nx-1, m = My-1. Computes top ghost points of u and v on the fly
template <typename T>
__global__
void kernel_arakawa_single_col(const T* u, address_t<T>** address_u,
                               const T* v, address_t<T>** address_v,
                               T* result, const twodads::slab_layout_t geom, const int col)
{
    const int row{static_cast<int>(cuda :: thread_idx :: get_row())};
    const size_t index{row * (geom.get_my() + geom.get_pad_y()) + col}; 
    const T inv_dx_dy{-1.0 / (12.0 * geom.get_deltax() * geom.get_deltay())};

    if(row > 0 && row < static_cast<int>(geom.get_nx() - 1))
    {
        result[index] = 
        (
        (((**address_u)(u, row    , col - 1) + 
          (**address_u)(u, row + 1, col - 1) - 
          (**address_u)(u, row    , col + 1) - 
          (**address_u)(u, row + 1, col + 1))
        *
         ((**address_v)(v, row + 1, col    ) + 
          (**address_v)(v, row    , col    ))
        -
        (((**address_u)(u, row - 1, col - 1) + 
          (**address_u)(u, row    , col - 1) - 
          (**address_u)(u, row - 1, col + 1) - 
          (**address_u)(u, row    , col + 1))
        *
         ((**address_v)(v, row    , col    ) + 
          (**address_v)(v, row - 1, col    )))
        +
        (((**address_u)(u, row + 1, col    ) + 
          (**address_u)(u, row + 1, col + 1) - 
          (**address_u)(u, row - 1, col    ) - 
          (**address_u)(u, row - 1, col + 1))
        *
         ((**address_v)(v, row    , col + 1) + 
          (**address_v)(v, row    , col    )))
        -
        (((**address_u)(u, row + 1, col - 1) + 
          (**address_u)(u, row + 1, col    ) - 
          (**address_u)(u, row - 1, col - 1) - 
          (**address_u)(u, row - 1, col    ))
        *
         ((**address_v)(v, row    , col    ) + 
          (**address_v)(v, row    , col - 1)))
        +
         ((**address_u)(u, row + 1, col    ) - 
          (**address_u)(u, row    , col + 1)) 
        * 
         ((**address_v)(v, row + 1, col + 1) + 
          (**address_v)(v, row    , col    ))
        -
         ((**address_u)(u, row    , col - 1) - 
          (**address_u)(u, row - 1, col    ))
        *
         ((**address_v)(v, row    , col    ) + 
          (**address_v)(v, row - 1, col - 1)) 
        +
         ((**address_u)(u, row    , col + 1) - 
          (**address_u)(u, row - 1, col    ))
        *
         ((**address_v)(v, row - 1, col + 1) + 
          (**address_v)(v, row    , col    ))
        -
         ((**address_u)(u, row + 1, col    ) - 
          (**address_u)(u, row    , col - 1))
        *
         ((**address_v)(v, row    , col    ) + 
          (**address_v)(v, row + 1, col - 1)))
        ) * inv_dx_dy;
    }
}
#endif //__CUDACC__
} // namespace device


namespace host
{
    template <typename T, typename O>
    void apply_threepoint_center(T* u, address_t<T>* address_u, T* res, O stencil_func, const twodads::slab_layout_t& geom)
    {
        const T inv_dx{1.0 / geom.get_deltax()};
        const T inv_dx2{inv_dx * inv_dx};

        for(size_t n = 1; n < geom.get_nx() - 1; n++)
        {
            for(size_t m = 0; m < geom.get_my(); m++)
            {
                res[n * (geom.get_my() + geom.get_pad_y()) + m] = stencil_func((*address_u).get_elem(u, n - 1, m),
                                                                               (*address_u).get_elem(u, n    , m),
                                                                               (*address_u).get_elem(u, n + 1, m),
                                                                               inv_dx, inv_dx2);
            }
        }
    }


    template <typename T, typename O> 
    void apply_threepoint(T* u, address_t<T>* address_u, T* res, O stencil_func, const twodads::slab_layout_t& geom,
                          std::vector<size_t>& row_vals, std::vector<size_t>& col_vals)
    {
        const T inv_dx{1.0 / geom.get_deltax()};
        const T inv_dx2{inv_dx * inv_dx};

        for(auto row : row_vals)
        {
            for(auto col : col_vals)
            {
                res[row * (geom.get_my() + geom.get_pad_y()) + col] = stencil_func((*address_u)(u, row - 1, col),
                                                                                   (*address_u)(u, row    , col),
                                                                                   (*address_u)(u, row + 1, col),
                                                                                   inv_dx, inv_dx2);
            }
        }
    }

    // TODO: Check if this function can be replaced by elementwise...
    template <typename T, typename O>
    void multiply_map(CuCmplx<T>* in, CuCmplx<T>* map, CuCmplx<T>* out, O op_func, twodads::slab_layout_t geom)
    {
        size_t index{0};
        for(size_t row = 0; row < geom.get_nx(); row++)
        {
            for(size_t col = 0; col < geom.get_my(); col++)
            {
                index = row * (geom.get_my() + geom.get_pad_y()) + col; 
                out[index] = op_func(in[index], map[index]);
            }
        }
    }

    template <typename T>
    void arakawa_center(const T* u, address_t<T>* address_u, 
                        const T* v, address_t<T>* address_v, 
                        T* result, const twodads::slab_layout_t& geom)
    {
        const T inv_dx_dy{-1.0 / (12.0 * geom.get_deltax() * geom.get_deltay())};
        size_t index{0};
        for(size_t row = 1; row < geom.get_nx() - 1; row++)
        {
           for(size_t col = 1; col < geom.get_my() - 1; col++)
            {
            index = (row * (geom.get_my() + geom.get_pad_y()) + col);
            result[index] = 
                ((((*address_u).get_elem(u, row    , col - 1) + 
                   (*address_u).get_elem(u, row + 1, col - 1) - 
                   (*address_u).get_elem(u, row    , col + 1) - 
                   (*address_u).get_elem(u, row + 1, col + 1))
                  *
                  ((*address_v).get_elem(v, row + 1, col    ) + 
                   (*address_v).get_elem(v, row    , col    )))
                 -
                 (((*address_u).get_elem(u, row - 1, col - 1) +
                   (*address_u).get_elem(u, row    , col - 1) -
                   (*address_u).get_elem(u, row - 1, col + 1) -
                   (*address_u).get_elem(u, row    , col + 1))
                  *
                  ((*address_v).get_elem(v, row    , col    ) +
                   (*address_v).get_elem(v, row - 1, col    )))
                 +
                 (((*address_u).get_elem(u, row + 1, col    ) +
                   (*address_u).get_elem(u, row + 1, col + 1) -
                   (*address_u).get_elem(u, row - 1, col    ) -
                   (*address_u).get_elem(u, row - 1, col + 1))
                  *
                  ((*address_v).get_elem(v, row    , col + 1) +
                   (*address_v).get_elem(v, row    , col    )))
                 -
                 (((*address_u).get_elem(u, row + 1, col - 1) +
                   (*address_u).get_elem(u, row + 1, col    ) -
                   (*address_u).get_elem(u, row - 1, col - 1) -
                   (*address_u).get_elem(u, row - 1, col    ))
                  *
                  ((*address_v).get_elem(v, row    , col    ) +
                   (*address_v).get_elem(v, row    , col - 1)))
                 +
                 (((*address_u).get_elem(u, row + 1, col    ) -
                   (*address_u).get_elem(u, row    , col + 1))
                  *
                  ((*address_v).get_elem(v, row + 1, col + 1) +
                   (*address_v).get_elem(v, row    , col    )))

                 -
                 (((*address_u).get_elem(u, row    , col - 1) -
                   (*address_u).get_elem(u, row - 1, col    ))
                  *
                  ((*address_v).get_elem(v, row    , col    ) +
                   (*address_v).get_elem(v, row - 1, col - 1)))

                 +
                 (((*address_u).get_elem(u, row    , col + 1) -
                   (*address_u).get_elem(u, row - 1, col    ))
                  *
                  ((*address_v).get_elem(v, row - 1, col + 1) +
                   (*address_v).get_elem(v, row    , col    )))
                 -
                 (((*address_u).get_elem(u, row + 1, col    ) -
                   (*address_u).get_elem(u, row    , col - 1))
                  *
                  ((*address_v).get_elem(v, row    , col    ) +
                   (*address_v).get_elem(v, row + 1, col - 1)))
         )
         * inv_dx_dy;
        }
            }
    }

    template <typename T>
    void arakawa_single(const T* u, address_t<T>* address_u, 
                        const T* v, address_t<T>* address_v, 
                        T* result, const twodads::slab_layout_t& geom,
                        std::vector<size_t> row_vals,
                        std::vector<size_t> col_vals)
    {
        const T inv_dx_dy{-1.0 / (12.0 * geom.get_deltax() * geom.get_deltay())};
        size_t index{0};
        for(size_t row : row_vals)
        {
            for(size_t col : col_vals)
            {
                index = (row * (geom.get_my() + geom.get_pad_y()) + col);
                result[index] = 
                   ((((*address_u)(u, row    , col - 1) + 
                      (*address_u)(u, row + 1, col - 1) - 
                      (*address_u)(u, row    , col + 1) - 
                      (*address_u)(u, row + 1, col + 1))
                     *
                     ((*address_v)(v, row + 1, col    ) + 
                      (*address_v)(v, row    , col    )))
                    -
                    (((*address_u)(u, row - 1, col - 1) +
                      (*address_u)(u, row    , col - 1) -
                      (*address_u)(u, row - 1, col + 1) -
                      (*address_u)(u, row    , col + 1))
                     *
                     ((*address_v)(v, row    , col    ) +
                      (*address_v)(v, row - 1, col    )))
                    +
                    (((*address_u)(u, row + 1, col    ) +
                      (*address_u)(u, row + 1, col + 1) -
                      (*address_u)(u, row - 1, col    ) -
                      (*address_u)(u, row - 1, col + 1))
                     *
                     ((*address_v)(v, row    , col + 1) +
                      (*address_v)(v, row    , col    )))
                    -
                    (((*address_u)(u, row + 1, col - 1) +
                      (*address_u)(u, row + 1, col    ) -
                      (*address_u)(u, row - 1, col - 1) -
                      (*address_u)(u, row - 1, col    ))
                     *
                     ((*address_v)(v, row    , col    ) +
                      (*address_v)(v, row    , col - 1)))
                    +
                    (((*address_u)(u, row + 1, col    ) -
                      (*address_u)(u, row    , col + 1))
                     *
                     ((*address_v)(v, row + 1, col + 1) +
                      (*address_v)(v, row    , col    )))

                    -
                    (((*address_u)(u, row    , col - 1) -
                      (*address_u)(u, row - 1, col    ))
                     *
                     ((*address_v)(v, row    , col    ) +
                      (*address_v)(v, row - 1, col - 1)))

                    +
                    (((*address_u)(u, row    , col + 1) -
                      (*address_u)(u, row - 1, col    ))
                     *
                     ((*address_v)(v, row - 1, col + 1) +
                      (*address_v)(v, row    , col    )))
                    -
                    (((*address_u)(u, row + 1, col    ) -
                      (*address_u)(u, row    , col - 1))
                     *
                     ((*address_v)(v, row    , col    ) +
                      (*address_v)(v, row + 1, col - 1)))
                    )
                 * inv_dx_dy;
           }    
        }
    }
}

namespace detail
{
    namespace fd
    {
#ifdef CUDACC
        template <typename T>
        void impl_dx(const cuda_array_bc_nogp<T, allocator_device>& in,
                    cuda_array_bc_nogp<T, allocator_device>& out,
                    const size_t t_src, const size_t t_dst, const size_t order, allocator_device<T>)
        {
            static dim3 block_single_row(cuda::blockdim_row, 1);
            static dim3 grid_single_row((in.get_geom().get_nx() + cuda::blockdim_row - 1) / cuda::blockdim_row, 1);

            // First and second order derivatives are both implemented as three-point stencils.
            // How the stencils are applied is the same for first and second order. Only the
            // exact stencil scheme is different.
            if(order == 1)
            {
                // Call kernel that accesses elements with get_elem; no wrapping/interpolation
                device :: kernel_threepoint_center<<<in.get_grid(), in.get_block()>>>(in.get_tlev_ptr(t_src), in.get_address_2ptr(),
                        out.get_tlev_ptr(t_dst), 
                        [] __device__ (T u_left, T u_middle, T u_right, T inv_dx, T inv_dx2) -> T
                        {return(0.5 * (u_right - u_left) * inv_dx);},
                        out.get_geom());
                gpuErrchk(cudaPeekAtLastError());
                // Call kernel that accesses elements with operator(); interpolates ghost point values
                device :: kernel_threepoint_single_row<<<grid_single_row, block_single_row>>>(in.get_tlev_ptr(t_src), in.get_address_2ptr(),
                        out.get_tlev_ptr(t_dst), 
                        [] __device__ (T u_left, T u_middle, T u_right, T inv_dx, T inv_dx2) -> T
                        {return(0.5 * (u_right - u_left) * inv_dx);},
                        out.get_geom(), 0);
                gpuErrchk(cudaPeekAtLastError());

                // Call kernel that accesses elements with operator(); interpolates ghost point values
                device :: kernel_threepoint_single_row<<<grid_single_row, block_single_row>>>(in.get_tlev_ptr(t_src), in.get_address_2ptr(),
                        out.get_tlev_ptr(t_dst), 
                        [] __device__ (T u_left, T u_middle, T u_right, T inv_dx, T inv_dx2) -> T
                        {return(0.5 * (u_right - u_left) * inv_dx);},
                        out.get_geom(), out.get_geom().get_nx() - 1);
                gpuErrchk(cudaPeekAtLastError());
            }
            else if (order == 2)
            {
                // Call kernel that accesses elements with get_elem; no wrapping around
                device :: kernel_threepoint_center<<<in.get_grid(), in.get_block()>>>(in.get_tlev_ptr(t_src), in.get_address_2ptr(),
                        out.get_tlev_ptr(t_dst), 
                        [] __device__ (T u_left, T u_middle, T u_right, T inv_dx, T inv_dx2) -> T
                        {return((u_left + u_right - 2.0 * u_middle) * inv_dx2);},
                        out.get_geom());
                gpuErrchk(cudaPeekAtLastError());

                // Call kernel that accesses elements with operator(); interpolates ghost point values
                device :: kernel_threepoint_single_row<<<grid_single_row, block_single_row>>>(in.get_tlev_ptr(t_src), in.get_address_2ptr(),
                        out.get_tlev_ptr(t_dst), 
                        [] __device__ (T u_left, T u_middle, T u_right, T inv_dx, T inv_dx2) -> T
                        {return((u_left + u_right - 2.0 * u_middle) * inv_dx2);}, 
                        out.get_geom(), 0);
                gpuErrchk(cudaPeekAtLastError());

                // Call kernel that accesses elements with operator(); interpolates ghost point values
                device :: kernel_threepoint_single_row<<<grid_single_row, block_single_row>>>(in.get_tlev_ptr(t_src), in.get_address_2ptr(),
                        out.get_tlev_ptr(t_dst), 
                        [] __device__ (T u_left, T u_middle, T u_right, T inv_dx, T inv_dx2) -> T
                        {return((u_left + u_right - 2.0 * u_middle) * inv_dx2);},
                        out.get_geom(), out.get_geom().get_nx() - 1);
                gpuErrchk(cudaPeekAtLastError());
            }
        }


        template <typename T>
        void impl_dy(const cuda_array_bc_nogp<T, allocator_device>& src,
                    cuda_array_bc_nogp<T, allocator_device>& dst,
                    const size_t t_src, const size_t t_dst, const size_t order,
                    const cuda_array_bc_nogp<twodads::cmplx_t, allocator_device>& coeffs_map_d1,
                    const cuda_array_bc_nogp<twodads::cmplx_t, allocator_device>& coeffs_map_d2, 
                    twodads::slab_layout_t geom_my21, allocator_device<T>)
        {
            const dim3 block_my21(cuda::blockdim_col, cuda::blockdim_row);
            const dim3 grid_my21((geom_my21.get_my() + cuda::blockdim_col - 1) / cuda::blockdim_col,
                                (geom_my21.get_nx() + cuda::blockdim_row - 1) / (cuda::blockdim_row));

            // Multiply with coefficients for ky
            // Coefficients for first and second order are stored in different maps.abs
            // Also, for first order we use 
            //      u_y_hat[index] = u_hat[index] * (0.0, I * ky)
            // while second order is
            //      u_y_hat[index] = u_hat[index] * (I * ky)^2

            if(order == 1)
                device :: kernel_multiply_map<<<grid_my21, block_my21>>>(reinterpret_cast<CuCmplx<T>*>(src.get_tlev_ptr(t_src)),
                    coeffs_map_d1.get_tlev_ptr(0), 
                    reinterpret_cast<CuCmplx<T>*>(dst.get_tlev_ptr(t_dst)),
                    [] __device__ (CuCmplx<T> val_in, CuCmplx<T> val_map) -> CuCmplx<T>
                    {return(val_in * CuCmplx<T>(0.0, val_map.im()));},
                    geom_my21);

            else if(order == 2)
                device :: kernel_multiply_map<<<grid_my21, block_my21>>>(reinterpret_cast<CuCmplx<T>*>(src.get_tlev_ptr(t_src)),
                    coeffs_map_d2.get_tlev_ptr(0), reinterpret_cast<CuCmplx<T>*>(dst.get_tlev_ptr(t_dst)),
                    [] __device__ (CuCmplx<T> val_in, CuCmplx<T> val_map) -> CuCmplx<T>
                    {return(val_in * val_map.im());},
                    geom_my21);

            gpuErrchk(cudaPeekAtLastError());
        }



        template <typename T>
        void impl_arakawa(const cuda_array_bc_nogp<T, allocator_device>& u,
                        const cuda_array_bc_nogp<T, allocator_device>& v,
                        cuda_array_bc_nogp<T, allocator_device> res,
                        const size_t t_srcu, const size_t t_srcv, 
                        const size_t t_dst, allocator_device<T>)
        {
            // Thread layout for accessing a single row (m = 0..My-1, n = 0, Nx-1)
            static dim3 block_single_row(cuda::blockdim_row, 1);
            static dim3 grid_single_row((u.get_geom().get_nx() + cuda::blockdim_row - 1) / cuda::blockdim_row, 1);

            // Thread layout for accessing a single column (m = 0, My - 1, n = 0...Nx-1)
            static dim3 block_single_col(1, cuda::blockdim_col);
            static dim3 grid_single_col(1, (u.get_geom().get_my() + cuda::blockdim_col - 1) / cuda::blockdim_col);

            device :: kernel_arakawa_center<<<u.get_grid(), u.get_block()>>>(u.get_tlev_ptr(t_srcu), u.get_address_2ptr(),
                    v.get_tlev_ptr(t_srcv), v.get_address_2ptr(),
                    res.get_tlev_ptr(t_dst), u.get_geom());
            gpuErrchk(cudaPeekAtLastError());

            // Create address objects to access ghost points 
            device :: kernel_arakawa_single_row<<<grid_single_row, block_single_row>>>(u.get_tlev_ptr(t_srcu), u.get_address_2ptr(),
                    v.get_tlev_ptr(t_srcv), v.get_address_2ptr(),
                    res.get_tlev_ptr(t_dst), u.get_geom(), 0);
            gpuErrchk(cudaPeekAtLastError());

            device :: kernel_arakawa_single_row<<<grid_single_row, block_single_row>>>(u.get_tlev_ptr(t_srcu), u.get_address_2ptr(),
                    v.get_tlev_ptr(t_srcv), v.get_address_2ptr(),
                    res.get_tlev_ptr(t_dst), u.get_geom(), u.get_geom().get_nx() - 1);
            gpuErrchk(cudaPeekAtLastError());

            device :: kernel_arakawa_single_col<<<grid_single_col, block_single_col>>>(u.get_tlev_ptr(t_srcu), u.get_address_2ptr(),
                    v.get_tlev_ptr(t_srcv), v.get_address_2ptr(),
                    res.get_tlev_ptr(t_dst), u.get_geom(), 0);
            gpuErrchk(cudaPeekAtLastError());

            device :: kernel_arakawa_single_col<<<grid_single_col, block_single_col>>>(u.get_tlev_ptr(t_srcu), u.get_address_2ptr(),
                    v.get_tlev_ptr(t_srcv), v.get_address_2ptr(),
                    res.get_tlev_ptr(t_dst), u.get_geom(), u.get_geom().get_my() - 1);
            gpuErrchk(cudaPeekAtLastError());
        }

        template <typename T>
        void impl_invert_laplace(const cuda_array_bc_nogp<T, allocator_device>& src,
                                cuda_array_bc_nogp<T, allocator_device>& dst,
                                const size_t t_src, const size_t t_dst,
                                cuda_array_bc_nogp<CuCmplx<T>, allocator_device>& diag,
                                cuda_array_bc_nogp<CuCmplx<T>, allocator_device>& diag_u,
                                cuda_array_bc_nogp<CuCmplx<T>, allocator_device>& diag_l,
                                solvers :: elliptic_base_t* ell_solver,
                                allocator_device<T>)                         
        {
            ell_solver -> solve(reinterpret_cast<CuCmplx<T>*>(src.get_tlev_ptr(t_src)), 
                                reinterpret_cast<CuCmplx<T>*>(dst.get_tlev_ptr(t_dst)),
                                diag_l.get_tlev_ptr(0), 
                                diag.get_tlev_ptr(0), 
                                diag_u.get_tlev_ptr(0));

            dst.set_transformed(t_dst, true);
        }
#endif //__CUDACC__


        template <typename T>
        void impl_dx(const cuda_array_bc_nogp<T, allocator_host>& in,
                    cuda_array_bc_nogp<T, allocator_host>& out,
                    const size_t t_src, const size_t t_dst, const size_t order, allocator_host<T>)
        {
            std::vector<size_t> col_vals(in.get_geom().get_my());
            std::vector<size_t> row_vals(1);

            row_vals[0] = 0;
            for(size_t m = 0; m < in.get_geom().get_my(); m++)
                col_vals[m] = m;

            if(order == 1)
            // Calculate the first derivative
            {
                // Apply threepoint stencil in interior domain, no interpolation here
                host :: apply_threepoint_center(in.get_tlev_ptr(t_src), in.get_address_ptr(), out.get_tlev_ptr(t_dst), 
                                                [] (T u_left, T u_middle, T u_right, T inv_dx, T inv_dx2) -> T
                                                {return(0.5 * (u_right - u_left) * inv_dx);}, 
                                                out.get_geom());

                // Call expensive interpolation routine only for 2 rows
                // 1) row n=0, m = 0...my-1
                host :: apply_threepoint(in.get_tlev_ptr(t_src), in.get_address_ptr(), out.get_tlev_ptr(t_dst), 
                                         [] (T u_left, T u_middle, T u_right, T inv_dx, T inv_dx2) -> T
                                         {return(0.5 * (u_right - u_left) * inv_dx);},
                                         out.get_geom(), row_vals, col_vals);

                // 2) row n=Nx - 1, m = 0..My-1
                row_vals[0] = in.get_geom().get_nx() - 1;
                host :: apply_threepoint(in.get_tlev_ptr(t_src), in.get_address_ptr(), out.get_tlev_ptr(t_dst), 
                                         [] (T u_left, T u_middle, T u_right, T inv_dx, T inv_dx2) -> T
                                         {return(0.5 * (u_right - u_left) * inv_dx);},
                                         out.get_geom(), row_vals, col_vals);
            }
            else if (order == 2)
            // Calculate the second derivative
            {
                // Apply threepoint stencil in interior domain, no interpolation here
                host :: apply_threepoint_center(in.get_tlev_ptr(t_src), in.get_address_ptr(), out.get_tlev_ptr(t_dst), 
                                                [=] (T u_left, T u_middle, T u_right, T inv_dx, T inv_dx2) -> T
                                                {return((u_left + u_right - 2.0 * u_middle) * inv_dx2);},
                                                out.get_geom());

                // Call expensive interpolation routine only for 2 columns
                host :: apply_threepoint(in.get_tlev_ptr(t_src), in.get_address_ptr(), out.get_tlev_ptr(t_dst),
                                        [=] (T u_left, T u_middle, T u_right, T inv_dx, T inv_dx2) -> T
                                        {return((u_left + u_right - 2.0 * u_middle) * inv_dx2);},
                                        out.get_geom(), row_vals, col_vals);

                row_vals[0] = in.get_geom().get_nx() - 1;
                host :: apply_threepoint(in.get_tlev_ptr(t_src), in.get_address_ptr(), out.get_tlev_ptr(t_dst),
                                        [=] (T u_left, T u_middle, T u_right, T inv_dx, T inv_dx2) -> T
                                        {return((u_left + u_right - 2.0 * u_middle) * inv_dx2);},
                                        out.get_geom(), row_vals, col_vals);
                }
            else
            {
                throw not_implemented_error(std::string("Derivatives order > 2 are not implemented"));
            }
        }


        template <typename T>
        void impl_dy(const cuda_array_bc_nogp<T, allocator_host>& src,
                    cuda_array_bc_nogp<T, allocator_host>& dst,
                    const size_t t_src, const size_t t_dst, const size_t order,
                    const cuda_array_bc_nogp<twodads::cmplx_t, allocator_host>& coeffs_map_d1,
                    const cuda_array_bc_nogp<twodads::cmplx_t, allocator_host>& coeffs_map_d2, 
                    twodads::slab_layout_t geom_my21, allocator_host<T>)
        {
            // Multiply with coefficients for ky
            // Coefficients for first and second order are stored in different maps.abs
            // Also, for first order we use 
            //      u_y_hat[index] = u_hat[index] * (0.0, I * ky)
            // while second order is
            //      u_y_hat[index] = u_hat[index] * (I * ky)^2
            if(order == 1)
            {
                host :: multiply_map(reinterpret_cast<CuCmplx<T>*>(src.get_tlev_ptr(t_src)),
                                    coeffs_map_d1.get_tlev_ptr(0),
                                    reinterpret_cast<CuCmplx<T>*>(dst.get_tlev_ptr(t_dst)),
                                    [] (CuCmplx<T> val_in, CuCmplx<T> val_map) -> CuCmplx<T>
                                    {return(val_in * CuCmplx<T>(0.0, val_map.im()));},
                                    geom_my21);
                dst.set_transformed(t_dst, true);
            }
            else if(order == 2)
            {
                host :: multiply_map(reinterpret_cast<CuCmplx<T>*>(src.get_tlev_ptr(t_src)),
                                    coeffs_map_d2.get_tlev_ptr(0),
                                    reinterpret_cast<CuCmplx<T>*>(dst.get_tlev_ptr(t_dst)),
                                    [] (CuCmplx<T> val_in, CuCmplx<T> val_map) -> CuCmplx<T>
                                    {return(val_in * val_map.im());},
                                    geom_my21);
                dst.set_transformed(t_dst, true);
            }
            else
            {
                throw not_implemented_error(std::string("Derivatives order > 2 are not implemented\n"));
            }
        }     


        template <typename T>
        void impl_arakawa(const cuda_array_bc_nogp<T, allocator_host>& u,
                        const cuda_array_bc_nogp<T, allocator_host>& v,
                        cuda_array_bc_nogp<T, allocator_host> res,
                        const size_t t_srcu, const size_t t_srcv, 
                        const size_t t_dst, allocator_host<T>)
        {
            std::vector<size_t> col_vals(0);
            std::vector<size_t> row_vals(0);

            // Uses address with direct element access, no interpolation
            host :: arakawa_center(u.get_tlev_ptr(t_srcu), u.get_address_ptr(),
                                v.get_tlev_ptr(t_srcv), v.get_address_ptr(),
                                res.get_tlev_ptr(t_dst),
                                u.get_geom());

            // Arakawa kernel for col 0, n = 0..Nx-1. Call arakawa method that calls interpolator
            // for element access
            col_vals.resize(1);
            col_vals[0] = 0;
            row_vals.resize(u.get_geom().get_nx());
            for(size_t n = 0; n < u.get_geom().get_nx(); n++)
                row_vals[n] = n;

            host :: arakawa_single(u.get_tlev_ptr(t_srcu), u.get_address_ptr(), 
                                v.get_tlev_ptr(t_srcv), v.get_address_ptr(),
                                res.get_tlev_ptr(t_dst),
                                u.get_geom(),
                                row_vals, col_vals);

            //Arakawa kernel for col = My-1, n = 0..Nx-1
            col_vals[0] = u.get_geom().get_my() - 1;
            host :: arakawa_single(u.get_tlev_ptr(t_srcu), u.get_address_ptr(), 
                                v.get_tlev_ptr(t_srcv), v.get_address_ptr(),
                                res.get_tlev_ptr(t_dst),
                                u.get_geom(),
                                row_vals, col_vals);

            // Arakawa kernel for col 0..My-1, row n = 0
            col_vals.resize(u.get_geom().get_my());
            row_vals.resize(1);
            row_vals[0] = 0;
            for(size_t m = 0; m < u.get_geom().get_my(); m++)
                col_vals[m] = m;

            host :: arakawa_single(u.get_tlev_ptr(t_srcu), u.get_address_ptr(), 
                                v.get_tlev_ptr(t_srcv), v.get_address_ptr(),
                                res.get_tlev_ptr(t_dst),
                                u.get_geom(),
                                row_vals, col_vals);
            // Arakawa kernel for col 0..My-1, row n = Nx - 1
            row_vals[0] = u.get_geom().get_nx() - 1;
            host :: arakawa_single(u.get_tlev_ptr(t_srcu), u.get_address_ptr(), 
                                v.get_tlev_ptr(t_srcv), v.get_address_ptr(),
                                res.get_tlev_ptr(t_dst),
                                u.get_geom(),
                                row_vals, col_vals);
        }

        template <typename T>
        void impl_invert_laplace(const cuda_array_bc_nogp<T, allocator_host>& src,
                                cuda_array_bc_nogp<T, allocator_host>& dst,
                                const size_t t_src, const size_t t_dst,
                                cuda_array_bc_nogp<CuCmplx<T>, allocator_host>& diag,
                                cuda_array_bc_nogp<CuCmplx<T>, allocator_host>& diag_u,
                                cuda_array_bc_nogp<CuCmplx<T>, allocator_host>& diag_l,
                                solvers :: elliptic_base_t* ell_solver,
                                allocator_host<T>)
        {
            // Copy input data for solver into dst.
            dst.copy(t_dst, src, t_src);


#ifndef __CUDACC__
// Mask call to ell_solver since we do not link to mkl when compiling in device mode
            ell_solver -> solve(nullptr,
                                reinterpret_cast<CuCmplx<T>*>(dst.get_tlev_ptr(t_dst)),
                                diag_l.get_tlev_ptr(0) + 1, 
                                diag.get_tlev_ptr(0), 
                                diag_u.get_tlev_ptr(0));
#endif //__CUDACC__
            dst.set_transformed(t_dst, true);
        } 
    } // namespace fd


    ////////////////////////////////////////////////////////////////////////////////
    //          Implementation of bispectral derivation methods                   //
    ////////////////////////////////////////////////////////////////////////////////
    namespace bispectral

    {
#ifdef __CUDACC__
        template <typename T>
        void impl_coeffs(cuda_array_bc_nogp<CuCmplx<T>, allocator_device>& coeffs_d1,
                         cuda_array_bc_nogp<CuCmplx<T>, allocator_device>& coeffs_d2,
                         const twodads::slab_layout_t& geom_my21,
                         allocator_device<T>)
        {
            const dim3 block_my21(cuda::blockdim_col, cuda::blockdim_row);
            const dim3 grid_my21((geom_my21.get_my() + cuda::blockdim_col - 1) / cuda::blockdim_col,
                                 (geom_my21.get_nx() + cuda::blockdim_row - 1) / (cuda::blockdim_row));

            device :: kernel_gen_coeffs<<<grid_my21, block_my21>>>(coeffs_d1.get_tlev_ptr(0), coeffs_d2.get_tlev_ptr(0), geom_my21);
            gpuErrchk(cudaPeekAtLastError()); 
        }


        template <typename T>
        void impl_deriv(cuda_array_bc_nogp<T, allocator_device>& src,
                        cuda_array_bc_nogp<T, allocator_device>& dst,
                        const size_t t_src, const size_t t_dst, const direction dir, const size_t order,
                        cuda_array_bc_nogp<twodads::cmplx_t, allocator_device>& coeffs_map_d1,
                        cuda_array_bc_nogp<twodads::cmplx_t, allocator_device>& coeffs_map_d2, 
                        twodads::slab_layout_t geom_my21, allocator_device<T>)
        {
            const dim3 block_my21(cuda::blockdim_col, cuda::blockdim_row);
            const dim3 grid_my21((geom_my21.get_my() + cuda::blockdim_col - 1) / cuda::blockdim_col,
                                 (geom_my21.get_nx() + cuda::blockdim_row - 1) / (cuda::blockdim_row));

            switch(dir)
            {
                case direction::x:
                    if(order == 1)
                    {
                        device :: kernel_multiply_map<<<grid_my21, block_my21>>>(reinterpret_cast<CuCmplx<T>*>(src.get_tlev_ptr(t_src)),
                            coeffs_map_d1.get_tlev_ptr(0),
                            reinterpret_cast<CuCmplx<T>*>(dst.get_tlev_ptr(t_dst)),
                            [] __device__ (CuCmplx<T> val_in, CuCmplx<T> val_map) -> CuCmplx<T>
                            {return(val_in * CuCmplx<T>(0.0, val_map.re()));},
                            geom_my21);
                    }
                    else if(order == 2)
                    {
                        device :: kernel_multiply_map<<<grid_my21, block_my21>>>(reinterpret_cast<CuCmplx<T>*>(src.get_tlev_ptr(t_src)),
                            coeffs_map_d2.get_tlev_ptr(0),
                            reinterpret_cast<CuCmplx<T>*>(dst.get_tlev_ptr(t_dst)),
                            [] __device__ (CuCmplx<T> val_in, CuCmplx<T> val_map) -> CuCmplx<T>
                            {return(val_in * val_map.re());},
                            geom_my21);
                    }
                    else
                    {
                        throw not_implemented_error(std::string("Derivatives order > 2 are not implemented"));
                    }
                    break;

                case direction::y:
                    if(order == 1)
                    {
                        device :: kernel_multiply_map<<<grid_my21, block_my21>>>(reinterpret_cast<CuCmplx<T>*>(src.get_tlev_ptr(t_src)),
                            coeffs_map_d1.get_tlev_ptr(0),
                            reinterpret_cast<CuCmplx<T>*>(dst.get_tlev_ptr(t_dst)),
                            [] __device__ (CuCmplx<T> val_in, CuCmplx<T> val_map) -> CuCmplx<T>
                            {return(val_in * CuCmplx<T>(0.0, val_map.im()));},
                            geom_my21);
                        break;
                    }
                    else if(order == 2)
                    {
                        device :: kernel_multiply_map<<<grid_my21, block_my21>>>(reinterpret_cast<CuCmplx<T>*>(src.get_tlev_ptr(t_src)),
                            coeffs_map_d2.get_tlev_ptr(0),
                            reinterpret_cast<CuCmplx<T>*>(dst.get_tlev_ptr(t_dst)),
                            [] __device__ (CuCmplx<T> val_in, CuCmplx<T> val_map) -> CuCmplx<T>
                            {return(val_in * val_map.im());},
                            geom_my21);
                    }
                    else
                    {
                        throw not_implemented_error(std::string("Derivatives order > 2 are not implemented"));
                    }
                    break;
            }
            gpuErrchk(cudaPeekAtLastError());
        }

#endif // __CUDACC__

        template <typename T>
        void impl_deriv(cuda_array_bc_nogp<T, allocator_host>& src,
                        cuda_array_bc_nogp<T, allocator_host>& dst,
                        const size_t t_src, const size_t t_dst, const direction dir, const size_t order,
                        cuda_array_bc_nogp<twodads::cmplx_t, allocator_host>& coeffs_map_d1,
                        cuda_array_bc_nogp<twodads::cmplx_t, allocator_host>& coeffs_map_d2,
                        const twodads::slab_layout_t geom_my21, allocator_host<T>)
        {
            switch(dir)
            {
                case direction::x:
                    if (order == 1)
                    {
                        host :: multiply_map(reinterpret_cast<CuCmplx<T>*>(src.get_tlev_ptr(t_src)),
                                            coeffs_map_d1.get_tlev_ptr(0),
                                            reinterpret_cast<CuCmplx<T>*>(dst.get_tlev_ptr(t_dst)),
                                            [] (CuCmplx<T> val_in, CuCmplx<T> val_map) -> CuCmplx<T>
                                                {return(val_in * CuCmplx<T>(0.0, val_map.re()));},
                                            geom_my21);
                    }
                    else if (order == 2)
                    {
                        host :: multiply_map(reinterpret_cast<CuCmplx<T>*>(src.get_tlev_ptr(t_src)),
                                            coeffs_map_d2.get_tlev_ptr(0),
                                            reinterpret_cast<CuCmplx<T>*>(dst.get_tlev_ptr(t_dst)),
                                            [] (CuCmplx<T> val_in, CuCmplx<T> val_map) -> CuCmplx<T>
                                                {return(val_in * val_map.re());},
                                            geom_my21);
                    }
                    else
                    {
                        throw not_implemented_error(std::string("Derivatives order > 2 are not implemented"));
                    }
                    break;
                case direction::y:
                    if (order == 1)
                    {
                        host :: multiply_map(reinterpret_cast<CuCmplx<T>*>(src.get_tlev_ptr(t_src)),
                                             coeffs_map_d1.get_tlev_ptr(0),
                                             reinterpret_cast<CuCmplx<T>*>(dst.get_tlev_ptr(t_dst)),
                                             [] (CuCmplx<T> val_in, CuCmplx<T> val_map) -> CuCmplx<T>
                                               {return(val_in * CuCmplx<T>(0.0, val_map.im()));},
                                             geom_my21);
                    }
                    else if (order == 2)  
                    {
                     host :: multiply_map(reinterpret_cast<CuCmplx<T>*>(src.get_tlev_ptr(t_src)),
                                          coeffs_map_d2.get_tlev_ptr(0),
                                          reinterpret_cast<CuCmplx<T>*>(dst.get_tlev_ptr(t_dst)),
                                          [] (CuCmplx<T> val_in, CuCmplx<T> val_map) -> CuCmplx<T>
                                            {return(val_in * val_map.im());},
                                          geom_my21);                        
                    }             
                    else
                    {
                        throw not_implemented_error(std::string("Derivatives order > 2 are not implemented"));
                    }
                    break;
            } // switch(dir)
        } // impl_deriv


        template <typename T>
        void impl_invert_laplace(const cuda_array_bc_nogp<T, allocator_host>& src, 
                                 const cuda_array_bc_nogp<T, allocator_host>& dst, 
                                 const cuda_array_bc_nogp<twodads::cmplx_t, allocator_host>& coeffs_map,
                                 const size_t t_src, const size_t t_dst, 
                                 const twodads::slab_layout_t& geom_my21, allocator_host<T>)
        {
            host :: multiply_map(reinterpret_cast<CuCmplx<T>*>(src.get_tlev_ptr(t_src)),
                                 coeffs_map.get_tlev_ptr(0),
                                 reinterpret_cast<CuCmplx<T>*>(dst.get_tlev_ptr(t_dst)),
                                 [] (CuCmplx<T> val_in, CuCmplx<T> val_map) -> CuCmplx<T>
                                 {
                                     return(val_in /(val_map.re() + val_map.im()));
                                 }, geom_my21);
            // Fix the zero mode
            (dst.get_tlev_ptr(0))[0] = T(0.0);
            (dst.get_tlev_ptr(0))[1] = T(0.0);
        }

    } // End namespace bispectral
} // End namespace detail


///////////////////////////////////////////////////////////////////////////////
//           Interface to derivation and elliptical solvers                  //
///////////////////////////////////////////////////////////////////////////////


template <typename T, template <typename> class allocator>
class deriv_base_t
{
    /**
     .. cpp:namespace-push:: deriv_base_t

    */

    /**
     .. cpp:class:: deriv_base_t

     Defines an interface to derivatives and Laplace solvers.

    */
    public:

    deriv_base_t() {}
    virtual ~deriv_base_t() {}

    /**
     .. cpp:function::  virtual void deriv_base_t :: dx(cuda_array_bc_nogp<T, allocator>& src, cuda_array_bc_nogp<T, allocator>& dst, const size_t t_src, const size_t t_ds, const size_t order) = 0

      :param cuda_array_bc_nogp<T, allocator>& src: Input array
      :param cuda_array_bc_nogp<T, allocator>& dst: Output array
      :param const size_t t_src: Time index for input array
      :param const size_t t_dst: Time index for output array
      :param const size_t order: Order of the derivative. Either 1 or 2.

      Calculates d^(order)/ dx^(order) the derivative of src, stores it in dst.
      Implemented by derived classes.
      
    */
              
    virtual void dx(cuda_array_bc_nogp<T, allocator>&,
                    cuda_array_bc_nogp<T, allocator>&,
                    const size_t, const size_t, const size_t) = 0;

    /**
     .. cpp:function::  virtual void deriv_base_t :: dy (cuda_array_bc_nogp<T, allocator>& src, cuda_array_bc_nogp<T, allocator>& dst, const size_t t_src, const size_t t_dst, const size_t order) = 0
     
      :param cuda_array_bc_nogp<T, allocator>& src: Input array
      :param cuda_array_bc_nogp<T, allocator>& dst: Output array
      :param const size_t t_src: Time index for input array
      :param const size_t t_src: Time index for output array
      :param const size_t order: Order of the derivative, either 1 or 2.


      Calculates d^(order)/ dy^(order) the derivative of src, stores it in dst.
      Implemented by derived classes.

    */
    virtual void dy(cuda_array_bc_nogp<T, allocator>&,
                    cuda_array_bc_nogp<T, allocator>&,
                    const size_t, const size_t, const size_t) = 0;
                      
    // Inverts laplace equation

    /**
     .. cpp:function:: virtual void deriv_base_t :: invert_laplace(cuda_array_bc_nogp<T, allocator>& src, cuda_array_bc_nogp<T, allocator>& dst, const size_t t_src, const size_t t_dst) = 0

      :param cuda_array_bc_nogp<T, allocator>& src: Input array
      :param cuda_array_bc_nogp<T, allocator>& dst: Output array
      :param const size_t t_src: Time index for input array
      :param const size_t t_src: Time index for output array

      Inverts the laplace equation, (d^2/dx^2 + d^2/dy^2) f = g.

    */
    virtual void invert_laplace(cuda_array_bc_nogp<T, allocator>&,
                                cuda_array_bc_nogp<T, allocator>&,
                                const size_t, const size_t) = 0;


    // Computes poisson bracket for the two input fields
    // result = {f, g} = dx(f dy(g)) - dy(f dx(g))

    /**
     .. cpp:function:: virtual void deriv_base_t :: pbracket(const cuda_array_bc_nogp<T, allocator>& f, const cuda_array_bc_nogp<T, allocator>& g, const cuda_array_bc_nogp<T, allocator>& dst, const size_t t_f, const size_t t_g, const_size_t t_dst) = 0 
    
     :param const cuda_array_bc_nogp<T, allocator>& f: Array for f
     :param const cuda_array_bc_nogp<T, allocator>& g: Array for g
     :param const cuda_array_bc_nogp<T, allocator>& dst: Output array 
     :param const size_t t_f: Time index for f array
     :param const size_t t_g: Time index for g array
     :param const size_t t_dst: Time index for output array

     Compute Poisson brackets of the input fields and stores them in dst.
     This method uses the Arakawa scheme.
     dst = {f, g} = d/dx(f d/dy(g)) - d/dy(f d/dx(g))

    */
    virtual void pbracket(const cuda_array_bc_nogp<T, allocator>&,
                          const cuda_array_bc_nogp<T, allocator>&,
                          cuda_array_bc_nogp<T, allocator>&,
                          const size_t, const size_t, const size_t) = 0;

    /**
     .. cpp:function:: virtual void deriv_base_t :: pbracket(const cuda_array_bc_nogp<T, allocator>& f, const cuda_array_bc_nogp<T, allocator>& g, const cuda_array_bc_nogp<T, allocator>& dst, const size_t t_f, const size_t t_g, const_size_t t_dst) = 0 
    
     :param const cuda_array_bc_nogp<T, allocator>& f: Array for f
     :param const cuda_array_bc_nogp<T, allocator>& g: Array for g
     :param const cuda_array_bc_nogp<T, allocator>& dst: Output array 
     :param const size_t t_f: Time index for f array
     :param const size_t t_g: Time index for g array
     :param const size_t t_dst: Time index for output array

     Compute Poisson brackets of the input fields and stores them in dst.
     This method does not use the Arakawa scheme.
     dst = {f, g} = d/dx(f d/dy(g)) - d/dy(f d/dx(g))

    */
    virtual void pbracket(const cuda_array_bc_nogp<T, allocator>&,
                          const cuda_array_bc_nogp<T, allocator>&,
                          const cuda_array_bc_nogp<T, allocator>&,
                          const cuda_array_bc_nogp<T, allocator>&,
                          cuda_array_bc_nogp<T, allocator>&,
                          const size_t, const size_t, const size_t) = 0;

    /**
     .. cpp:namespace-pop::

     */
};


/////////////////////////////////////////////////////////////////////////////////////
//  Implmentation of finite difference, spectral derivation and elliptical solvers //
//  for semi-periodic boundary geometries                                          //
/////////////////////////////////////////////////////////////////////////////////////

template <typename T, template <typename> class allocator>
class deriv_fd_t : public deriv_base_t<T, allocator>
{
    /**
     .. cpp:namespace-push:: deriv_fd_t 

    */

    /**
     .. cpp:class:: deriv_fd_t : public deriv_base_t

     Implements derivative and Laplace members using finite-difference scheme
     in x-direction and spectral methods in y-direction. Does not provide
     an override for pbracket(f_x, f_y, g_x, g_y,...) member.

    */
    public:
        using cmplx_t = CuCmplx<T>;
        using cmplx_arr = cuda_array_bc_nogp<cmplx_t, allocator>;

        #ifdef HOST
        using dft_library_t = fftw_object_t<T>;
        using elliptic_t = solvers :: elliptic_mkl_t;
        #endif //HOST

        #ifdef DEVICE
        using dft_library_t = cufft_object_t<T>;
        using elliptic_t = solvers :: elliptic_cublas_t;
        #endif //DEVICE

        deriv_fd_t(const twodads::slab_layout_t&);    
        ~deriv_fd_t() {delete my_solver;}

        virtual void dx(cuda_array_bc_nogp<T, allocator>& src,
                        cuda_array_bc_nogp<T, allocator>& dst,
                        const size_t t_src, const size_t t_dst, const size_t order)
        {
            assert(src.is_transformed(t_src) == false && "deriv_fd_t :: void dx: src must not be transformed");
            if(order < 3)
                detail :: fd :: impl_dx(src, dst, t_src, t_dst, order, allocator<T>{});
            else
            {
                std::stringstream err_str;
                err_str << __PRETTY_FUNCTION__ << ": order = " << order << "not implemented"; 
                throw(not_implemented_error(err_str.str()));
            } 
        }

        
        virtual void dy(cuda_array_bc_nogp<T, allocator>& src,
                        cuda_array_bc_nogp<T, allocator>& dst,
                        const size_t t_src, const size_t t_dst, const size_t order)
        {
            assert(src.is_transformed(t_src) == true && "deriv_fd_t :: void dy: src must be transformed");

            // Multiply with ky coefficients
            if (order < 3)
                detail :: fd :: impl_dy(src, dst, t_src, t_dst, order, get_coeffs_dy1(), get_coeffs_dy2(), get_geom_my21(), allocator<T>{});
            else
            {
                std::stringstream err_str;
                err_str << __PRETTY_FUNCTION__ << ": order = " << order << "not implemented"; 
                throw(not_implemented_error(err_str.str()));
            } 
        }


        virtual void invert_laplace(cuda_array_bc_nogp<T, allocator>& src,
                                    cuda_array_bc_nogp<T, allocator>& dst,
                                    const size_t t_src, const size_t t_dst)
        {
            assert(src.get_geom() == dst.get_geom() && "deriv_fd_t :: invert_laplace: src and dst need to have the same geometry");
            assert(src.get_geom() == get_geom());
            assert(src.get_bvals() == dst.get_bvals() && "deriv_fd_t :: invert_laplace: src and dst need to have the same boundary values");

            assert(src.is_transformed(t_src) && "deriv_fd_t: void invert_laplace: src must be transformed");
            // When solving Ax=b, update the boundary terms in b
            // The boundary conditions change the ky=0 mode of the n=0/Nx-1 row

            // Note that we take the DFT of the boundary value. For a real boundary
            // value, this is just the value multiplied by the number of Fourier modes.
            // See http://fftw.org/fftw3_doc/The-1d-Real_002ddata-DFT.html#The-1d-Real_002ddata-DFT
            T bval_left_hat{src.get_bvals().get_bv_left() * static_cast<T>(src.get_my())};
            T bval_right_hat{src.get_bvals().get_bv_right() * static_cast<T>(src.get_my())};
            T add_to_boundary_left{0.0};
            T add_to_boundary_right{0.0};
            switch(src.get_bvals().get_bc_left())
            {
                case twodads::bc_t::bc_dirichlet:
                    add_to_boundary_left = -2.0 * bval_left_hat;
                    break;
                case twodads::bc_t::bc_neumann:
                    add_to_boundary_left = -1.0 * src.get_geom().get_deltax() * bval_left_hat;
                    break;
                case twodads::bc_t::bc_periodic:
                    std::cerr << "Periodic boundary conditions not implemented by this class. We shouldn't be here!." << std::endl;
                    break;
                case twodads::bc_t::bc_null:
                    std::cerr << "Null boundary conditions not implemented by this class. We shouldn't be here!'" << std::endl;
                    break;
            }

            switch(src.get_bvals().get_bc_right())
            {
                case twodads::bc_t::bc_dirichlet:
                    add_to_boundary_right = -2.0 * bval_right_hat;
                    break;
                case twodads::bc_t::bc_neumann:
                    add_to_boundary_right = src.get_geom().get_deltax() * bval_right_hat;
                    break;
                case twodads::bc_t::bc_periodic:
                    std::cerr << "Periodic boundary conditions not implemented by this class. We shouldn't be here!." << std::endl;
                    break;
                case twodads::bc_t::bc_null:
                    std::cerr << "Null boundary conditions not implemented by this class. We shouldn't be here!'" << std::endl;
                    break;
            }    

            //Add boundary terms to b before solving Ax=b
            src.apply([=] LAMBDACALLER (T input, const size_t n, const size_t m, twodads::slab_layout_t geom) -> T
            {
            if(n == 0  && m == 0)
                return(input + add_to_boundary_left);
            else if(n == geom.get_nx() - 1 && m == 0)
                return(input + add_to_boundary_right);
            else
                return(input);
            }, t_src);
                
            detail :: fd :: impl_invert_laplace(src, dst, t_src, t_dst,  
                                                get_diag(), get_diag_u(), get_diag_l(),
                                                get_ell_solver(),
                                                allocator<T>{});

            // Remove boundary terms after solving the system
            src.apply([=] LAMBDACALLER (T input, const size_t n, const size_t m, twodads::slab_layout_t geom) -> T
            {
            if(n == 0  && m == 0)
                return(input - add_to_boundary_left);
            else if(n == geom.get_nx() - 1 && m == 0)
                return(input - add_to_boundary_right);
            else
                return(input);
            }, t_src);
        }

        virtual void pbracket(const cuda_array_bc_nogp<T, allocator>& u,
                              const cuda_array_bc_nogp<T, allocator>& v,
                              cuda_array_bc_nogp<T, allocator>& dst,
                              const size_t t_srcu, const size_t t_srcv, const size_t t_dst)
        {
            assert(u.is_transformed(t_srcu) == false);
            assert(v.is_transformed(t_srcv) == false);
            // The Arakawa scheme computes -{f,g} = {g,f}.
            // Swap the position of the input parameters here.
            detail :: fd :: impl_arakawa(v, u, dst, t_srcv, t_srcu, t_dst, allocator<T>{});
        }


        virtual void pbracket(const cuda_array_bc_nogp<T, allocator>& f_x,
                              const cuda_array_bc_nogp<T, allocator>& f_y,
                              const cuda_array_bc_nogp<T, allocator>& g_x,
                              const cuda_array_bc_nogp<T, allocator>& g_y,
                              cuda_array_bc_nogp<T, allocator>& dst,
                              const size_t t_src_f, const size_t t_src_g, const size_t t_dst)
        {
            throw not_implemented_error("This method is not implemented for finite differences\n");
        }


        void init_diagonals();

        cmplx_arr& get_coeffs_dy1() {return(coeffs_dy1);};
        cmplx_arr& get_coeffs_dy2() {return(coeffs_dy2);};
        cmplx_arr& get_diag() {return(diag);};
        cmplx_arr& get_diag_u() {return(diag_u);};
        cmplx_arr& get_diag_l() {return(diag_l);};
        // Layout of the real fields, i.e. Nx * My
        twodads::slab_layout_t get_geom() const {return(geom);};
        // Layout of complex fields, i.e. Nx * My21
        twodads::slab_layout_t get_geom_my21() const {return(geom_my21);};
        // Layouf of the diagonals, i.e. My21 * Nx
        twodads::slab_layout_t get_geom_transpose() const {return(geom_transpose);};

        inline elliptic_t* get_ell_solver() {return(my_solver);};

    private:
        const twodads::slab_layout_t geom;          // Layout for Nx * My arrays
        const twodads::slab_layout_t geom_my21;     // Layout for spectrally transformed NX * My21 arrays
        const twodads::slab_layout_t geom_transpose;     // Transposed complex layout (My21 * Nx) for the tridiagonal solver
        elliptic_t* my_solver;

        // Coefficient storage for spectral derivation
        cmplx_arr coeffs_dy1;
        cmplx_arr coeffs_dy2;
        // Matrix storage for solving tridiagonal equations
        cmplx_arr   diag;
        cmplx_arr   diag_l;
        cmplx_arr   diag_u;

    /**
     .. cpp:namespace-pop::

     */
};


template <typename T, template <typename> class allocator>
deriv_fd_t<T, allocator> :: deriv_fd_t(const twodads::slab_layout_t& _geom) :
    geom{_geom},
    geom_my21{get_geom().get_xleft(), 
              get_geom().get_deltax(), 
              get_geom().get_ylo(), 
              get_geom().get_deltay(), 
              get_geom().get_nx(), get_geom().get_pad_x(),
              (get_geom().get_my() + get_geom().get_pad_y()) / 2, 0, 
              get_geom().get_grid()},
    geom_transpose{get_geom().get_ylo(),
                   get_geom().get_deltay(),
                   get_geom().get_xleft(),
                   get_geom().get_deltax(),
                   (get_geom().get_my() + get_geom().get_pad_y()) / 2, 0,
                   get_geom().get_nx(), 0,
                   get_geom().get_grid()},
    my_solver{new elliptic_t(get_geom())},
    // Very fancy way of initializing a complex Nx * My / 2 + 1 array
    coeffs_dy1{get_geom_my21(), twodads::bvals_t<CuCmplx<T>>(), 1},
    coeffs_dy2{get_geom_my21(), twodads::bvals_t<CuCmplx<T>>(), 1},
    diag{get_geom_transpose(), twodads::bvals_t<CuCmplx<T>>(), 1},
    diag_l{get_geom_transpose(), twodads::bvals_t<CuCmplx<T>>(), 1},
    diag_u{get_geom_transpose(), twodads::bvals_t<CuCmplx<T>>(), 1}
{
    // Initialize the diagonals in a function as CUDA currently doesn't allow to call
    // Lambdas in the constructor.
    utility :: bispectral :: init_deriv_coeffs(get_coeffs_dy1(), get_coeffs_dy2(), get_geom_my21(), allocator<T>{});
    init_diagonals();
}

// Remember that the diagonals are transposed:
// The normal layout has the columns contiguous in memory.
// After a fourier transformation, contiguous elements correspond to different fourier modes.
// The tridiagonal solver however solves one linear system for one fourier mode at a time
// Thus, the diagonals have a layout in memory where contiguous values correspond to a single fourier mode

template <typename T, template <typename> class allocator>
void deriv_fd_t<T, allocator> :: init_diagonals() 
{
    diag.apply([] LAMBDACALLER (CuCmplx<T> dummy, const size_t n, const size_t m, twodads::slab_layout_t geom) -> CuCmplx<T>
    {
        // ky runs with index n (the kernel addressing function, see cuda::thread_idx
        // We are transposed, Lx = dx * (2 * nx - 1) as we have cut nx roughly in half
        const T Lx{geom.get_deltax() * 2 * (geom.get_nx() - 1)};
        const CuCmplx<T> ky2 = twodads::TWOPI * twodads::TWOPI * static_cast<T>(n * n) / (Lx * Lx);
        const CuCmplx<T> inv_dx2{1.0 / (geom.get_deltay() * geom.get_deltay())};
        if(m > 0 && m < geom.get_my() - 1)
        {
            // Use shitty notation ... * (-2.0) because operator* is not a friend of CuCmplx<T>
            return (inv_dx2 * (-2.0) - ky2);
        }
        else if (m == 0)
        {
            return(inv_dx2 * (-3.0) - ky2);
        }
        else if (m == geom.get_my() - 1)
        {
            return(inv_dx2 * (-3.0) - ky2);
        }
        return(-1.0);
    }, 0);

    diag_l.apply([] LAMBDACALLER (CuCmplx<T> dummy, const size_t n, const size_t m, twodads::slab_layout_t geom) -> CuCmplx<T>
    {
        // CUBLAS requires the first element in the lower diagonal to be zero.
        // Remember to shift the pointer in the MKL implementation when passing to the
        // MKL caller routine in solver
        const CuCmplx<T> inv_dx2 = 1.0 / (geom.get_deltax() * geom.get_deltax());
        if(m > 0)
            //return(inv_dx2);
            return(inv_dx2);
        else if(m == 0)
            return(0.0);
        return(-1.0);
    }, 0);

    diag_u.apply([] LAMBDACALLER (CuCmplx<T> dummy, const size_t n, const size_t m, twodads::slab_layout_t geom) -> CuCmplx<T>
    {
        // CUBLAS requires the last element in the upper diagonal to be zero.
        // Remember to shift the pointer in the MKL implementation when passing to the
        // MKL caller routine in solver
        const CuCmplx<T> inv_dx2{1.0 / (geom.get_deltax() * geom.get_deltax())};
        if(m < geom.get_my() - 1)
            return(inv_dx2);
        else if(m == geom.get_my() - 1)
            return(0.0);  
        return(-1.0);  
    }, 0);
}


template <typename T, template <typename> class allocator>
class deriv_spectral_t : public deriv_base_t<T, allocator>
{
    /**
     .. cpp:namespace-push:: deriv_fd_t 

    */

    /**
     .. cpp:class:: deriv_bs_t : public deriv_base_t

     Implements derivative and Laplace members using finite-difference scheme
     in x-direction and spectral methods in y-direction. Does not provide
     an override for pbracket(f_x, f_y, g_x, g_y,...) member.

    */
    public:
        using cmplx_t = CuCmplx<T>;
        using cmplx_arr = cuda_array_bc_nogp<cmplx_t, allocator>;
        using real_arr = cuda_array_bc_nogp<T, allocator>;

        #ifdef HOST
        using dft_library_t = fftw_object_t<T>;
        #endif //HOST

        #ifdef DEVICE
        using dft_library_t = cufft_object_t<T>;
        #endif //DEVICE
   
        deriv_spectral_t(const twodads::slab_layout_t& _geom) :
        geom{_geom},
        // Transposed geometry. Required for transposed arrays passed to the  Matrix solver.
        geom_my21{get_geom().get_xleft(), 
                get_geom().get_deltax(), 
                get_geom().get_ylo(), 
                get_geom().get_deltay(), 
                get_geom().get_nx(), get_geom().get_pad_x(),
                (get_geom().get_my() + 2) / 2, 0, 
                get_geom().get_grid()}, 
                coeffs_d1(get_geom_my21(),
                          twodads::bvals_t<CuCmplx<T>>(twodads::bc_t::bc_periodic, twodads::bc_t::bc_periodic, cmplx_t{0.0}, cmplx_t{0.0}), 
                          1),
                coeffs_d2(get_geom_my21(),
                          twodads::bvals_t<CuCmplx<T>>(twodads::bc_t::bc_periodic, twodads::bc_t::bc_periodic, cmplx_t{0.0}, cmplx_t{0.0}), 
                          1),
                tmp_arr(get_geom(), twodads::bvals_t<T>(twodads::bc_t::bc_dirichlet, twodads::bc_t::bc_dirichlet, 0.0, 0.0), 1)                   
        {
            utility :: bispectral :: init_deriv_coeffs(get_coeffs_d1(), get_coeffs_d2(), get_geom_my21(), allocator<T>{});
            std::cout << "deriv_spectral_t :: deriv_spectral_t: constructed" << std::endl;
        }

        virtual void dx(cuda_array_bc_nogp<T, allocator>& src,
                        cuda_array_bc_nogp<T, allocator>& dst,
                        const size_t t_src, const size_t t_dst, const size_t order)
        {
            assert(src.is_transformed(t_src));
            // Delegate by tag-based dispatching
            detail :: bispectral :: impl_deriv(src, dst, t_src, t_dst, direction::x, order, get_coeffs_d1(), get_coeffs_d2(), get_geom_my21(), allocator<T>{});
            dst.set_transformed(t_dst, true);
        }

        virtual void dy(cuda_array_bc_nogp<T, allocator>& src,
                        cuda_array_bc_nogp<T, allocator>& dst,
                        const size_t t_src, const size_t t_dst, const size_t order)
        {
            assert(src.is_transformed(t_src));
            // Delegate by tag-based dispatching
            detail :: bispectral :: impl_deriv(src, dst, t_src, t_dst, direction::y, order, get_coeffs_d1(), get_coeffs_d2(), get_geom_my21(), allocator<T>{});
            dst.set_transformed(t_dst, true);
        }
  
                        
        virtual void invert_laplace(cuda_array_bc_nogp<T, allocator>& src,
                                    cuda_array_bc_nogp<T, allocator>& dst,
                                    const size_t t_src, const size_t t_dst)
        {
            assert(src.get_geom() == dst.get_geom());
            assert(src.get_geom() == get_geom());
            assert(src.get_bvals() == dst.get_bvals());

            assert(src.is_transformed(t_src));
            // Delegate by tag-based dispatching
            detail :: bispectral :: impl_invert_laplace(src, dst, get_coeffs_d2(), t_src, t_dst, get_geom_my21(), allocator<T>{});
            dst.set_transformed(t_dst, true);
        };   

        virtual void pbracket(const cuda_array_bc_nogp<T, allocator>& f,
                      const cuda_array_bc_nogp<T, allocator>& g,
                      cuda_array_bc_nogp<T, allocator>& dst,
                      const size_t t_src_f, const size_t t_src_g, const size_t t_dst)
        {
            throw not_implemented_error("This method is not implemented for spectral methods");
        }


        // Compute f_x g_y - f_y g_x, store result in dst
        virtual void pbracket(const cuda_array_bc_nogp<T, allocator>& f_x,
                      const cuda_array_bc_nogp<T, allocator>& f_y,
                      const cuda_array_bc_nogp<T, allocator>& g_x,
                      const cuda_array_bc_nogp<T, allocator>& g_y,
                      cuda_array_bc_nogp<T, allocator>& dst,
                      const size_t t_src_f, const size_t t_src_g, const size_t t_dst)
        {
            assert(f_x.get_geom() == f_y.get_geom());
            assert(f_x.get_geom() == g_x.get_geom());
            assert(f_x.get_geom() == g_y.get_geom());
            assert(f_x.get_geom() == dst.get_geom());

            assert(f_x.is_transformed(t_src_f) == false);
            assert(f_y.is_transformed(t_src_f) == false);
            assert(g_x.is_transformed(t_src_g) == false);
            assert(g_y.is_transformed(t_src_f) == false);

            // dst <- f_x
            dst.copy(0, f_x, t_src_f);
            // dst *= g_y
            dst.elementwise([] LAMBDACALLER(twodads::real_t lhs, twodads::real_t rhs) -> twodads::real_t
                            {return(lhs * rhs); }, g_y, 0, t_dst);
            // tmp <- f_y
            tmp_arr.copy(0, f_y, t_src_f);
            // tmp * g_x
            tmp_arr.elementwise([] LAMBDACALLER(twodads::real_t lhs, twodads::real_t rhs) -> twodads::real_t
                                { return(lhs * rhs); }, g_x, 0, t_src_g);
            // dst <- dst - tmp = f_x g_y - f_y g_x
            dst.elementwise([] LAMBDACALLER(twodads::real_t lhs, twodads::real_t rhs) -> twodads::real_t
                            { return(lhs - rhs); }, tmp_arr, 0, t_dst);
        };   


        // Layout of the real fields, i.e. Nx * My
        const twodads::slab_layout_t get_geom() const {return(geom);}
        // Layout of complex fields, i.e. Nx * My21
        const twodads::slab_layout_t get_geom_my21() const {return(geom_my21);}

        cmplx_arr& get_coeffs_d1() {return(coeffs_d1);}
        cmplx_arr& get_coeffs_d2() {return(coeffs_d2);}

    private:
        const twodads::slab_layout_t geom;
        // Transposed geometry. Required for transposed arrays passed to the  Matrix solver.
        const twodads::slab_layout_t geom_my21;

        // Coefficients for spectral derivation
        cmplx_arr coeffs_d1;
        cmplx_arr coeffs_d2;

        // Temporary storage for Poisson brackets
        real_arr tmp_arr;

    /**
     ..cpp:namespace-pop

    */
};

#endif //DERIVATIVES_H
// End of file derivatives.h
