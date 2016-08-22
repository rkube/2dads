/*
 * Interface to derivation functions
 */

#ifndef DERIVATIVES_H
#define DERIVATIVES_H


#include "cuda_array_bc_nogp.h"
#include "cucmplx.h"
#include "error.h"
#include "dft_type.h"
//#include "solvers.h"

#include <cassert>
#include <fstream>

#ifdef __CUDACC__
#include "cuda_types.h"
#include <cusolverSp.h>
#include <cublas_v2.h>

namespace device
{
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


/// T* u is the data pointed to by a cuda_array u, address_u its address object
/// T* u is the data pointed to by a cuda_array v, address_v its address object
/// Assume that u and v have the same geometry
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


// Store coefficients for derivation in Fourier-space 
//    d/dy: Compute spectral y-derivative for frequencies 0 .. My/2 - 1,
//          stored in the columns 0..My/2-1
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
void kernel_gen_coeffs(CuCmplx<T>* kmap_dx1, CuCmplx<T>* kmap_dx2, twodads::slab_layout_t geom)
{
    const size_t col{cuda :: thread_idx :: get_col()};
    const size_t row{cuda :: thread_idx :: get_row()};
    const size_t index{row * (geom.get_my() + geom.get_pad_y()) + col}; 
    const T two_pi_Lx{twodads::TWOPI / geom.get_Lx()};
    const T two_pi_Ly{twodads::TWOPI / (static_cast<T>((geom.get_my() - 1) * 2) * geom.get_deltay())}; 
    //const size_t My{(geom.get_my() - 1) * 2};

    CuCmplx<T> tmp1(0.0, 0.0);
    CuCmplx<T> tmp2(0.0, 0.0);

    if(row < geom.get_nx() / 2)
        tmp1.set_re(two_pi_Lx * T(row));

    else if (row == geom.get_nx() / 2)
        tmp1.set_re(0.0);
    else
        tmp1.set_re(two_pi_Lx * (T(row) - T(geom.get_nx())));

    if(col < geom.get_my() - 1)
    {
        tmp1.set_im(two_pi_Ly * T(col));
        tmp2.set_im(-1.0 * two_pi_Ly * two_pi_Ly * T(col * col));
    }
    else
    {
        tmp2.set_im(-1.0 * two_pi_Ly * two_pi_Ly * T(col * col));
        tmp1.set_im(0.0);
    }

    if(col < geom.get_my() && row < geom.get_nx())
    {
        kmap_dx1[index] = tmp1;
        kmap_dx2[index] = tmp2;
    }
}


// Multiply the input array with the imaginary part of the map. Store result in output
template <typename T, typename O>
__global__
void kernel_multiply_map(CuCmplx<T>* in, CuCmplx<T>* map, CuCmplx<T>* out, O op_func,
                            twodads::slab_layout_t geom)
{
    const size_t col{cuda :: thread_idx :: get_col()};
    const size_t row{cuda :: thread_idx :: get_row()};
    const size_t index{row * (geom.get_my() + geom.get_pad_y()) + col}; 

    if(col < geom.get_my() && row < geom.get_nx())
    {
        out[index] = op_func(in[index], map[index]);
    }
}

}

#endif //__CUDACC__


namespace host
{
    template <typename T, typename O>
    void apply_threepoint_center(T* u, address_t<T>* address_u, T* res, O stencil_func, const twodads::slab_layout_t geom)
    {
        const T inv_dx{1.0 / geom.get_deltax()};
        const T inv_dx2{inv_dx * inv_dx};

        for(size_t n = 0; n < geom.get_nx(); n++)
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
    void apply_threepoint(T* u, address_t<T>* address_u, T* res, O stencil_func, const twodads::slab_layout_t geom,
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
                        T* result, const twodads::slab_layout_t geom)
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
                        T* result, const twodads::slab_layout_t geom,
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
#ifdef __CUDACC__
    template <typename T>
    void impl_init_coeffs(cuda_array_bc_nogp<CuCmplx<T>, allocator_device>& coeffs_dy1,
                          cuda_array_bc_nogp<CuCmplx<T>, allocator_device>& coeffs_dy2,
                          const twodads::slab_layout_t geom_my21,
                          allocator_device<T>)
    {
        const dim3 block_my21(cuda::blockdim_col, cuda::blockdim_row);
        const dim3 grid_my21((geom_my21.get_my() + cuda::blockdim_col - 1) / cuda::blockdim_col,
                             (geom_my21.get_nx() + cuda::blockdim_row - 1) / (cuda::blockdim_row));

        device :: kernel_gen_coeffs<<<grid_my21, block_my21>>>(coeffs_dy1.get_tlev_ptr(0), coeffs_dy2.get_tlev_ptr(0), geom_my21);
        gpuErrchk(cudaPeekAtLastError());    
    }


    template <typename T>
    void impl_dx1(const cuda_array_bc_nogp<T, allocator_device>& in,
                  cuda_array_bc_nogp<T, allocator_device>& out,
                  const size_t t_src, const size_t t_dst, allocator_device<T>)
    {
        static dim3 block_single_row(cuda::blockdim_row, 1);
        static dim3 grid_single_row((in.get_geom().get_nx() + cuda::blockdim_row - 1) / cuda::blockdim_row, 1);

        // Call kernel that accesses elements with get_elem; no wrapping around
        device :: kernel_threepoint_center<<<in.get_grid(), in.get_block()>>>(in.get_tlev_ptr(t_src), in.get_address_2ptr(),
                out.get_tlev_ptr(t_dst), 
                [=] __device__ (T u_left, T u_middle, T u_right, T inv_dx, T inv_dx2) -> T
                {return(0.5 * (u_right - u_left) * inv_dx);},
                out.get_geom());
        gpuErrchk(cudaPeekAtLastError());
        // Call kernel that accesses elements with operator(); interpolates ghost point values
        device :: kernel_threepoint_single_row<<<grid_single_row, block_single_row>>>(in.get_tlev_ptr(t_src), in.get_address_2ptr(),
                out.get_tlev_ptr(t_dst), 
                [=] __device__ (T u_left, T u_middle, T u_right, T inv_dx, T inv_dx2) -> T
                {return(0.5 * (u_right - u_left) * inv_dx);},
                out.get_geom(), 0);
        gpuErrchk(cudaPeekAtLastError());

        // Call kernel that accesses elements with operator(); interpolates ghost point values
        device :: kernel_threepoint_single_row<<<grid_single_row, block_single_row>>>(in.get_tlev_ptr(t_src), in.get_address_2ptr(),
                out.get_tlev_ptr(t_dst), 
                [=] __device__ (T u_left, T u_middle, T u_right, T inv_dx, T inv_dx2) -> T
                {return(0.5 * (u_right - u_left) * inv_dx);},
                out.get_geom(), out.get_geom().get_nx() - 1);
        gpuErrchk(cudaPeekAtLastError());
    }


    template <typename T>
    void impl_dx2(const cuda_array_bc_nogp<T, allocator_device>& in,
                  cuda_array_bc_nogp<T, allocator_device>& out,
                  const size_t t_src, const size_t t_dst, allocator_device<T>)
    {
    static dim3 block_single_row(cuda::blockdim_row, 1);
    static dim3 grid_single_row((in.get_geom().get_nx() + cuda::blockdim_row - 1) / cuda::blockdim_row, 1);

    // Call kernel that accesses elements with get_elem; no wrapping around
    device :: kernel_threepoint_center<<<in.get_grid(), in.get_block()>>>(in.get_tlev_ptr(t_src), in.get_address_2ptr(),
              out.get_tlev_ptr(t_dst), 
              [=] __device__ (T u_left, T u_middle, T u_right, T inv_dx, T inv_dx2) -> T
              {return((u_left + u_right - 2.0 * u_middle) * inv_dx2);},
              out.get_geom());
    gpuErrchk(cudaPeekAtLastError());

    // Call kernel that accesses elements with operator(); interpolates ghost point values
    device :: kernel_threepoint_single_row<<<grid_single_row, block_single_row>>>(in.get_tlev_ptr(t_src), in.get_address_2ptr(),
              out.get_tlev_ptr(t_dst), 
              [=] __device__ (T u_left, T u_middle, T u_right, T inv_dx, T inv_dx2) -> T
              {return((u_left + u_right - 2.0 * u_middle) * inv_dx2);}, 
              out.get_geom(), 0);
    gpuErrchk(cudaPeekAtLastError());

    // Call kernel that accesses elements with operator(); interpolates ghost point values
    device :: kernel_threepoint_single_row<<<grid_single_row, block_single_row>>>(in.get_tlev_ptr(t_src), in.get_address_2ptr(),
              out.get_tlev_ptr(t_dst), 
              [=] __device__ (T u_left, T u_middle, T u_right, T inv_dx, T inv_dx2) -> T
              {return((u_left + u_right - 2.0 * u_middle) * inv_dx2);},
              out.get_geom(), out.get_geom().get_nx() - 1);
    gpuErrchk(cudaPeekAtLastError());
    }


    template <typename T>
    void impl_dy1(const cuda_array_bc_nogp<T, allocator_device>& src,
                  cuda_array_bc_nogp<T, allocator_device>& dst,
                  const size_t t_src, const size_t t_dst, 
                  const cuda_array_bc_nogp<twodads::cmplx_t, allocator_device>& coeffs_dy1, 
                  twodads::slab_layout_t geom_my21, allocator_device<T>)
    {
        const dim3 block_my21(cuda::blockdim_col, cuda::blockdim_row);
        const dim3 grid_my21((geom_my21.get_my() + cuda::blockdim_col - 1) / cuda::blockdim_col,
                             (geom_my21.get_nx() + cuda::blockdim_row - 1) / (cuda::blockdim_row));
        // Multiply with coefficients for ky
        device :: kernel_multiply_map<<<grid_my21, block_my21>>>(reinterpret_cast<CuCmplx<T>*>(src.get_tlev_ptr(t_src)),
                coeffs_dy1.get_tlev_ptr(0), 
                reinterpret_cast<CuCmplx<T>*>(dst.get_tlev_ptr(t_dst)),
                [=] __device__ (CuCmplx<T> val_in, CuCmplx<T> val_map) -> CuCmplx<T>
                {return(val_in * CuCmplx<T>(0.0, val_map.im()));},
                geom_my21);
        gpuErrchk(cudaPeekAtLastError());
    }




    template <typename T>
    void impl_dy2(const cuda_array_bc_nogp<T, allocator_device>& src, 
                  cuda_array_bc_nogp<T, allocator_device>& dst,
                  const size_t t_src, const size_t t_dst,
                  const cuda_array_bc_nogp<twodads::cmplx_t, allocator_device>& coeffs_dy2,
                  twodads::slab_layout_t geom_my21, allocator_device<T>)
    {
        const dim3 block_my21(cuda::blockdim_col, cuda::blockdim_row);
        const dim3 grid_my21((geom_my21.get_my() + cuda::blockdim_col - 1) / cuda::blockdim_col,
                             (geom_my21.get_nx() + cuda::blockdim_row - 1) / (cuda::blockdim_row));

        // Multiply with coefficients for ky
        device :: kernel_multiply_map<<<grid_my21, block_my21>>>(reinterpret_cast<CuCmplx<T>*>(src.get_tlev_ptr(t_src)),
                coeffs_dy2.get_tlev_ptr(0), reinterpret_cast<CuCmplx<T>*>(dst.get_tlev_ptr(t_dst)),
                [=] __device__ (CuCmplx<T> val_in, CuCmplx<T> val_map) -> CuCmplx<T>
                {return(val_in * val_map.im());},
                geom_my21);
        gpuErrchk(cudaPeekAtLastError());

    }


    template <typename T>
    void impl_arakawa(const cuda_array_bc_nogp<T, allocator_device>& u,
                           const cuda_array_bc_nogp<T, allocator_device>& v,
                           cuda_array_bc_nogp<T, allocator_device> res,
                           const size_t t_src, const size_t t_dst, allocator_device<T>)
    {
        // Thread layout for accessing a single row (m = 0..My-1, n = 0, Nx-1)
        static dim3 block_single_row(cuda::blockdim_row, 1);
        static dim3 grid_single_row((u.get_geom().get_nx() + cuda::blockdim_row - 1) / cuda::blockdim_row, 1);

        // Thread layout for accessing a single column (m = 0, My - 1, n = 0...Nx-1)
        static dim3 block_single_col(1, cuda::blockdim_col);
        static dim3 grid_single_col(1, (u.get_geom().get_my() + cuda::blockdim_col - 1) / cuda::blockdim_col);

        device :: kernel_arakawa_center<<<u.get_grid(), u.get_block()>>>(u.get_tlev_ptr(t_src), u.get_address_2ptr(),
                v.get_tlev_ptr(t_src), v.get_address_2ptr(),
                res.get_tlev_ptr(t_dst), u.get_geom());
        gpuErrchk(cudaPeekAtLastError());

        // Create address objects to access ghost points 
        device :: kernel_arakawa_single_row<<<grid_single_row, block_single_row>>>(u.get_tlev_ptr(t_src), u.get_address_2ptr(),
                v.get_tlev_ptr(t_src), v.get_address_2ptr(),
                res.get_tlev_ptr(t_dst), u.get_geom(), 0);
        gpuErrchk(cudaPeekAtLastError());

        device :: kernel_arakawa_single_row<<<grid_single_row, block_single_row>>>(u.get_tlev_ptr(t_src), u.get_address_2ptr(),
                v.get_tlev_ptr(t_src), v.get_address_2ptr(),
                res.get_tlev_ptr(t_dst), u.get_geom(), u.get_geom().get_nx() - 1);
        gpuErrchk(cudaPeekAtLastError());

        device :: kernel_arakawa_single_col<<<grid_single_col, block_single_col>>>(u.get_tlev_ptr(t_src), u.get_address_2ptr(),
                v.get_tlev_ptr(t_src), v.get_address_2ptr(),
                res.get_tlev_ptr(t_dst), u.get_geom(), 0);
        gpuErrchk(cudaPeekAtLastError());

        device :: kernel_arakawa_single_col<<<grid_single_col, block_single_col>>>(u.get_tlev_ptr(t_src), u.get_address_2ptr(),
                v.get_tlev_ptr(t_src), v.get_address_2ptr(),
                res.get_tlev_ptr(t_dst), u.get_geom(), u.get_geom().get_my() - 1);
        gpuErrchk(cudaPeekAtLastError());
    }

    template <typename T>
    void impl_invert_laplace(const cuda_array_bc_nogp<T, allocator_device>& src,
                             cuda_array_bc_nogp<T, allocator_device>& dst,
                             const twodads::bc_t bc_t_left, const T bval_left,
                             const twodads::bc_t bc_t_right, const T bval_right,
                             const size_t t_src, const size_t t_dst,
                             CuCmplx<T>* h_diag,
                             cuda_array_bc_nogp<CuCmplx<T>, allocator_device>& diag,
                             cuda_array_bc_nogp<CuCmplx<T>, allocator_device>& diag_u,
                             cuda_array_bc_nogp<CuCmplx<T>, allocator_device>& diag_l,
                             allocator_device<T>)                         
    {
        const T inv_dx2{1.0 / (src.get_geom().get_deltax() * src.get_geom().get_deltax())};
        const T delta_y{T(src.get_geom().get_my()) / src.get_geom().get_Ly()};

        //if (dst.get_bvals() != src.get_bvals())
        //    throw assert_error(std::string("assert_error: invert_laplace: src and dst must have the same boundary conditions\n"));

        solvers :: elliptic my_ell_solver(src.get_geom());

        //// Solve the tridiagonal system
        //// 1.) Update the main diagonal for ky=0 mode with the boundary values
        ////     Add the Fourier coefficient for mode m=0 of f(0.0, y) = u: hat(u) = My * u
        for(size_t n = 0; n < src.get_geom().get_nx(); n++)
        {
            h_diag[n] = -2.0 * inv_dx2; // -ky2(=0) 
        }
        switch(bc_t_left)
        {
            case twodads::bc_t::bc_dirichlet:
                h_diag[0] = -3.0 * inv_dx2 + 2.0 * bval_left * twodads::TWOPI * delta_y;
                break;
            case twodads::bc_t::bc_neumann:
                h_diag[0] = -3.0 * inv_dx2 - bval_left * twodads::TWOPI * delta_y;
                break;
            case twodads::bc_t::bc_periodic:
                std::cerr << "Periodic boundary conditions not implemented yet." << std::endl;
                std::cerr << "Treating as dirichlet, bval=0" << std::endl;
                break;
        }

        switch(bc_t_right)
        {
            case twodads::bc_t::bc_dirichlet:
                h_diag[src.get_geom().get_nx() - 1] = -3.0 * inv_dx2 + 2.0 * bval_right * twodads::TWOPI * delta_y;
                break;
            case twodads::bc_t::bc_neumann:
                h_diag[src.get_geom().get_nx() - 1] = -3.0 * inv_dx2 - bval_right * twodads::TWOPI * delta_y;
                break;
            case twodads::bc_t::bc_periodic:
                std::cerr << "Periodic boundary conditions not implemented yet." << std::endl;
                std::cerr << "Treating as dirichlet, bval=0" << std::endl;
                break;
        }
        gpuErrchk(cudaMemcpy(diag.get_tlev_ptr(0), h_diag, src.get_geom().get_nx() * sizeof(CuCmplx<T>), cudaMemcpyHostToDevice));


        const size_t My21{(src.get_geom().get_my() + src.get_geom().get_pad_y()) / 2};
        CuCmplx<T>* arr_in = new CuCmplx<T>[src.get_geom().get_nx() * My21];
        CuCmplx<T>* arr_out = new CuCmplx<T>[src.get_geom().get_nx() * My21];

        gpuErrchk(cudaMemcpy(arr_in, src.get_tlev_ptr(t_src), src.get_geom().get_nx() * My21 * sizeof(CuCmplx<T>), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(arr_out, dst.get_tlev_ptr(t_dst), src.get_geom().get_nx() * My21 * sizeof(CuCmplx<T>), cudaMemcpyDeviceToHost));

        //size_t off{1};
        //for(size_t n = 0; n < src.get_geom().get_nx(); n++)
        //{
        //    std::cout << off + n * My21<< ": \t";
        //    std::cout << h_diag[off + n] << "\t";
        //    std::cout << arr_in[off + n * My21] << std::endl;
        //    //std::cout << arr_out[off + n] << std::endl;
        //}

        my_ell_solver.solve(reinterpret_cast<cuDoubleComplex*>(src.get_tlev_ptr(t_src)), 
                            reinterpret_cast<cuDoubleComplex*>(dst.get_tlev_ptr(t_dst)),
                            reinterpret_cast<cuDoubleComplex*>(diag_l.get_tlev_ptr(0)), 
                            reinterpret_cast<cuDoubleComplex*>(diag.get_tlev_ptr(0)), 
                            reinterpret_cast<cuDoubleComplex*>(diag_u.get_tlev_ptr(0)));

        gpuErrchk(cudaMemcpy(arr_out, dst.get_tlev_ptr(t_dst), src.get_geom().get_nx() * My21 * sizeof(CuCmplx<T>), cudaMemcpyDeviceToHost));

        //std::cout << "After:" << std::endl;
        //for(size_t n = 0; n < src.get_geom().get_nx(); n++)
        //{
        //    std::cout << off + n * My21<< ": \t";
        //    std::cout << arr_in[off + m] << "\t";
        //    std::cout << arr_out[off + n * My21] << std::endl;
        //}   

        delete [] arr_in;
        delete [] arr_out;

        dst.set_transformed(true);
    }

#endif //__CUDACC__

    template <typename T>
    void impl_init_coeffs(cuda_array_bc_nogp<CuCmplx<T>, allocator_host>& coeffs_dy1,
                          cuda_array_bc_nogp<CuCmplx<T>, allocator_host>& coeffs_dy2,
                          const twodads::slab_layout_t geom_my21,
                          allocator_host<T>)
    {
        const T two_pi_Lx{twodads::TWOPI / geom_my21.get_Lx()};
        const T two_pi_Ly{twodads::TWOPI / (static_cast<T>((geom_my21.get_my() - 1) * 2) * geom_my21.get_deltay())};

        size_t n{0};
        size_t m{0};
        // Access data in coeffs_dy via T get_elem function below.
        address_t<CuCmplx<T>>* arr_dy1{coeffs_dy1.get_address_ptr()};
        address_t<CuCmplx<T>>* arr_dy2{coeffs_dy2.get_address_ptr()};
  
        CuCmplx<T>* dy1_data = coeffs_dy1.get_tlev_ptr(0);
        CuCmplx<T>* dy2_data = coeffs_dy2.get_tlev_ptr(0);
        
        /////////////////////////////////////////////////////////////////////////////////////////////
        // n = 0..nx/2-1
        for(n = 0; n < geom_my21.get_nx() / 2; n++)
        {
            for(m = 0; m < geom_my21.get_my() - 1; m++)
            {
                (*arr_dy1).get_elem(dy1_data, n, m).set_re(two_pi_Lx * T(n)); 
                (*arr_dy1).get_elem(dy1_data, n, m).set_im(two_pi_Ly * T(m));

                (*arr_dy2).get_elem(dy2_data, n, m).set_re(T(0.0));
                (*arr_dy2).get_elem(dy2_data, n, m).set_im(-1.0 * two_pi_Ly * two_pi_Ly * T(m * m));
            }
            m = geom_my21.get_my() - 1;
            (*arr_dy1).get_elem(dy1_data, n, m).set_re(two_pi_Lx * T(n));
            (*arr_dy1).get_elem(dy1_data, n, m).set_im(T(0.0));

            (*arr_dy2).get_elem(dy2_data, n, m).set_re(T(0.0));
            (*arr_dy2).get_elem(dy2_data, n, m).set_im(-1.0 * two_pi_Ly * two_pi_Ly * T(m * m));
        }
        
        /////////////////////////////////////////////////////////////////////////////////////////////
        // n = nx/2
        n = geom_my21.get_nx() / 2;
        for(m = 0; m < geom_my21.get_my() - 1; m++)
        {
            (*arr_dy1).get_elem(dy1_data, n, m).set_re(T(0.0)); 
            (*arr_dy1).get_elem(dy1_data, n, m).set_im(two_pi_Ly * T(m));

            (*arr_dy2).get_elem(dy2_data, n, m).set_re(T(0.0));
            (*arr_dy2).get_elem(dy2_data, n, m).set_im(-1.0 * two_pi_Ly * two_pi_Ly * T(m * m));
        }
        m = geom_my21.get_my() - 1;

        (*arr_dy1).get_elem(dy1_data, n, m).set_re(T(0.0));
        (*arr_dy1).get_elem(dy1_data, n, m).set_im(T(0.0));

        (*arr_dy2).get_elem(dy2_data, n, m).set_re(T(0.0));
        (*arr_dy2).get_elem(dy2_data, n, m).set_im(-1.0 * two_pi_Ly * two_pi_Ly * T(m * m));

        /////////////////////////////////////////////////////////////////////////////////////////////
        // n = nx/2+1 .. Nx-2
        for(n = geom_my21.get_nx() / 2 + 1; n < geom_my21.get_nx(); n++)
        {
            for(m = 0; m < geom_my21.get_my() - 1; m++)
            {
                (*arr_dy1).get_elem(dy1_data, n, m).set_re(two_pi_Lx * (T(n) - T(geom_my21.get_nx())));
                (*arr_dy1).get_elem(dy1_data, n, m).set_im(two_pi_Ly * T(m));

                (*arr_dy2).get_elem(dy2_data, n, m).set_re(T(0.0));
                (*arr_dy2).get_elem(dy2_data, n, m).set_im(-1.0 * two_pi_Ly * two_pi_Ly * T(m * m));
            }
            
            m = geom_my21.get_my() - 1;
            (*arr_dy1).get_elem(dy1_data, n, m).set_re(two_pi_Lx * (T(n) - T(geom_my21.get_nx())));
            (*arr_dy1).get_elem(dy1_data, n, m).set_im(T(0.0));

            (*arr_dy2).get_elem(dy2_data, n, m).set_re(T(0.0));
            (*arr_dy2).get_elem(dy2_data, n, m).set_im(-1.0 * two_pi_Ly * two_pi_Ly * T(m * m));
        }
    }


    template <typename T>
    void impl_dx1(const cuda_array_bc_nogp<T, allocator_host>& in,
                  cuda_array_bc_nogp<T, allocator_host>& out,
                  const size_t t_src, const size_t t_dst, allocator_host<T>)
    {
        std::vector<size_t> col_vals(in.get_geom().get_my());
        std::vector<size_t> row_vals(1);

        row_vals[0] = 0;
        for(size_t m = 0; m < in.get_geom().get_my(); m++)
            col_vals[m] = m;

        // Apply threepoint stencil in interior domain, no interpolation here
        host :: apply_threepoint_center(in.get_tlev_ptr(t_src), in.get_address_ptr(), out.get_tlev_ptr(t_dst), 
                                        [] (T u_left, T u_middle, T u_right, T inv_dx, T inv_dx2) -> T
                                        {return(0.5 * (u_right - u_left) * inv_dx);}, 
                                        out.get_geom());

        // Call expensive interpolation routine only for 2 columns
        std::cout << "dx1(host): apply_threepoint" << std::endl;
        host :: apply_threepoint(in.get_tlev_ptr(t_src), in.get_address_ptr(), out.get_tlev_ptr(t_dst), 
                                 [] (T u_left, T u_middle, T u_right, T inv_dx, T inv_dx2) -> T
                                 {return(0.5 * (u_right - u_left) * inv_dx);},
                                 out.get_geom(), row_vals, col_vals);

        row_vals[0] = in.get_geom().get_nx() - 1;
        std::cout << "dx1(host): apply_threepoint" << std::endl;
        host :: apply_threepoint(in.get_tlev_ptr(t_src), in.get_address_ptr(), out.get_tlev_ptr(t_dst), 
                                 [] (T u_left, T u_middle, T u_right, T inv_dx, T inv_dx2) -> T
                                 {return(0.5 * (u_right - u_left) * inv_dx);},
                                 out.get_geom(), row_vals, col_vals);
    }


   

    template <typename T>
    void impl_dx2(const cuda_array_bc_nogp<T, allocator_host>& in,
                  cuda_array_bc_nogp<T, allocator_host>& out,
                  const size_t t_src, const size_t t_dst, allocator_host<T>)
    {
        std::vector<size_t> col_vals(in.get_geom().get_my());
        std::vector<size_t> row_vals(1);

        row_vals[0] = 0;
        for(size_t m = 0; m < in.get_geom().get_my(); m++)
            col_vals[m] = m;

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


    template <typename T>
    void impl_dy1(const cuda_array_bc_nogp<T, allocator_host>& src,
                  cuda_array_bc_nogp<T, allocator_host>& dst,
                  const size_t t_src, const size_t t_dst, 
                  const cuda_array_bc_nogp<twodads::cmplx_t, allocator_host>& coeffs_dy1, 
                  twodads::slab_layout_t geom_my21, allocator_host<T>)
    {
        host :: multiply_map(reinterpret_cast<CuCmplx<T>*>(src.get_tlev_ptr(t_src)),
                             coeffs_dy1.get_tlev_ptr(0),
                             reinterpret_cast<CuCmplx<T>*>(dst.get_tlev_ptr(t_dst)),
                             [=] (CuCmplx<T> val_in, CuCmplx<T> val_map) -> CuCmplx<T>
                             {return(val_in * CuCmplx<T>(0.0, val_map.im()));},
                             geom_my21); 
    }


    template <typename T>
    void impl_dy2(const cuda_array_bc_nogp<T, allocator_host>& src,
                  cuda_array_bc_nogp<T, allocator_host>& dst,
                  const size_t t_src, const size_t t_dst, 
                  const cuda_array_bc_nogp<twodads::cmplx_t, allocator_host>& coeffs_dy2, 
                  twodads::slab_layout_t geom_my21, allocator_host<T>)
    {
        host :: multiply_map(reinterpret_cast<CuCmplx<T>*>(src.get_tlev_ptr(t_src)),
                             coeffs_dy2.get_tlev_ptr(0),
                             reinterpret_cast<CuCmplx<T>*>(dst.get_tlev_ptr(t_dst)),
                             [=] (CuCmplx<T> val_in, CuCmplx<T> val_map) -> CuCmplx<T>
                             {return(val_in * val_map.im());},
                             geom_my21);
        
    }


    template <typename T>
    void impl_arakawa(const cuda_array_bc_nogp<T, allocator_host>& u,
            const cuda_array_bc_nogp<T, allocator_host>& v,
            cuda_array_bc_nogp<T, allocator_host> res,
            const size_t t_src, const size_t t_dst, allocator_host<T>)
    {
        std::vector<size_t> col_vals(0);
        std::vector<size_t> row_vals(0);

        // Uses address with direct element access avoiding ifs etc.
        host :: arakawa_center(u.get_tlev_ptr(t_src), u.get_address_ptr(),
                               v.get_tlev_ptr(t_src), v.get_address_ptr(),
                               res.get_tlev_ptr(t_dst),
                               u.get_geom());

        // Arakawa kernel for col 0, n = 0..Nx-1
        col_vals.resize(1);
        col_vals[0] = 0;
        row_vals.resize(u.get_geom().get_nx());
        for(size_t n = 0; n < u.get_geom().get_nx(); n++)
            row_vals[n] = n;
        host :: arakawa_single(u.get_tlev_ptr(t_src), u.get_address_ptr(), 
                               v.get_tlev_ptr(t_src), v.get_address_ptr(),
                               res.get_tlev_ptr(t_src),
                               u.get_geom(),
                               row_vals, col_vals);

        //Arakawa kernel for col = My-1, n = 0..Nx-1
        col_vals[0] = u.get_geom().get_my() - 1;
        host :: arakawa_single(u.get_tlev_ptr(t_src), u.get_address_ptr(), 
                               v.get_tlev_ptr(t_src), v.get_address_ptr(),
                               res.get_tlev_ptr(t_src),
                               u.get_geom(),
                               row_vals, col_vals);

        // Arakawa kernel for col 0..My-1, row n = 0
        col_vals.resize(u.get_geom().get_my());
        row_vals.resize(1);
        row_vals[0] = 0;
        for(size_t m = 0; m < u.get_geom().get_my(); m++)
            col_vals[m] = m;
        host :: arakawa_single(u.get_tlev_ptr(t_src), u.get_address_ptr(), 
                               v.get_tlev_ptr(t_src), v.get_address_ptr(),
                               res.get_tlev_ptr(t_src),
                               u.get_geom(),
                               row_vals, col_vals);
        // Arakawa kernel for col 0..My-1, row n = Nx - 1
        row_vals[0] = u.get_geom().get_nx() - 1;
        host :: arakawa_single(u.get_tlev_ptr(t_src), u.get_address_ptr(), 
                               v.get_tlev_ptr(t_src), v.get_address_ptr(),
                               res.get_tlev_ptr(t_src),
                               u.get_geom(),
                               row_vals, col_vals);
    }

    template <typename T>
    void impl_invert_laplace(const cuda_array_bc_nogp<T, allocator_host>& src,
                             cuda_array_bc_nogp<T, allocator_host>& dst,
                             const twodads::bc_t bc_t_left, const T bval_left,
                             const twodads::bc_t bc_t_right, const T bval_right,
                             const size_t t_src, const size_t t_dst,
                             CuCmplx<T>* h_diag,
                             cuda_array_bc_nogp<CuCmplx<T>, allocator_host>& diag,
                             cuda_array_bc_nogp<CuCmplx<T>, allocator_host>& diag_u,
                             cuda_array_bc_nogp<CuCmplx<T>, allocator_host>& diag_l,
                             allocator_host<T>)
    {
        const T inv_dx2{1.0 / (src.get_geom().get_deltax() * src.get_geom().get_deltax())};
        const T delta_y{T(src.get_geom().get_my()) / src.get_geom().get_Ly()};

        //if (dst.get_bvals() != src.get_bvals())
        //    throw assert_error(std::string("assert_error: invert_laplace: src and dst must have the same boundary conditions\n"));

        solvers :: elliptic my_ell_solver(src.get_geom());

        //// Solve the tridiagonal system
        //// 1.) Update the main diagonal for ky=0 mode with the boundary values
        ////     Add the Fourier coefficient for mode m=0 of f(0.0, y) = u: hat(u) = My * u
        CuCmplx<T>* hd = diag.get_tlev_ptr(0);
        for(size_t n = 0; n < src.get_geom().get_nx(); n++)
        {
            hd[n] = -2.0 * inv_dx2; // -ky2(=0) 
        }
        switch(bc_t_left)
        {
            case twodads::bc_t::bc_dirichlet:
                hd[0] = -3.0 * inv_dx2 + 2.0 * bval_left * twodads::TWOPI * delta_y;
                break;
            case twodads::bc_t::bc_neumann:
                hd[0] = -3.0 * inv_dx2 - bval_left * twodads::TWOPI * delta_y;
                break;
            case twodads::bc_t::bc_periodic:
                std::cerr << "Periodic boundary conditions not implemented yet." << std::endl;
                std::cerr << "Treating as dirichlet, bval=0" << std::endl;
                break;
        }

        switch(bc_t_right)
        {
            case twodads::bc_t::bc_dirichlet:
                hd[src.get_geom().get_nx() - 1] = -3.0 * inv_dx2 + 2.0 * bval_right * twodads::TWOPI * delta_y;
                break;
            case twodads::bc_t::bc_neumann:
                hd[src.get_geom().get_nx() - 1] = -3.0 * inv_dx2 - bval_right * twodads::TWOPI * delta_y;
                break;
            case twodads::bc_t::bc_periodic:
                std::cerr << "Periodic boundary conditions not implemented yet." << std::endl;
                std::cerr << "Treating as dirichlet, bval=0" << std::endl;
                break;
        } 

        // Copy input data for solver into dst.
        dst.copy(t_dst, src, t_src);

        //size_t My21 {(src.get_geom().get_my() + src.get_geom().get_pad_y()) / 2}; 
        //size_t off{1};
        
        //CuCmplx<T>* ep = reinterpret_cast<CuCmplx<T>*>(dst.get_tlev_ptr(t_src));
        //CuCmplx<T>* dp = reinterpret_cast<CuCmplx<T>*>(diag.get_tlev_ptr(0));
        //CuCmplx<T>* dp_u = reinterpret_cast<CuCmplx<T>*>(diag_u.get_tlev_ptr(0));
        //CuCmplx<T>* dp_u = reinterpret_cast<CuCmplx<T>*>(diag_u.get_tlev_ptr(0));
        //
        //for(size_t n = 0; n < src.get_geom().get_nx(); n++)
        //{
            //std::cout << off + n * My21 << ": \t"; 
            //std::cout << dp[off * src.get_geom().get_nx() + m] << "\t";
            //std::cout << dp_u[off * src.get_geom().get_nx() + m] << "\t";
            //std::cout << dp_l[off * src.get_geom().get_nx() + m] << "\t";
//
            //std::cout << ep[off + n * My21] << std::endl;
        ////    //std::cout << (dst.get_tlev_ptr(t_dst))[2 * off + 2 * m] << ", " << (dst.get_tlev_ptr(t_dst))[2 * off + 2 * m + 1] << "\n";
        //}

#ifndef __CUDACC__
        my_ell_solver.solve(nullptr,
                            reinterpret_cast<lapack_complex_double*>(dst.get_tlev_ptr(t_dst)),
                            reinterpret_cast<lapack_complex_double*>(diag_l.get_tlev_ptr(0)) + 1, 
                            reinterpret_cast<lapack_complex_double*>(diag.get_tlev_ptr(0)), 
                            reinterpret_cast<lapack_complex_double*>(diag_u.get_tlev_ptr(0)));
#endif //__CUDACC__
        
        //std::cout << "After:" << std::endl;
        //for(size_t n = 0; n < src.get_geom().get_nx(); n++)
        //{
            //std::cout << n + off << ": \t";
            //std::cout << dp[off + n * My21] << std::endl; 
        //}                     
        
        dst.set_transformed(true);
    }    
}


/*
 * Datatype that provides derivation routines and elliptic solver
 */
template <typename T, template <typename> class allocator>
class deriv_t
{
    public:
        using cmplx_t = CuCmplx<T>;
        using cmplx_arr = cuda_array_bc_nogp<cmplx_t, allocator>;

        #ifdef HOST
        using dft_library_t = fftw_object_t<T>;
        #endif //HOST

        #ifdef DEVICE
        using dft_library_t = cufft_object_t<T>;
        #endif //DEVICE

        deriv_t(const twodads::slab_layout_t);    
        ~deriv_t()
        {
            delete [] h_diag;   
        };

        void dx_1(const cuda_array_bc_nogp<T, allocator>& src,
                  cuda_array_bc_nogp<T, allocator>& dst,
                  const size_t t_src, const size_t t_dst)
        {
            detail :: impl_dx1(src, dst, t_src, t_dst, allocator<T>{}); 
        }

        void dx_2(const cuda_array_bc_nogp<T, allocator>& src,
                  cuda_array_bc_nogp<T, allocator>& dst,
                  const size_t t_src, const size_t t_dst)
        {
            detail :: impl_dx2(src, dst, t_src, t_dst, allocator<T>{});
        }

        void dy_1(cuda_array_bc_nogp<T, allocator>& src,
                  cuda_array_bc_nogp<T, allocator>& dst,
                  const size_t t_src, const size_t t_dst)
        {
            // DFT r2c
            if(!(src.is_transformed()))
            {
                myfft -> dft_r2c(src.get_tlev_ptr(t_src), reinterpret_cast<CuCmplx<T>*>(src.get_tlev_ptr(t_src)));
                src.set_transformed(true);
            }
            // Multiply with ky coefficients
            detail :: impl_dy1(src, dst, t_src, t_dst, get_coeffs_dy1(), get_geom_my21(), allocator<T>{});

            // DFT c2r and normalize
            myfft -> dft_c2r(reinterpret_cast<CuCmplx<T>*>(dst.get_tlev_ptr(t_dst)), dst.get_tlev_ptr(t_dst));
            dst.set_transformed(false);
            utility :: normalize(dst, t_dst);

            myfft -> dft_c2r(reinterpret_cast<CuCmplx<T>*>(src.get_tlev_ptr(t_src)), src.get_tlev_ptr(t_src));
            src.set_transformed(false);
            utility :: normalize(src, t_src);
        }

        void dy_2(cuda_array_bc_nogp<T, allocator>& src,
                  cuda_array_bc_nogp<T, allocator>& dst,
                  const size_t t_src, const size_t t_dst)
        {
            // DFT r2c
            if(!(src.is_transformed()))
            {
                myfft -> dft_r2c(src.get_tlev_ptr(t_src), reinterpret_cast<CuCmplx<T>*>(src.get_tlev_ptr(t_src)));
                src.set_transformed(true);
            }
            // Multiply with ky^2 coefficients
            detail :: impl_dy2(src, dst, t_src, t_dst, get_coeffs_dy2(), get_geom_my21(), allocator<T>{});

            // DFT c2r and normalize
            myfft -> dft_c2r(reinterpret_cast<CuCmplx<T>*>(dst.get_tlev_ptr(t_dst)), dst.get_tlev_ptr(t_dst));
            dst.set_transformed(false);
            utility :: normalize(dst, t_dst);

            myfft -> dft_c2r(reinterpret_cast<CuCmplx<T>*>(src.get_tlev_ptr(t_src)), src.get_tlev_ptr(t_src));
            src.set_transformed(false);
            utility :: normalize(src, t_src);            
        }

        void invert_laplace(cuda_array_bc_nogp<T, allocator>& src,
                            cuda_array_bc_nogp<T, allocator>& dst,
                            const twodads::bc_t bc_t_left, const T bv_left,
                            const twodads::bc_t bc_t_right, const T bv_right,
                            const size_t t_src, const size_t t_dst)
        {
            if(!(src.is_transformed()))
            {
                myfft -> dft_r2c(src.get_tlev_ptr(t_src), reinterpret_cast<CuCmplx<T>*>(src.get_tlev_ptr(t_src)));
                src.set_transformed(true);
            }
            detail :: impl_invert_laplace(src, dst, bc_t_left, bv_left, bc_t_right, bv_right, 
                                          t_src, t_dst, 
                                          get_hdiag(), get_diag(), get_diag_u(), get_diag_l(),
                                          allocator<T>{});

            myfft -> dft_c2r(reinterpret_cast<CuCmplx<T>*>(src.get_tlev_ptr(t_src)), src.get_tlev_ptr(t_src));
            src.set_transformed(false);
            utility :: normalize(src, t_src); 

            myfft -> dft_c2r(reinterpret_cast<CuCmplx<T>*>(dst.get_tlev_ptr(t_dst)), dst.get_tlev_ptr(t_dst));
            dst.set_transformed(false);
            utility :: normalize(dst, t_dst);
        }

        void arakawa(const cuda_array_bc_nogp<T, allocator>& u,
                     const cuda_array_bc_nogp<T, allocator>& v,
                     cuda_array_bc_nogp<T, allocator>& dst,
                     const size_t t_src, const size_t t_dst)
        {
            detail :: impl_arakawa(u, v, dst, t_src, t_dst, allocator<T>{});
        }

        void init_diagonals();

        cmplx_arr& get_coeffs_dy1() {return(coeffs_dy1);};
        cmplx_arr& get_coeffs_dy2() {return(coeffs_dy2);};
        cmplx_arr& get_diag() {return(diag);};
        cmplx_arr& get_diag_u() {return(diag_u);};
        cmplx_arr& get_diag_l() {return(diag_l);};
        cmplx_t* get_hdiag() {return(h_diag);};
        // Layout of the real fields, i.e. Nx * My
        twodads::slab_layout_t get_geom() const {return(geom);};
        // Layout of complex fields, i.e. Nx * My21
        twodads::slab_layout_t get_geom_my21() const {return(geom_my21);};
        // Layouf of the diagonals, i.e. My21 * Nx
        twodads::slab_layout_t get_geom_transpose() const {return(geom_transpose);};

    private:
        const twodads::slab_layout_t geom;          // Layout for Nx * My arrays
        const twodads::slab_layout_t geom_my21;     // Layout for spectrally transformed NX * My21 arrays
        const twodads::slab_layout_t geom_transpose;     // Transposed complex layout (My21 * Nx) for the tridiagonal solver
        dft_object_t<twodads::real_t>* myfft;

        // Coefficient storage for spectral derivation
        cmplx_arr coeffs_dy1;
        cmplx_arr coeffs_dy2;
        // Matrix storage for solving tridiagonal equations
        cmplx_arr   diag;
        cmplx_arr   diag_l;
        cmplx_arr   diag_u;
        cmplx_t* h_diag;     // Main diagonal, host copy. This one is updated with the boundary conditions passed to invert_laplace routine.
};


template <typename T, template <typename> class allocator>
deriv_t<T, allocator> :: deriv_t(const twodads::slab_layout_t _geom) :
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
    myfft{new dft_library_t(get_geom(), twodads::dft_t::dft_1d)},
    // Very fancy way of initializing a complex Nx * My / 2 + 1 array
    coeffs_dy1{get_geom_my21(), 
               twodads::bvals_t<CuCmplx<T>>(twodads::bc_t::bc_dirichlet, twodads::bc_t::bc_dirichlet, twodads::bc_t::bc_periodic, twodads::bc_t::bc_periodic, cmplx_t{0.0}, cmplx_t{0.0}, cmplx_t{0.0}, cmplx_t{0.0}),
               1},
    // Very fancy way of initializing a complex Nx * My / 2 + 1 array
    coeffs_dy2{get_geom_my21(),
               twodads::bvals_t<CuCmplx<T>>(twodads::bc_t::bc_dirichlet, twodads::bc_t::bc_dirichlet, twodads::bc_t::bc_periodic, twodads::bc_t::bc_periodic, cmplx_t{0.0}, cmplx_t{0.0}, cmplx_t{0.0}, cmplx_t{0.0}), 
               1},
    // Very fancy way of initializing a complex Nx * My / 2 + 1 array
    diag(//get_geom_my21(), 
         get_geom_transpose(), 
         twodads::bvals_t<CuCmplx<T>>(twodads::bc_t::bc_dirichlet, twodads::bc_t::bc_dirichlet, twodads::bc_t::bc_periodic, twodads::bc_t::bc_periodic, cmplx_t{0.0}, cmplx_t{0.0}, cmplx_t{0.0}, cmplx_t{0.0}), 
         1),
    // Very fancy way of initializing a complex Nx * My / 2 + 1 array
    diag_l(//get_geom_my21(), 
           get_geom_transpose(), 
           twodads::bvals_t<CuCmplx<T>>(twodads::bc_t::bc_dirichlet, twodads::bc_t::bc_dirichlet, twodads::bc_t::bc_periodic, twodads::bc_t::bc_periodic, cmplx_t{0.0}, cmplx_t{0.0}, cmplx_t{0.0}, cmplx_t{0.0}), 1),
    // Very fancy way of initializing a complex Nx * My / 2 + 1 array
    diag_u(//get_geom_my21(), 
           get_geom_transpose(), 
           twodads::bvals_t<CuCmplx<T>>(twodads::bc_t::bc_dirichlet, twodads::bc_t::bc_dirichlet, twodads::bc_t::bc_periodic, twodads::bc_t::bc_periodic, cmplx_t{0.0}, cmplx_t{0.0}, cmplx_t{0.0}, cmplx_t{0.0}), 1),
    h_diag{new cmplx_t[get_geom().get_nx()]} 
{
    // Initialize the diagonals in a function as CUDA currently doesn't allow to call
    // Lambdas in the constructor.

    //detail :: impl_init_coeffs(get_coeffs_dy1(), get_coeffs_dy2(), get_geom_my21(), allocator<T>{});
    init_diagonals();
    //utility :: print(get_diag(), 0, std::cout);
    //std::cout << std::endl;
    //utility :: print(get_diag_l(), 0, std::cout);
    //std::cout << std::endl;
    //utility :: print(get_diag_u(), 0, std::cout);
}

// Remember that the diagonals are transposed:
// The normal layout has the columns contiguous in memory.
// After a fourier transformation, contiguous elements correspond to different fourier modes.
// The tridiagonal solver however solves one linear system for one fourier mode at a time
// Thus, the diagonals have a layout in memory where contiguous values correspond to a single fourier mode

template <typename T, template <typename> class allocator>
void deriv_t<T, allocator> :: init_diagonals() 
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

#endif //DERIVATIVES_H
///////////////////////////////// Only commented code below /////////////////////////////////


//template <typename allocator>
//class derivs
//{
//    public:
//        using value_t = typename my_allocator_traits<allocator> :: value_type;
//        using cmplx_t = CuCmplx<value_t>;
//
//        derivs(const twodads::slab_layout_t);
//        ~derivs();
//
//        // Compute first derivative in x-direction
//        void dx_1(const cuda_array_bc_nogp<allocator>&, cuda_array_bc_nogp<allocator>&, 
//                  const size_t, const size_t);
//
//        // Compute second derivative in x-direction
//        void dx_2(const cuda_array_bc_nogp<allocator>&, cuda_array_bc_nogp<allocator>&, 
//                  const size_t, const size_t);
//
//        // Compute first derivative in y-direction
//        void dy_1(cuda_array_bc_nogp<allocator>&, cuda_array_bc_nogp<allocator>&,
//                  const size_t, const size_t);
//
//        // Compute second derivative in y-direction
//        void dy_2(cud1177Ga_array_bc_nogp<allocator>&, cuda_array_bc_nogp<allocator>&,
//                  const size_t, const size_t);
//
//        // Invert laplace equation
//        void invert_laplace(cuda_array_bc_nogp<allocator>&, cuda_array_bc_nogp<allocator>&, 
//                            const twodads::bc_t, const value_t,
//                            const twodads::bc_t, const value_t,
//                            const size_t, const size_t);
//       
//        // Compute arakawa bracket 
//        void arakawa(const cuda_array_bc_nogp<allocator>&, 
//                     const cuda_array_bc_nogp<allocator>&, 
//                     cuda_array_bc_nogp<allocator>&, 
//                     const size_t, const size_t);
//   
//
//        // Initialize DFT, 1d-periodic, column-major 
//        cmplx_t* get_coeffs_dy1() const {return(d_coeffs_dy1);};
//        cmplx_t* get_coeffs_dy2() const {return(d_coeffs_dy2);};
//        twodads::slab_layout_t get_geom() const {return(geom);};
//
//    private:
//        const size_t My21;
//        const twodads::slab_layout_t geom;
//        dft_object_t<twodads::real_t> myfft;
//
//        // Coefficient storage for spectral derivation
//        cmplx_t* d_coeffs_dy1;
//        cmplx_t* d_coeffs_dy2;
//        // Matrix storage for solving tridiagonal equations
//        cmplx_t* d_diag;     // Main diagonal
//        cmplx_t* d_diag_l;   // lower diagonal
//        cmplx_t* d_diag_u;   // upper diagonal, for Laplace equation this is the same as the lower diagonal
//        cmplx_t* h_diag;     // Main diagonal, host copy. This one is updated with the boundary conditions passed to invert_laplace routine.
//};



//template <typename allocator>
//derivs<allocator> :: derivs(const twodads::slab_layout_t _geom) : 
//    My21(static_cast<int>(_geom.get_my()) / 2 + 1), 
//    geom(_geom),
//    myfft(geom, twodads::dft_t::dft_1d),
//    d_coeffs_dy1{nullptr}, d_coeffs_dy2{nullptr},
//    d_diag{nullptr}, d_diag_l{nullptr}, d_diag_u{nullptr},
//    h_diag{nullptr}
//{
//    cout << "Allocating " <<  get_geom().get_nx() * My21 * sizeof(CuCmplx<value_t>) << " bytes" << endl;
//    gpuErrchk(cudaMalloc((void**) &d_coeffs_dy1, get_geom().get_nx() * My21 * sizeof(CuCmplx<value_t>)))
//    gpuErrchk(cudaMalloc((void**) &d_coeffs_dy2, get_geom().get_nx() * My21 * sizeof(CuCmplx<value_t>)))
//
//    const dim3 block_my21(cuda::blockdim_col, cuda::blockdim_row);
//    const dim3 grid_my21((My21 + cuda::blockdim_col - 1) / cuda::blockdim_col,
//            (get_geom().get_nx() + cuda::blockdim_row - 1) / (cuda::blockdim_row));
//                         
//
//    kernel_gen_coeffs<<<grid_my21, block_my21>>>(get_coeffs_dy1(), get_coeffs_dy2(), 
//                                                 twodads::slab_layout_t(get_geom().get_xleft(), get_geom().get_deltax(), 
//                                                                     get_geom().get_ylo(), get_geom().get_deltay(), 
//                                                                     get_geom().get_nx(), 0, My21, 0, twodads::grid_t::cell_centered));
//
//    // Host copy of main and lower diagonal
//    h_diag = new cmplx_t[get_geom().get_nx()];
//    cmplx_t* h_diag_u = new cmplx_t[get_geom().get_nx()];
//    cmplx_t* h_diag_l = new cmplx_t[get_geom().get_nx()];
//
//    // Allocate memory for the lower and main diagonal for tridiagonal matrix factorization
//    // The upper diagonal is equal to the lower diagonal
//    gpuErrchk(cudaMalloc((void**) &d_diag, get_geom().get_nx() * My21 * sizeof(CuCmplx<value_t>)));
//    gpuErrchk(cudaMalloc((void**) &d_diag_u, get_geom().get_nx() * My21 * sizeof(CuCmplx<value_t>)));
//    gpuErrchk(cudaMalloc((void**) &d_diag_l, get_geom().get_nx() * My21 * sizeof(CuCmplx<value_t>)));
//
//    value_t ky2{0.0};                             // ky^2
//    const value_t inv_dx{1.0 / get_geom().get_deltax()};      // 1 / delta_x
//    const value_t inv_dx2{inv_dx * inv_dx};       // 1 / delta_x^2
//    const value_t Ly{static_cast<value_t>(get_geom().get_my()) * get_geom().get_deltay()};
//
//    // Initialize the main diagonal separately for every ky
//    for(size_t m = 0; m < My21; m++)
//    {
//        ky2 = twodads::TWOPI * twodads::TWOPI * static_cast<value_t>(m * m) / (Ly * Ly);
//        for(size_t n = 0; n < get_geom().get_nx(); n++)
//        {
//            h_diag[n] = -2.0 * inv_dx2 - ky2;
//        }
//        h_diag[0] = h_diag[0] - inv_dx2;
//        h_diag[get_geom().get_nx() - 1] = h_diag[get_geom().get_nx() - 1] - inv_dx2;
//
//        gpuErrchk(cudaMemcpy(d_diag + m * get_geom().get_nx(), h_diag, get_geom().get_nx() * sizeof(cmplx_t), cudaMemcpyHostToDevice));
//    }
//
//    // Initialize the upper and lower diagonal with 1/delta_x^2
//    for(size_t n = 0; n < get_geom().get_nx(); n++)
//    {
//        h_diag_u[n] = inv_dx2;
//        h_diag_l[n] = inv_dx2;
//    }
//    // Set first/last element of lower/upper diagonal to zero (required by cusparseZgtsvStridedBatch)
//    h_diag_u[get_geom().get_nx() - 1] = 0.0;
//    h_diag_l[0] = 0.0;
//
//    // Concatenate My21 copies of these vector together (required by cusparseZgtsvStridedBatch
//    for(size_t m = 0; m < My21; m++)
//    {
//        gpuErrchk(cudaMemcpy(d_diag_l + m * get_geom().get_nx(), h_diag_l, get_geom().get_nx() * sizeof(cmplx_t), cudaMemcpyHostToDevice)); 
//        gpuErrchk(cudaMemcpy(d_diag_u + m * get_geom().get_nx(), h_diag_u, get_geom().get_nx() * sizeof(cmplx_t), cudaMemcpyHostToDevice)); 
//    }
//
//    delete [] h_diag_l;
//    delete [] h_diag_u;
//}
//


//template <typename T, template <typename> class allocator>
//void deriv_t<T, allocator> :: dx_1(const cuda_array_bc_nogp<T, allocator>& in,
//                                  cuda_array_bc_nogp<T, allocator>& out,
//                                  const size_t t_src, const size_t t_dst)
//{
//}
//
//
//template <typename T, template <typename> class allocator>
//void deriv_t<T, allocator> :: dx_2(const cuda_array_bc_nogp<T, allocator>& in,
//                                   cuda_array_bc_nogp<T, allocator>& out,
//                                   const size_t t_src, const size_t t_dst)
//{
//    detail :: impl_dx2(in, out, t_src, t_dst, allocator<T>{});
//}
//
//
//
//template <typename T, template<typename> class allocator>
//void deriv_t<T, allocator> :: arakawa(const cuda_array_bc_nogp<T, allocator>& u,
//        const cuda_array_bc_nogp<T, allocator>& v,
//        cuda_array_bc_nogp<T, allocator>& res,
//        const size_t t_src, const size_t t_dst)
//{
//    detail :: impl_arakawa(u, v, res, t_src, t_dst, allocator<T>{});
//}
//
//
//template <typename T, template<typename> class allocator>
//deriv_t<T, allocator> :: ~deriv_t()
//{
//    delete [] h_diag;
//}

//// Call three point stencil with centered difference formula for first derivative
//template <typename allocator>
//void derivs<allocator> :: dx_1(const cuda_array_bc_nogp<allocator>& in,
//                               cuda_array_bc_nogp<allocator>& out,
//                               const size_t t_src, const size_t t_dst)
//{
//    static dim3 block_single_row(cuda::blockdim_row, 1);
//    static dim3 grid_single_row((get_geom().get_nx() + cuda::blockdim_row - 1) / cuda::blockdim_row, 1);
//
//    // Call kernel that accesses elements with get_elem; no wrapping around
//    kernel_threepoint_center<<<in.get_grid(), in.get_block()>>>(in.get_array_d(t_src), in.get_address(),
//                                                                out.get_array_d(t_dst), 
//                                                                [=] __device__ (value_t u_left, value_t u_middle, 
//                                                                                value_t u_right, value_t inv_dx, 
//                                                                                value_t inv_dx2) -> value_t
//                                                                {
//                                                                  return(0.5 * (u_right - u_left) * inv_dx);
//                                                                }, 
//                                                                out.get_geom());
//
//    // Call kernel that accesses elements with operator(); interpolates ghost point values
//    kernel_threepoint_single_row<<<grid_single_row, block_single_row>>>(in.get_array_d(t_src), in.get_address(),
//                                                                        out.get_array_d(t_dst), 
//                                                                        [=] __device__ (value_t u_left, value_t u_middle, 
//                                                                                        value_t u_right, value_t inv_dx, 
//                                                                                        value_t inv_dx2) -> value_t
//                                                                        {
//                                                                          return(0.5 * (u_right - u_left) * inv_dx);
//                                                                        },
//                                                                        out.get_geom(), 0);
//
//    // Call kernel that accesses elements with operator(); interpolates ghost point values
//    kernel_threepoint_single_row<<<grid_single_row, block_single_row>>>(in.get_array_d(t_src), in.get_address(),
//                                                                        out.get_array_d(t_dst), 
//                                                                        [=] __device__ (value_t u_left, value_t u_middle, 
//                                                                                        value_t u_right, value_t inv_dx, 
//                                                                                        value_t inv_dx2) -> value_t
//                                                                        {
//                                                                          return(0.5 * (u_right - u_left) * inv_dx);
//                                                                        }, 
//                                                                        out.get_geom(), out.get_geom().get_nx() - 1);
//}
//
//
//// Call three point stencil with centered difference formula for first derivative
//template <typename allocator>
//void derivs<allocator> :: dx_2(const cuda_array_bc_nogp<allocator>& in,
//                               cuda_array_bc_nogp<allocator>& out,
//                               const size_t t_src, const size_t t_dst)
//{
//    static dim3 block_single_row(cuda::blockdim_row, 1);
//    static dim3 grid_single_row((get_geom().get_nx() + cuda::blockdim_row - 1) / cuda::blockdim_row, 1);
//
//    // Call kernel that accesses elements with get_elem; no wrapping around
//    kernel_threepoint_center<<<in.get_grid(), in.get_block()>>>(in.get_array_d(t_src), in.get_address(),
//                                                                out.get_array_d(t_dst), 
//                                                                [=] __device__ (value_t u_left, value_t u_middle, 
//                                                                                value_t u_right, value_t inv_dx, 
//                                                                                value_t inv_dx2) -> value_t
//                                                                {
//                                                                  return((u_left + u_right - 2.0 * u_middle) * inv_dx2);
//                                                                }, 
//                                                                out.get_geom());
//
//    // Call kernel that accesses elements with operator(); interpolates ghost point values
//    kernel_threepoint_single_row<<<grid_single_row, block_single_row>>>(in.get_array_d(t_src), in.get_address(),
//                                                                        out.get_array_d(t_dst), 
//                                                                        [=] __device__ (value_t u_left, value_t u_middle, 
//                                                                                        value_t u_right, value_t inv_dx, 
//                                                                                        value_t inv_dx2) -> value_t
//                                                                        {
//                                                                          return((u_left + u_right - 2.0 * u_middle) * inv_dx2);
//                                                                        },
//                                                                        out.get_geom(), 0);
//
//    // Call kernel that accesses elements with operator(); interpolates ghost point values
//    kernel_threepoint_single_row<<<grid_single_row, block_single_row>>>(in.get_array_d(t_src), in.get_address(),
//                                                                        out.get_array_d(t_dst), 
//                                                                        [=] __device__ (value_t u_left, value_t u_middle, 
//                                                                                        value_t u_right, value_t inv_dx, 
//                                                                                        value_t inv_dx2) -> value_t
//                                                                        {
//                                                                          return((u_left + u_right - 2.0 * u_middle) * inv_dx2);
//                                                                        }, 
//                                                                        out.get_geom(), out.get_geom().get_nx() - 1);
//}
//
//
//template <typename allocator>
//void derivs<allocator> :: dy_1(cuda_array_bc_nogp<allocator>& in,
//                               cuda_array_bc_nogp<allocator> &out,
//                               const size_t t_src, const size_t t_dst)
//{
//    const dim3 block_my21(cuda::blockdim_col, cuda::blockdim_row);
//    const dim3 grid_my21((My21 + cuda::blockdim_col - 1) / cuda::blockdim_col,
//            (get_geom().get_nx() + cuda::blockdim_row - 1) / (cuda::blockdim_row));
//    // In-place DFT of in, multiply by kmap, in-place iDFT of in
//    if(!(in.is_transformed()))
//    {
//        cout << "dy_1: DFT" << endl;
//        myfft.dft_r2c(in.get_array_d(t_src), reinterpret_cast<CuCmplx<value_t>*>(in.get_array_d(t_src)));
//        in.set_transformed(true);
//    }
//
//    // Multiply with coefficients for ky
//    kernel_multiply_map<<<grid_my21, block_my21>>>(reinterpret_cast<CuCmplx<value_t>*>(in.get_array_d(t_src)),
//                                                   get_coeffs_dy1(),
//                                                   reinterpret_cast<CuCmplx<value_t>*>(out.get_array_d(t_dst)),
//                                                   [=] __device__ (CuCmplx<value_t> val_in, CuCmplx<value_t> val_map) -> CuCmplx<value_t>
//                                                   {
//                                                     return(val_in * CuCmplx<value_t>(0.0, val_map.im()));
//                                                   },
//                                                   twodads::slab_layout_t(get_geom().get_xleft(), get_geom().get_deltax(), 
//                                                                       get_geom().get_ylo(), get_geom().get_deltay(), 
//                                                                       get_geom().get_nx(), 0, My21, 0, twodads::grid_t::cell_centered));
//
//    myfft.dft_c2r(reinterpret_cast<CuCmplx<value_t>*>(out.get_array_d(t_src)), out.get_array_d(t_src));
//    out.set_transformed(false);
//    out.normalize(t_src);
//}
//
//
//template <typename allocator>
//void derivs<allocator> :: dy_2(cuda_array_bc_nogp<allocator>& in,
//                               cuda_array_bc_nogp<allocator> &out,
//                               const size_t t_src, const size_t t_dst)
//{
//    const dim3 block_my21(cuda::blockdim_col, cuda::blockdim_row);
//    const dim3 grid_my21((My21 + cuda::blockdim_col - 1) / cuda::blockdim_col,
//                         (get_geom().get_nx() + cuda::blockdim_row - 1) / (cuda::blockdim_row));
//
//    // In-place DFT of in, multiply by kmap, in-place iDFT of in
//    if(!(in.is_transformed()))
//    {
//        cout << "dy_1: DFT" << endl;
//        myfft.dft_r2c(in.get_array_d(t_src), reinterpret_cast<CuCmplx<value_t>*>(in.get_array_d(t_src)));
//        in.set_transformed(true);
//    }
//
//    // Multiply with coefficients for ky
//    kernel_multiply_map<<<grid_my21, block_my21>>>(reinterpret_cast<CuCmplx<value_t>*>(in.get_array_d(t_src)),
//                                                   get_coeffs_dy2(),
//                                                   reinterpret_cast<CuCmplx<value_t>*>(out.get_array_d(t_dst)),
//                                                   [=] __device__ (CuCmplx<value_t> val_in, CuCmplx<value_t> val_map) -> CuCmplx<value_t>
//                                                   {
//                                                     return(val_in * val_map.im());
//                                                   },
//                                                   twodads::slab_layout_t(get_geom().get_xleft(), get_geom().get_deltax(), 
//                                                                       get_geom().get_ylo(), get_geom().get_deltay(), 
//                                                                       get_geom().get_nx(), 0, My21, 0, twodads::grid_t::cell_centered));
//
//    myfft.dft_c2r(reinterpret_cast<CuCmplx<value_t>*>(out.get_array_d(t_src)), out.get_array_d(t_src));
//    out.set_transformed(false);
//    out.normalize(t_src);
//}
//
//
//template <typename allocator>
//void derivs<allocator> :: arakawa(const cuda_array_bc_nogp<allocator>& u, const cuda_array_bc_nogp<allocator>& v, cuda_array_bc_nogp<allocator>& res,
//                                  const size_t t_src, const size_t t_dst)
//{
//    // Thread layout for accessing a single row (m = 0..My-1, n = 0, Nx-1)
//    static dim3 block_single_row(cuda::blockdim_row, 1);
//    static dim3 grid_single_row((get_geom().get_nx() + cuda::blockdim_row - 1) / cuda::blockdim_row, 1);
//
//    // Thread layout for accessing a single column (m = 0, My - 1, n = 0...Nx-1)
//    static dim3 block_single_col(1, cuda::blockdim_col);
//    static dim3 grid_single_col(1, (get_geom().get_my() + cuda::blockdim_col - 1) / cuda::blockdim_col);
//
//    kernel_arakawa_center<<<u.get_grid(), u.get_block()>>>(u.get_array_d(t_src), u.get_address(),
//                                                           v.get_array_d(t_src), v.get_address(),
//                                                           res.get_array_d(t_dst), u.get_geom());
//    
//    // Create address objects to access ghost points 
//    kernel_arakawa_single_row<<<grid_single_row, block_single_row>>>(u.get_array_d(t_src), u.get_address(),
//                                                                     v.get_array_d(t_src), v.get_address(),
//                                                                     res.get_array_d(t_dst), get_geom(), 0);
//
//    kernel_arakawa_single_row<<<grid_single_row, block_single_row>>>(u.get_array_d(t_src), u.get_address(),
//                                                                     v.get_array_d(t_src), v.get_address(),
//                                                                     res.get_array_d(t_dst), get_geom(), get_geom().get_nx() - 1);
//
//    kernel_arakawa_single_col<<<grid_single_col, block_single_col>>>(u.get_array_d(t_src), u.get_address(),
//                                                                     v.get_array_d(t_src), v.get_address(),
//                                                                     res.get_array_d(t_dst), get_geom(), 0);
//
//    kernel_arakawa_single_col<<<grid_single_col, block_single_col>>>(u.get_array_d(t_src), u.get_address(),
//                                                                     v.get_array_d(t_src), v.get_address(),
//                                                                     res.get_array_d(t_dst), get_geom(), get_geom().get_my() - 1);
//    cout << "done" << endl;
//}
//
//
//
//template <typename allocator>
//void derivs<allocator> :: invert_laplace(cuda_array_bc_nogp<allocator>& dst, cuda_array_bc_nogp<allocator>& src, 
//                                         const twodads::bc_t bctype_left, const value_t bval_left,
//                                         const twodads::bc_t bctype_right, const value_t bval_right,
//                                         const size_t t_src, const size_t t_dst)
//{
//    const value_t inv_dx2{1.0 / (get_geom().get_deltax() * get_geom().get_deltax())};
//
//    if (dst.get_bvals() != src.get_bvals())
//    {
//        throw assert_error(string("assert_error: invert_laplace: src and dst must have the same boundary conditions\n"));
//    }
//
//    solvers::elliptic my_ell_solver(get_geom());
//
//    //// Solve the tridiagonal system
//    //// 1.) Update the main diagonal for ky=0 mode with the boundary values
//    ////     Add the Fourier coefficient for mode m=0 of f(0.0, y) = u: hat(u) = My * u
//    for(size_t n = 0; n < get_geom().get_nx(); n++)
//    {
//        h_diag[n] = -2.0 * inv_dx2; // -ky2(=0) 
//    }
//    switch(bctype_left)
//    {
//        case twodads::bc_t::bc_dirichlet:
//            h_diag[0] = -3.0 * inv_dx2 + 2.0 * bval_left * twodads::TWOPI * static_cast<value_t>(get_geom().get_my()) / get_geom().get_Ly();
//            break;
//        case twodads::bc_t::bc_neumann:
//            h_diag[0] = -3.0 * inv_dx2 - bval_left * twodads::TWOPI * static_cast<value_t>(get_geom().get_my()) / get_geom().get_Ly();
//            break;
//        case twodads::bc_t::bc_periodic:
//            cerr << "Periodic boundary conditions not implemented yet." << endl;
//            cerr << "Treating as dirichlet, bval=0" << endl;
//            break;
//    }
//
//    switch(bctype_right)
//    {
//        case twodads::bc_t::bc_dirichlet:
//            h_diag[geom.get_nx() - 1] = -3.0 * inv_dx2 + 2.0 * bval_right * twodads::TWOPI * static_cast<value_t>(get_geom().get_my()) / get_geom().get_Ly();
//            break;
//        case twodads::bc_t::bc_neumann:
//            h_diag[geom.get_nx() - 1] = -3.0 * inv_dx2 - bval_right * twodads::TWOPI * static_cast<value_t>(get_geom().get_my()) / get_geom().get_Ly();
//            break;
//        case twodads::bc_t::bc_periodic:
//            cerr << "Periodic boundary conditions not implemented yet." << endl;
//            cerr << "Treating as dirichlet, bval=0" << endl;
//            break;
//    }
//    gpuErrchk(cudaMemcpy(d_diag, h_diag, get_geom().get_nx() * sizeof(cmplx_t), cudaMemcpyHostToDevice));
//
//    my_ell_solver.solve((cuDoubleComplex*) dst.get_array_d(t_dst), (cuDoubleComplex*) src.get_array_d(t_src),
//                        (cuDoubleComplex*)d_diag_l, (cuDoubleComplex*) d_diag, nullptr);
//
//    dst.set_transformed(true);
//}


//#ifdef DEBUG
//    // Test the precision of the solution
//
//    // 1.) Build the laplace matrix in csr form
//    // Host data
//    size_t nnz{get_geom().get_nx() + 2 * (get_geom().get_nx() - 1)};    // Main diagonal plus 2 side diagonals
//    cmplx_t* csrValA_h = new cmplx_t[nnz];  
//    int* csrRowPtrA_h = new int[get_geom().get_nx() + 1];
//    int* csrColIndA_h = new int[nnz];
//
//    // Input columns
//    cmplx_t* h_inp_mat_col = new cmplx_t[get_geom().get_nx()];
//    // Result columns
//    cmplx_t* h_tmp_mat_col = new cmplx_t[get_geom().get_nx()];
//
//    // Device data
//    cmplx_t* csrValA_d{nullptr};
//    int* csrRowPtrA_d{nullptr}; 
//    int* csrColIndA_d{nullptr};
//
//    cmplx_t* d_inp_mat{nullptr};
//
//    // Some constants we need later on
//    //const value_t inv_dx2 = 1.0 / (get_geom().get_deltax() * get_geom().get_deltax());
//    const cmplx_t inv_dx2_cmplx = cmplx_t(inv_dx2);
//
//    gpuErrchk(cudaMalloc((void**) &csrValA_d, nnz * sizeof(cmplx_t)));
//    gpuErrchk(cudaMalloc((void**) &csrRowPtrA_d, (get_geom().get_nx() + 1) * sizeof(int)));
//    gpuErrchk(cudaMalloc((void**) &csrColIndA_d, nnz * sizeof(int)));
//    gpuErrchk(cudaMalloc((void**) &d_inp_mat, get_geom().get_nx() * My21 * sizeof(cmplx_t)));
//
//    // Build Laplace matrix structure: ColIndA and RowPtrA
//    // Matrix values on main diagonal are updated individually lateron for each ky mode.
//    // Side bands (upper/lower diagonal) are computed once here.
//    csrColIndA_h[0] = 0;
//    csrColIndA_h[1] = 1;
//
//    csrRowPtrA_h[0] = 0;
//    csrValA_h[1] = inv_dx2_cmplx;
//
//    for(size_t n = 2; n < nnz - 3; n += 3)
//    {
//        csrColIndA_h[n    ] = static_cast<int>((n - 2) / 3);
//        csrColIndA_h[n + 1] = static_cast<int>((n - 2) / 3) + 1;
//        csrColIndA_h[n + 2] = static_cast<int>((n - 2) / 3) + 2;
//
//        csrRowPtrA_h[(n - 2) / 3 + 1] = static_cast<int>(n); 
//
//        csrValA_h[n] = inv_dx2_cmplx;
//        csrValA_h[n + 2] = inv_dx2_cmplx;
//    }   
//
//    csrColIndA_h[nnz - 2] = static_cast<int>(get_geom().get_nx() - 2);
//    csrColIndA_h[nnz - 1] = static_cast<int>(get_geom().get_nx() - 1);  
//
//    csrRowPtrA_h[get_geom().get_nx() - 1] = static_cast<int>(nnz - 2);
//    csrRowPtrA_h[get_geom().get_nx()] = static_cast<int>(nnz);
//
//    csrValA_h[nnz - 2] = cmplx_t(inv_dx2);
//
//    gpuErrchk(cudaMemcpy(csrRowPtrA_d, csrRowPtrA_h, (get_geom().get_nx() + 1) * sizeof(int), cudaMemcpyHostToDevice)); 
//    gpuErrchk(cudaMemcpy(csrColIndA_d, csrColIndA_h, nnz * sizeof(int), cudaMemcpyHostToDevice));
//
//    cusparseMatDescr_t mat_type;
//    cusparseCreateMatDescr(&mat_type); 
//
//    gpuErrchk(cudaMemcpy(csrColIndA_h, csrColIndA_d, nnz * sizeof(int), cudaMemcpyDeviceToHost));
//    cout << endl;
//
//    // 2.) Compare solution for all ky modes
//    //     -> Solution of ZgtsvStridedBatch is stored, column-wise in d_tmp_mat. Apply Laplace matrix A to this data.
//    //     -> Input to ZgtsvStridedBatch is stored, column-wise in d_inp_mat. Compare A * d_tmp_mat to d_inp_mat
//    //     -> Iterate over ky modes 
//    //        * update values of csrValA_h for the current mode
//    //        * Create Laplace matrix A
//    //        * Apply Laplace matrix to d_tmp_mat, column-wise
//    //        * Compute ||(A * d_tmp_mat - d_inp_mat)||_2
//    
//    if((cublas_status = cublasZgeam(//cublas_handle,
//                                    solvers::cublas_handle_t::get_handle(),
//                                    CUBLAS_OP_T, CUBLAS_OP_N,
//                                    Nx_int, My21_int,
//                                    &alpha, // 1.0 + 0.0i
//                                    (cuDoubleComplex*) src.get_array_d(t_src), My21_int,
//                                    &beta,  // 0.0 + 0.0i
//                                    nullptr, Nx_int,
//                                    (cuDoubleComplex*) d_inp_mat, Nx_int
//                                    )) != CUBLAS_STATUS_SUCCESS)
//    {
//        cerr << "cublas_status: " << cublas_status << endl;
//        throw cublas_err(cublas_status);
//    } 
//
//    // Pointers to current row in d_inp_mat (the result we check against) and
//    //                            d_tmp_mat (the input for Dcsrmv)
//    cmplx_t* row_ptr_d_inp{nullptr};
//    cmplx_t* row_ptr_d_tmp{nullptr};
//
//    value_t ky2{0.0};
//    const value_t Ly{static_cast<value_t>(get_geom().get_my()) * get_geom().get_deltay()};
//    for(size_t m = 0; m < My21; m++)
//    //for(size_t m = 0; m < 2; m++)
//    {
//        row_ptr_d_inp = d_inp_mat + m * get_geom().get_nx();
//        row_ptr_d_tmp = d_tmp_mat + m * get_geom().get_nx();
//
//        // Copy reference data in d_inp_mat
//        // These are fourier coefficients for ky=0 mode at various xn positions
//        gpuErrchk(cudaMemcpy(h_inp_mat_col, row_ptr_d_inp, get_geom().get_nx() * sizeof(cmplx_t), cudaMemcpyDeviceToHost));
//        //for(size_t n = 0; n < Nx; n++)
//        //{
//        //    cout << n << ":\t h_inp_mat_col = " << h_inp_mat_col[n] << endl;
//        //}
//
//        // Update values of csrValA_h
//        // Assume Dirichlet=0 boundary conditions for now
//        ky2 = twodads::TWOPI * twodads::TWOPI * static_cast<value_t>(m * m) / (Ly * Ly);
//        csrValA_h[0] = cmplx_t(-3.0 * inv_dx2 - ky2);
//        for(size_t n = 2; n < nnz - 3; n += 3)
//        {
//            csrValA_h[n + 1] = cmplx_t(-2.0 * inv_dx2 - ky2);
//        }
//        csrValA_h[nnz - 1] = cmplx_t(-3.0 * inv_dx2 - ky2);
//
//        //for(size_t n = 0; n < nnz; n++)
//        //{
//        //    cout << n << ":\tcsrValA_h = " << csrValA_h[n] << "\t\t\tcsrColIndA = " << csrColIndA_h[n];
//        //    if(n < Nx + 1)
//        //        cout << "\t\tcsrRowPtrA_h = " << csrRowPtrA_h[n];
//        //    cout << endl;
//        //}
//        //cout << "===========================================================================================" << endl;
//
//        gpuErrchk(cudaMemcpy(csrValA_d, csrValA_h, nnz * sizeof(cmplx_t), cudaMemcpyHostToDevice));
//        
//        // Apply Laplace matrix to every column in d_tmp_mat.
//        // Overwrite the current column in d_inp_mat
//        // Store result in h_tmp_mat
//        if((cusparse_status = cusparseZcsrmv(//cusparse_handle, 
//                                             solvers::cusparse_handle_t::get_handle(),
//                                             CUSPARSE_OPERATION_NON_TRANSPOSE,   
//                                             static_cast<int>(get_geom().get_nx()), static_cast<int>(get_geom().get_nx()), static_cast<int>(nnz),
//                                             &alpha, mat_type,
//                                             (cuDoubleComplex*) csrValA_d,
//                                             csrRowPtrA_d,
//                                             csrColIndA_d,
//                                             (cuDoubleComplex*) row_ptr_d_tmp,
//                                             &beta,
//                                             (cuDoubleComplex*) row_ptr_d_inp)
//           ) != CUSPARSE_STATUS_SUCCESS)
//        {
//            cerr << "cusparse_status = " << cusparse_status << endl;
//            throw cusparse_err(cusparse_status);
//        }
//
//        gpuErrchk(cudaMemcpy(h_tmp_mat_col, row_ptr_d_inp, get_geom().get_nx() * sizeof(cmplx_t), cudaMemcpyDeviceToHost));
//        //if(m > 4 && m < 7)
//        //{
//        //    for(size_t n = 0; n < Nx; n++)
//        //    {
//        //        cout << n << ": h_inp_mat_col = " << h_inp_mat_col[n] << "\th_tmp_mat_col = " << h_tmp_mat_col[n] << endl;
//        //    }
//        //}
//
//        // Compute L2 distance between h_inp_mat and h_tmp_mat
//        value_t L2_norm{0.0};
//        for(size_t n = 0; n < get_geom().get_nx(); n++)
//        {
//            L2_norm += ((h_inp_mat_col[n] - h_tmp_mat_col[n]) * (h_inp_mat_col[n] - h_tmp_mat_col[n])).abs();
//        }
//        L2_norm = sqrt(L2_norm / static_cast<value_t>(get_geom().get_nx()));
//        cout << m << ": ky = " << sqrt(ky2) << "\t L2 = " << L2_norm << endl;
//    }
//    row_ptr_d_inp = nullptr;
//    row_ptr_d_tmp = nullptr;
//
//    // 4.) Clean up sparse storage
//    cudaFree(d_inp_mat);
//    cudaFree(csrRowPtrA_d);
//    cudaFree(csrColIndA_d);
//    cudaFree(csrValA_d);
//    
//    // Input columns
//    delete [] h_tmp_mat_col;
//    delete [] h_inp_mat_col;
//
//    delete [] csrColIndA_h;
//    delete [] csrRowPtrA_h;
//    delete [] csrValA_h;
//#endif


