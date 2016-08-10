/*
 * Interface to derivation functions
 */

#ifndef DERIVATIVES_H
#define DERIVATIVES_H

#include "cuda_types.h"
#include "cuda_array_bc_nogp.h"
#include "cucmplx.h"
#include "error.h"
#include "dft_type.h"
#include "solvers.h"
#include <cusolverSp.h>
#include <cublas_v2.h>
#include <cassert>
#include <fstream>


#ifdef __CUDACC__

__device__ inline size_t d_get_col_2()
{
    return (blockIdx.x * blockDim.x + threadIdx.x); 
}


__device__ inline size_t d_get_row_2()
{
    return (blockIdx.y * blockDim.y + threadIdx.y); 
}


__device__ inline bool good_idx2(size_t row, size_t col, const cuda::slab_layout_t geom)
{
    return((row < geom.get_nx()) && (col < geom.get_my()));
}  


// Apply three point stencil to points within the domain, rows 1..Nx-2
template <typename T, typename O>
__global__
void kernel_threepoint_center(const T* u, address_t<T>** address_u,
                              T* result, O stencil_func, const cuda::slab_layout_t geom)
{
    const int col{static_cast<int>(d_get_col_2())};
    const int row{static_cast<int>(d_get_row_2())};
    const size_t index{row * (geom.get_my() + geom.get_pad_y()) + col};
    const T inv_dx{1.0 / geom.get_deltax()};
    const T inv_dx2{1.0 / (geom.get_deltax() * geom.get_deltax())};

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
                                  const cuda::slab_layout_t geom, const int row)
{
    const int col{static_cast<int>(d_get_col_2())};
    const size_t index{row * (geom.get_my() + geom.get_pad_y()) + col};
    const T inv_dx{1.0 / geom.get_deltax()};
    const T inv_dx2{1.0 / (geom.get_deltax() * geom.get_deltax())};

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
                           T* result, const cuda::slab_layout_t geom)
{
    const int col{static_cast<int>(d_get_col_2())};
    const int row{static_cast<int>(d_get_row_2())};
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
                               T* result, const cuda::slab_layout_t geom,
                               const int row)
{
    // Use int for col and row to pass them into address<T>::operator()
    const int col{static_cast<int>(d_get_col_2())};
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
                               T* result, const cuda::slab_layout_t geom, const int col)
{
    const int row{static_cast<int>(d_get_row_2())};
    const size_t index{row * (geom.get_my() + geom.get_pad_y()) + col}; 
    const real_t inv_dx_dy{-1.0 / (12.0 * geom.get_deltax() * geom.get_deltay())};

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
void kernel_gen_coeffs(CuCmplx<T>* kmap_dx1, CuCmplx<T>* kmap_dx2, cuda::slab_layout_t geom)
{
    const size_t col{d_get_col_2()};
    const size_t row{d_get_row_2()};
    const size_t index{row * (geom.get_my() + geom.get_pad_y()) + col}; 
    const T two_pi_Lx{cuda::TWOPI / geom.get_Lx()};
    const T two_pi_Ly{cuda::TWOPI / (static_cast<T>((geom.get_my() - 1) * 2) * geom.get_deltay())}; 
    const size_t My{(geom.get_my() - 1) * 2};

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
                            cuda::slab_layout_t geom)
{
    const size_t col{d_get_col_2()};
    const size_t row{d_get_row_2()};
    const size_t index{row * (geom.get_my() + geom.get_pad_y()) + col}; 

    if(col < geom.get_my() && row < geom.get_nx())
    {
        out[index] = op_func(in[index], map[index]);
    }
}


#endif //__CUDACC__

/*
 * Datatype that provides derivation routines and elliptic solver
 */

template <typename allocator>
class derivs
{
    public:
        using value_t = typename my_allocator_traits<allocator> :: value_type;
        using cmplx_t = CuCmplx<value_t>;

        derivs(const cuda::slab_layout_t);
        ~derivs();

        // Compute first derivative in x-direction
        void dx_1(const cuda_array_bc_nogp<allocator>&, cuda_array_bc_nogp<allocator>&, 
                  const size_t, const size_t);

        // Compute second derivative in x-direction
        void dx_2(const cuda_array_bc_nogp<allocator>&, cuda_array_bc_nogp<allocator>&, 
                  const size_t, const size_t);

        // Compute first derivative in y-direction
        void dy_1(cuda_array_bc_nogp<allocator>&, cuda_array_bc_nogp<allocator>&,
                  const size_t, const size_t);

        // Compute second derivative in y-direction
        void dy_2(cuda_array_bc_nogp<allocator>&, cuda_array_bc_nogp<allocator>&,
                  const size_t, const size_t);

        // Invert laplace equation
        void invert_laplace(cuda_array_bc_nogp<allocator>&, cuda_array_bc_nogp<allocator>&, 
                            const cuda::bc_t, const value_t,
                            const cuda::bc_t, const value_t,
                            const size_t, const size_t);
       
        // Compute arakawa bracket 
        void arakawa(const cuda_array_bc_nogp<allocator>&, 
                     const cuda_array_bc_nogp<allocator>&, 
                     cuda_array_bc_nogp<allocator>&, 
                     const size_t, const size_t);
   

        // Initialize DFT, 1d-periodic, column-major 
        cmplx_t* get_coeffs_dy1() const {return(d_coeffs_dy1);};
        cmplx_t* get_coeffs_dy2() const {return(d_coeffs_dy2);};
        cuda::slab_layout_t get_geom() const {return(geom);};

    private:
        const size_t My21;
        const cuda::slab_layout_t geom;
        dft_object_t<cuda::real_t> myfft;

        // Coefficient storage for spectral derivation
        cmplx_t* d_coeffs_dy1;
        cmplx_t* d_coeffs_dy2;
        // Matrix storage for solving tridiagonal equations
        cmplx_t* d_diag;     // Main diagonal
        cmplx_t* d_diag_l;   // lower diagonal
        cmplx_t* d_diag_u;   // upper diagonal, for Laplace equation this is the same as the lower diagonal
        cmplx_t* h_diag;     // Main diagonal, host copy. This one is updated with the boundary conditions passed to invert_laplace routine.
};


template <typename allocator>
derivs<allocator> :: derivs(const cuda::slab_layout_t _geom) : 
    My21(static_cast<int>(_geom.get_my()) / 2 + 1), 
    geom(_geom),
    myfft(geom, cuda::dft_t::dft_1d),
    d_coeffs_dy1{nullptr}, d_coeffs_dy2{nullptr},
    d_diag{nullptr}, d_diag_l{nullptr}, d_diag_u{nullptr},
    h_diag{nullptr}
{
    cout << "Allocating " <<  get_geom().get_nx() * My21 * sizeof(CuCmplx<value_t>) << " bytes" << endl;
    gpuErrchk(cudaMalloc((void**) &d_coeffs_dy1, get_geom().get_nx() * My21 * sizeof(CuCmplx<value_t>)))
    gpuErrchk(cudaMalloc((void**) &d_coeffs_dy2, get_geom().get_nx() * My21 * sizeof(CuCmplx<value_t>)))

    const dim3 block_my21(cuda::blockdim_col, cuda::blockdim_row);
    const dim3 grid_my21((My21 + cuda::blockdim_col - 1) / cuda::blockdim_col,
            (get_geom().get_nx() + cuda::blockdim_row - 1) / (cuda::blockdim_row));
                         

    kernel_gen_coeffs<<<grid_my21, block_my21>>>(get_coeffs_dy1(), get_coeffs_dy2(), 
                                                 cuda::slab_layout_t(get_geom().get_xleft(), get_geom().get_deltax(), 
                                                                     get_geom().get_ylo(), get_geom().get_deltay(), 
                                                                     get_geom().get_nx(), 0, My21, 0, cuda::grid_t::cell_centered));

    // Host copy of main and lower diagonal
    h_diag = new cmplx_t[get_geom().get_nx()];
    cmplx_t* h_diag_u = new cmplx_t[get_geom().get_nx()];
    cmplx_t* h_diag_l = new cmplx_t[get_geom().get_nx()];

    // Allocate memory for the lower and main diagonal for tridiagonal matrix factorization
    // The upper diagonal is equal to the lower diagonal
    gpuErrchk(cudaMalloc((void**) &d_diag, get_geom().get_nx() * My21 * sizeof(CuCmplx<value_t>)));
    gpuErrchk(cudaMalloc((void**) &d_diag_u, get_geom().get_nx() * My21 * sizeof(CuCmplx<value_t>)));
    gpuErrchk(cudaMalloc((void**) &d_diag_l, get_geom().get_nx() * My21 * sizeof(CuCmplx<value_t>)));

    value_t ky2{0.0};                             // ky^2
    const value_t inv_dx{1.0 / get_geom().get_deltax()};      // 1 / delta_x
    const value_t inv_dx2{inv_dx * inv_dx};       // 1 / delta_x^2
    const value_t Ly{static_cast<value_t>(get_geom().get_my()) * get_geom().get_deltay()};

    // Initialize the main diagonal separately for every ky
    for(size_t m = 0; m < My21; m++)
    {
        ky2 = cuda::TWOPI * cuda::TWOPI * static_cast<value_t>(m * m) / (Ly * Ly);
        for(size_t n = 0; n < get_geom().get_nx(); n++)
        {
            h_diag[n] = -2.0 * inv_dx2 - ky2;
        }
        h_diag[0] = h_diag[0] - inv_dx2;
        h_diag[get_geom().get_nx() - 1] = h_diag[get_geom().get_nx() - 1] - inv_dx2;

        gpuErrchk(cudaMemcpy(d_diag + m * get_geom().get_nx(), h_diag, get_geom().get_nx() * sizeof(cmplx_t), cudaMemcpyHostToDevice));
    }

    // Initialize the upper and lower diagonal with 1/delta_x^2
    for(size_t n = 0; n < get_geom().get_nx(); n++)
    {
        h_diag_u[n] = inv_dx2;
        h_diag_l[n] = inv_dx2;
    }
    // Set first/last element of lower/upper diagonal to zero (required by cusparseZgtsvStridedBatch)
    h_diag_u[get_geom().get_nx() - 1] = 0.0;
    h_diag_l[0] = 0.0;

    // Concatenate My21 copies of these vector together (required by cusparseZgtsvStridedBatch
    for(size_t m = 0; m < My21; m++)
    {
        gpuErrchk(cudaMemcpy(d_diag_l + m * get_geom().get_nx(), h_diag_l, get_geom().get_nx() * sizeof(cmplx_t), cudaMemcpyHostToDevice)); 
        gpuErrchk(cudaMemcpy(d_diag_u + m * get_geom().get_nx(), h_diag_u, get_geom().get_nx() * sizeof(cmplx_t), cudaMemcpyHostToDevice)); 
    }

    delete [] h_diag_l;
    delete [] h_diag_u;
}


// Call three point stencil with centered difference formula for first derivative
template <typename allocator>
void derivs<allocator> :: dx_1(const cuda_array_bc_nogp<allocator>& in,
                               cuda_array_bc_nogp<allocator>& out,
                               const size_t t_src, const size_t t_dst)
{
    static dim3 block_single_row(cuda::blockdim_row, 1);
    static dim3 grid_single_row((get_geom().get_nx() + cuda::blockdim_row - 1) / cuda::blockdim_row, 1);

    // Call kernel that accesses elements with get_elem; no wrapping around
    kernel_threepoint_center<<<in.get_grid(), in.get_block()>>>(in.get_array_d(t_src), in.get_address(),
                                                                out.get_array_d(t_dst), 
                                                                [=] __device__ (value_t u_left, value_t u_middle, 
                                                                                value_t u_right, value_t inv_dx, 
                                                                                value_t inv_dx2) -> value_t
                                                                {
                                                                  return(0.5 * (u_right - u_left) * inv_dx);
                                                                }, 
                                                                out.get_geom());

    // Call kernel that accesses elements with operator(); interpolates ghost point values
    kernel_threepoint_single_row<<<grid_single_row, block_single_row>>>(in.get_array_d(t_src), in.get_address(),
                                                                        out.get_array_d(t_dst), 
                                                                        [=] __device__ (value_t u_left, value_t u_middle, 
                                                                                        value_t u_right, value_t inv_dx, 
                                                                                        value_t inv_dx2) -> value_t
                                                                        {
                                                                          return(0.5 * (u_right - u_left) * inv_dx);
                                                                        },
                                                                        out.get_geom(), 0);

    // Call kernel that accesses elements with operator(); interpolates ghost point values
    kernel_threepoint_single_row<<<grid_single_row, block_single_row>>>(in.get_array_d(t_src), in.get_address(),
                                                                        out.get_array_d(t_dst), 
                                                                        [=] __device__ (value_t u_left, value_t u_middle, 
                                                                                        value_t u_right, value_t inv_dx, 
                                                                                        value_t inv_dx2) -> value_t
                                                                        {
                                                                          return(0.5 * (u_right - u_left) * inv_dx);
                                                                        }, 
                                                                        out.get_geom(), out.get_geom().get_nx() - 1);
}


// Call three point stencil with centered difference formula for first derivative
template <typename allocator>
void derivs<allocator> :: dx_2(const cuda_array_bc_nogp<allocator>& in,
                               cuda_array_bc_nogp<allocator>& out,
                               const size_t t_src, const size_t t_dst)
{
    static dim3 block_single_row(cuda::blockdim_row, 1);
    static dim3 grid_single_row((get_geom().get_nx() + cuda::blockdim_row - 1) / cuda::blockdim_row, 1);

    // Call kernel that accesses elements with get_elem; no wrapping around
    kernel_threepoint_center<<<in.get_grid(), in.get_block()>>>(in.get_array_d(t_src), in.get_address(),
                                                                out.get_array_d(t_dst), 
                                                                [=] __device__ (value_t u_left, value_t u_middle, 
                                                                                value_t u_right, value_t inv_dx, 
                                                                                value_t inv_dx2) -> value_t
                                                                {
                                                                  return((u_left + u_right - 2.0 * u_middle) * inv_dx2);
                                                                }, 
                                                                out.get_geom());

    // Call kernel that accesses elements with operator(); interpolates ghost point values
    kernel_threepoint_single_row<<<grid_single_row, block_single_row>>>(in.get_array_d(t_src), in.get_address(),
                                                                        out.get_array_d(t_dst), 
                                                                        [=] __device__ (value_t u_left, value_t u_middle, 
                                                                                        value_t u_right, value_t inv_dx, 
                                                                                        value_t inv_dx2) -> value_t
                                                                        {
                                                                          return((u_left + u_right - 2.0 * u_middle) * inv_dx2);
                                                                        },
                                                                        out.get_geom(), 0);

    // Call kernel that accesses elements with operator(); interpolates ghost point values
    kernel_threepoint_single_row<<<grid_single_row, block_single_row>>>(in.get_array_d(t_src), in.get_address(),
                                                                        out.get_array_d(t_dst), 
                                                                        [=] __device__ (value_t u_left, value_t u_middle, 
                                                                                        value_t u_right, value_t inv_dx, 
                                                                                        value_t inv_dx2) -> value_t
                                                                        {
                                                                          return((u_left + u_right - 2.0 * u_middle) * inv_dx2);
                                                                        }, 
                                                                        out.get_geom(), out.get_geom().get_nx() - 1);
}


template <typename allocator>
void derivs<allocator> :: dy_1(cuda_array_bc_nogp<allocator>& in,
                               cuda_array_bc_nogp<allocator> &out,
                               const size_t t_src, const size_t t_dst)
{
    const dim3 block_my21(cuda::blockdim_col, cuda::blockdim_row);
    const dim3 grid_my21((My21 + cuda::blockdim_col - 1) / cuda::blockdim_col,
            (get_geom().get_nx() + cuda::blockdim_row - 1) / (cuda::blockdim_row));
    // In-place DFT of in, multiply by kmap, in-place iDFT of in
    if(!(in.is_transformed()))
    {
        cout << "dy_1: DFT" << endl;
        myfft.dft_r2c(in.get_array_d(t_src), reinterpret_cast<CuCmplx<value_t>*>(in.get_array_d(t_src)));
        in.set_transformed(true);
    }

    // Multiply with coefficients for ky
    kernel_multiply_map<<<grid_my21, block_my21>>>(reinterpret_cast<CuCmplx<value_t>*>(in.get_array_d(t_src)),
                                                   get_coeffs_dy1(),
                                                   reinterpret_cast<CuCmplx<value_t>*>(out.get_array_d(t_dst)),
                                                   [=] __device__ (CuCmplx<value_t> val_in, CuCmplx<value_t> val_map) -> CuCmplx<value_t>
                                                   {
                                                     return(val_in * CuCmplx<value_t>(0.0, val_map.im()));
                                                   },
                                                   cuda::slab_layout_t(get_geom().get_xleft(), get_geom().get_deltax(), 
                                                                       get_geom().get_ylo(), get_geom().get_deltay(), 
                                                                       get_geom().get_nx(), 0, My21, 0, cuda::grid_t::cell_centered));

    myfft.dft_c2r(reinterpret_cast<CuCmplx<value_t>*>(out.get_array_d(t_src)), out.get_array_d(t_src));
    out.set_transformed(false);
    out.normalize(t_src);
}


template <typename allocator>
void derivs<allocator> :: dy_2(cuda_array_bc_nogp<allocator>& in,
                               cuda_array_bc_nogp<allocator> &out,
                               const size_t t_src, const size_t t_dst)
{
    const dim3 block_my21(cuda::blockdim_col, cuda::blockdim_row);
    const dim3 grid_my21((My21 + cuda::blockdim_col - 1) / cuda::blockdim_col,
                         (get_geom().get_nx() + cuda::blockdim_row - 1) / (cuda::blockdim_row));

    // In-place DFT of in, multiply by kmap, in-place iDFT of in
    if(!(in.is_transformed()))
    {
        cout << "dy_1: DFT" << endl;
        myfft.dft_r2c(in.get_array_d(t_src), reinterpret_cast<CuCmplx<value_t>*>(in.get_array_d(t_src)));
        in.set_transformed(true);
    }

    // Multiply with coefficients for ky
    kernel_multiply_map<<<grid_my21, block_my21>>>(reinterpret_cast<CuCmplx<value_t>*>(in.get_array_d(t_src)),
                                                   get_coeffs_dy2(),
                                                   reinterpret_cast<CuCmplx<value_t>*>(out.get_array_d(t_dst)),
                                                   [=] __device__ (CuCmplx<value_t> val_in, CuCmplx<value_t> val_map) -> CuCmplx<value_t>
                                                   {
                                                     return(val_in * val_map.im());
                                                   },
                                                   cuda::slab_layout_t(get_geom().get_xleft(), get_geom().get_deltax(), 
                                                                       get_geom().get_ylo(), get_geom().get_deltay(), 
                                                                       get_geom().get_nx(), 0, My21, 0, cuda::grid_t::cell_centered));

    myfft.dft_c2r(reinterpret_cast<CuCmplx<value_t>*>(out.get_array_d(t_src)), out.get_array_d(t_src));
    out.set_transformed(false);
    out.normalize(t_src);
}


template <typename allocator>
void derivs<allocator> :: arakawa(const cuda_array_bc_nogp<allocator>& u, const cuda_array_bc_nogp<allocator>& v, cuda_array_bc_nogp<allocator>& res,
                                  const size_t t_src, const size_t t_dst)
{
    // Thread layout for accessing a single row (m = 0..My-1, n = 0, Nx-1)
    static dim3 block_single_row(cuda::blockdim_row, 1);
    static dim3 grid_single_row((get_geom().get_nx() + cuda::blockdim_row - 1) / cuda::blockdim_row, 1);

    // Thread layout for accessing a single column (m = 0, My - 1, n = 0...Nx-1)
    static dim3 block_single_col(1, cuda::blockdim_col);
    static dim3 grid_single_col(1, (get_geom().get_my() + cuda::blockdim_col - 1) / cuda::blockdim_col);

    kernel_arakawa_center<<<u.get_grid(), u.get_block()>>>(u.get_array_d(t_src), u.get_address(),
                                                           v.get_array_d(t_src), v.get_address(),
                                                           res.get_array_d(t_dst), u.get_geom());
    
    // Create address objects to access ghost points 
    kernel_arakawa_single_row<<<grid_single_row, block_single_row>>>(u.get_array_d(t_src), u.get_address(),
                                                                     v.get_array_d(t_src), v.get_address(),
                                                                     res.get_array_d(t_dst), get_geom(), 0);

    kernel_arakawa_single_row<<<grid_single_row, block_single_row>>>(u.get_array_d(t_src), u.get_address(),
                                                                     v.get_array_d(t_src), v.get_address(),
                                                                     res.get_array_d(t_dst), get_geom(), get_geom().get_nx() - 1);

    kernel_arakawa_single_col<<<grid_single_col, block_single_col>>>(u.get_array_d(t_src), u.get_address(),
                                                                     v.get_array_d(t_src), v.get_address(),
                                                                     res.get_array_d(t_dst), get_geom(), 0);

    kernel_arakawa_single_col<<<grid_single_col, block_single_col>>>(u.get_array_d(t_src), u.get_address(),
                                                                     v.get_array_d(t_src), v.get_address(),
                                                                     res.get_array_d(t_dst), get_geom(), get_geom().get_my() - 1);
    cout << "done" << endl;
}


template <typename allocator>
derivs<allocator> :: ~derivs()
{
    cudaFree(d_diag_l);
    cudaFree(d_diag_u);
    cudaFree(d_diag);

    delete [] h_diag;

    cudaFree(d_coeffs_dy1);
    cudaFree(d_coeffs_dy2);
}

template <typename allocator>
void derivs<allocator> :: invert_laplace(cuda_array_bc_nogp<allocator>& dst, cuda_array_bc_nogp<allocator>& src, 
                                         const cuda::bc_t bctype_left, const value_t bval_left,
                                         const cuda::bc_t bctype_right, const value_t bval_right,
                                         const size_t t_src, const size_t t_dst)
{
    const value_t inv_dx2{1.0 / (get_geom().get_deltax() * get_geom().get_deltax())};

    if (dst.get_bvals() != src.get_bvals())
    {
        throw assert_error(string("assert_error: invert_laplace: src and dst must have the same boundary conditions\n"));
    }

    solvers::elliptic my_ell_solver(get_geom());

    //// Solve the tridiagonal system
    //// 1.) Update the main diagonal for ky=0 mode with the boundary values
    ////     Add the Fourier coefficient for mode m=0 of f(0.0, y) = u: hat(u) = My * u
    for(size_t n = 0; n < get_geom().get_nx(); n++)
    {
        h_diag[n] = -2.0 * inv_dx2; // -ky2(=0) 
    }
    switch(bctype_left)
    {
        case cuda::bc_t::bc_dirichlet:
            h_diag[0] = -3.0 * inv_dx2 + 2.0 * bval_left * cuda::TWOPI * static_cast<value_t>(get_geom().get_my()) / get_geom().get_Ly();
            break;
        case cuda::bc_t::bc_neumann:
            h_diag[0] = -3.0 * inv_dx2 - bval_left * cuda::TWOPI * static_cast<value_t>(get_geom().get_my()) / get_geom().get_Ly();
            break;
        case cuda::bc_t::bc_periodic:
            cerr << "Periodic boundary conditions not implemented yet." << endl;
            cerr << "Treating as dirichlet, bval=0" << endl;
            break;
    }

    switch(bctype_right)
    {
        case cuda::bc_t::bc_dirichlet:
            h_diag[geom.get_nx() - 1] = -3.0 * inv_dx2 + 2.0 * bval_right * cuda::TWOPI * static_cast<value_t>(get_geom().get_my()) / get_geom().get_Ly();
            break;
        case cuda::bc_t::bc_neumann:
            h_diag[geom.get_nx() - 1] = -3.0 * inv_dx2 - bval_right * cuda::TWOPI * static_cast<value_t>(get_geom().get_my()) / get_geom().get_Ly();
            break;
        case cuda::bc_t::bc_periodic:
            cerr << "Periodic boundary conditions not implemented yet." << endl;
            cerr << "Treating as dirichlet, bval=0" << endl;
            break;
    }
    gpuErrchk(cudaMemcpy(d_diag, h_diag, get_geom().get_nx() * sizeof(cmplx_t), cudaMemcpyHostToDevice));

    my_ell_solver.solve((cuDoubleComplex*) dst.get_array_d(t_dst), (cuDoubleComplex*) src.get_array_d(t_src),
                        (cuDoubleComplex*)d_diag_l, (cuDoubleComplex*) d_diag, nullptr);

    dst.set_transformed(true);
}


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
//        ky2 = cuda::TWOPI * cuda::TWOPI * static_cast<value_t>(m * m) / (Ly * Ly);
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


#endif //DERIVATIVES_H
