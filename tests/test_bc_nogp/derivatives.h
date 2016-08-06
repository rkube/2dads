/*
 * Interface to derivation functions
 */

#ifndef DERIVATIVES_H
#define DERIVATIVES_H

#include "cuda_types.h"
#include "cuda_array_bc_nogp.h"
#include "cucmplx.h"
#include "error.h"
#include <cusolverSp.h>
#include <cublas_v2.h>
#include <cassert>
#include <fstream>


#ifdef __CUDACC__

__device__ inline size_t d_get_col_2(){
    return (blockIdx.x * blockDim.x + threadIdx.x); 
}


__device__ inline size_t d_get_row_2(){
    return (blockIdx.y * blockDim.y + threadIdx.y); 
}


__device__ inline bool good_idx2(size_t row, size_t col, const cuda::slab_layout_t geom){
    return((row < geom.Nx) && (col < geom.My));
}  


// compute the first derivative in x-direction in rows 1..Nx-2
// no interpolation needed
template <typename T>
__global__ 
void kernel_dx1_center(T* in, T* out, const cuda::bvals_t<T> bc, const cuda::slab_layout_t geom)
{
    const size_t col{d_get_col_2()};
    const size_t row{d_get_row_2()};
    const size_t index{row * (geom.My + geom.pad_y) + col};
    const T inv_2_dx{0.5 / geom.delta_x};

    if(row > 0 && row < geom.Nx - 1 && col < geom.My)
    {
        // Index of element to the "right" (in x-direction)
        const size_t idx_r{(row + 1) * (geom.My + geom.pad_y) + col};
        // Index of element to the "left" (in x-direction)
        const size_t idx_l{(row - 1) * (geom.My + geom.pad_y) + col};

        out[index] = (in[idx_r] - in[idx_l]) * inv_2_dx;
    }   
}


// Compute first derivative in row n=0
// Interpolate to the value in row n=-1 get the ghost point value
template <typename T>
__global__ 
void kernel_dx1_boundary_left(T* in, T* out, const cuda::bvals_t<T> bc, const cuda::slab_layout_t geom)
{
    const size_t col{d_get_col_2()};
    const size_t row{0};
    const size_t index{col};
    const T inv_2_dx{0.5 / geom.delta_x};

    if(col < geom.My)
    {
        // The value to the right in column 1
        const T val_r{in[(geom.My + geom.pad_y) + col]};
        // Interpolate the value in column -1
        T val_l{-1.0};
        switch(bc.bc_left)
        {
            case cuda::bc_t::bc_dirichlet:
                 val_l = 2.0 * bc.bval_left - in[index];
                break;
            case cuda::bc_t::bc_neumann:
                val_l = -1.0 * geom.delta_x * bc.bval_left + in[index];
                break;
            case cuda::bc_t::bc_periodic:
                val_l = in[(geom.Nx - 1) * (geom.My + geom.pad_y) + col];
                break;
        }
        out[index] = (val_r - val_l) * inv_2_dx;
    }

}


// Compute first derivative in row n = Nx - 1
// Interpolate the value at row n = Nx to get the ghost point value
template <typename T>
__global__ 
void kernel_dx1_boundary_right(T* in, T* out, const cuda::bvals_t<T> bc, const cuda::slab_layout_t geom)
{
    const size_t col{d_get_col_2()};
    const size_t row{geom.Nx - 1};
    const size_t index{row * (geom.My + geom.pad_y) + col};
    const T inv_2_dx{0.5 / geom.delta_x};

    if(col < geom.My)
    {
        // The value to the left in column Nx - 2
        const T val_l{in[(geom.Nx - 2) * (geom.My + geom.pad_y) + col]};
        // Interpolate the value in column -1
        T val_r{-1.0};
        switch(bc.bc_left)
        {
            case cuda::bc_t::bc_dirichlet:
                 val_r = 2.0 * bc.bval_right - in[index];
                break;
            case cuda::bc_t::bc_neumann:
                val_r = geom.delta_x * bc.bval_right + in[index];
                break;
            case cuda::bc_t::bc_periodic:
                val_r = in[col];
                break;
        }
        out[index] = (val_r - val_l) * inv_2_dx;
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


#endif //__CUDACC__

template <typename T>
void dx_1(cuda_array_bc_nogp<T>& in, cuda_array_bc_nogp<T>& out, const size_t tlev,
             const cuda::slab_layout_t geom, const cuda::bvals_t<T> bc)
{
    cout << "Computing x derivative\n";

    // Size of the grid for boundary kernels in x-direction
    dim3 gridsize_line(int((geom.My + cuda::blockdim_row - 1) / cuda::blockdim_row));

    kernel_dx1_center<T> <<<in.get_grid(), in.get_block()>>>(in.get_array_d(tlev), out.get_array_d(0), bc, geom);
    kernel_dx1_boundary_left<<<gridsize_line, cuda::blockdim_row>>>(in.get_array_d(tlev), out.get_array_d(0), bc, geom);
    kernel_dx1_boundary_right<<<gridsize_line, cuda::blockdim_row>>>(in.get_array_d(tlev), out.get_array_d(0), bc, geom);
}


/*
 * Datatype that provides derivation routines and solver for QR factorization
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
        //void dx_2(const cuda_array_bc_nogp<allocator>&, cuda_array_bc_nogp<allocator>&, 
        //          const size_t, const size_t);

        // Compute first derivative in y-direction
        //void dy_1(const cuda_array_bc_nogp<allocator>&, cuda_array_bc_nogp<allocator>&,
        //          const size_t, const size_t);

        // Compute second derivative in y-direction
        //void dy_2(const cuda_array_bc_nogp<allocator>&, cuda_array_bc_nogp<allocator>&,
        //          const size_t, const size_t);

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
    
    private:
        const size_t Nx;
        const size_t My;
        const size_t My21;
        const cuda::slab_layout_t geom;
        // Handles for cusparse library
        cusparseHandle_t cusparse_handle;
        cusparseMatDescr_t cusparse_descr;

        // Handles for cuBLAS library
        cublasHandle_t cublas_handle;

        // Matrix storage for solving tridiagonal equations
        cmplx_t* d_diag;     // Main diagonal
        cmplx_t* d_diag_l;   // lower diagonal
        cmplx_t* d_diag_u;   // upper diagonal, for Laplace equatoin this is the same as the lower diagonal

        cmplx_t* d_tmp_mat;  // workspace, used to transpose matrices for invert_laplace

        cmplx_t* h_diag;     // Main diagonal, host copy. This one is updated with the boundary conditions
                                // passed to invert_laplace routine.
};


template <typename allocator>
derivs<allocator> :: derivs(const cuda::slab_layout_t _geom) :
    Nx(_geom.Nx), 
    My(_geom.My), 
    My21(static_cast<int>(My / 2 + 1)), 
    geom(_geom),
    d_diag{nullptr}, d_diag_l{nullptr}, d_diag_u{nullptr},
    h_diag{nullptr}
{
    // Host copy of main and lower diagonal
    h_diag = new cmplx_t[Nx];
    cmplx_t* h_diag_u = new cmplx_t[Nx];
    cmplx_t* h_diag_l = new cmplx_t[Nx];

    // Initialize cublas
    cublasStatus_t cublas_status;
    if((cublas_status = cublasCreate(&cublas_handle)) != CUBLAS_STATUS_SUCCESS)
    {
        throw cublas_err(cublas_status);
    }

    // Initialize cusparse
    cusparseStatus_t cusparse_status;
    if((cusparse_status = cusparseCreate(&cusparse_handle)) != CUSPARSE_STATUS_SUCCESS)
    {
        throw cusparse_err(cusparse_status);
    }

    if((cusparse_status = cusparseCreateMatDescr(&cusparse_descr)) != CUSPARSE_STATUS_SUCCESS)
    {
        throw cusparse_err(cusparse_status);
    }
    
    cusparseSetMatType(cusparse_descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(cusparse_descr, CUSPARSE_INDEX_BASE_ZERO);

    // Allocate memory for temporary matrix storage
    gpuErrchk(cudaMalloc((void**) &d_tmp_mat, Nx * My21 * sizeof(CuCmplx<value_t>)));

    // Allocate memory for the lower and main diagonal for tridiagonal matrix factorization
    // The upper diagonal is equal to the lower diagonal
    gpuErrchk(cudaMalloc((void**) &d_diag, Nx * My21 * sizeof(CuCmplx<value_t>)));
    gpuErrchk(cudaMalloc((void**) &d_diag_u, Nx * My21 * sizeof(CuCmplx<value_t>)));
    gpuErrchk(cudaMalloc((void**) &d_diag_l, Nx * My21 * sizeof(CuCmplx<value_t>)));

    value_t ky2{0.0};                             // ky^2
    const value_t inv_dx{1.0 / geom.delta_x};      // 1 / delta_x
    const value_t inv_dx2{inv_dx * inv_dx};       // 1 / delta_x^2
    const value_t Ly{static_cast<value_t>(geom.My) * geom.delta_y};

    // Initialize the main diagonal separately for every ky
    for(size_t m = 0; m < My21; m++)
    {
        ky2 = cuda::TWOPI * cuda::TWOPI * static_cast<value_t>(m * m) / (Ly * Ly);
        for(size_t n = 0; n < Nx; n++)
        {
            h_diag[n] = -2.0 * inv_dx2 - ky2;
        }
        h_diag[0] = h_diag[0] - inv_dx2;
        h_diag[Nx - 1] = h_diag[Nx - 1] - inv_dx2;
        gpuErrchk(cudaMemcpy(d_diag + m * Nx, h_diag, Nx * sizeof(cmplx_t), cudaMemcpyHostToDevice));
    }

    // Initialize the upper and lower diagonal with 1/delta_x^2
    for(size_t n = 0; n < Nx; n++)
    {
        h_diag_u[n] = inv_dx2;
        h_diag_l[n] = inv_dx2;
    }
    // Set first/last element of lower/upper diagonal to zero (required by cusparseZgtsvStridedBatch)
    h_diag_u[Nx - 1] = 0.0;
    h_diag_l[0] = 0.0;

    // Concatenate My21 copies of these vector together (required by cusparseZgtsvStridedBatch
    for(size_t m = 0; m < My21; m++)
    {
        gpuErrchk(cudaMemcpy(d_diag_l + m * Nx, h_diag_l, Nx * sizeof(cmplx_t), cudaMemcpyHostToDevice)); 
        gpuErrchk(cudaMemcpy(d_diag_u + m * Nx, h_diag_u, Nx * sizeof(cmplx_t), cudaMemcpyHostToDevice)); 
    }

    delete [] h_diag_l;
    delete [] h_diag_u;
}


template <typename allocator>
void derivs<allocator> :: dx_1(const cuda_array_bc_nogp<allocator>& in,
                               cuda_array_bc_nogp<allocator>& out,
                               const size_t t_src, const size_t t_dst)
{
    cout << "Computing d/dx" << endl;
    //static dim3 block_single_row(cuda::blockdim_row, 1);
    //static dim3 grid_single_row((Nx + cuda::blockdim_row - 1) / cuda::blockdim_row, 1);

    //kernel_derivx_center<<<u.get_grid(), u.get_block()>>>(u.get_array_d(t_src), u.get_address(),
    //                                                      res.get_array_d(t_dst), u.get_geom());
}


template <typename allocator>
void derivs<allocator> :: arakawa(const cuda_array_bc_nogp<allocator>& u, const cuda_array_bc_nogp<allocator>& v, cuda_array_bc_nogp<allocator>& res,
                                  const size_t t_src, const size_t t_dst)
{
    cout << "Computing arakawa bracket for u and v" << endl;
    // Thread layout for accessing a single row (m = 0..My-1, n = 0, Nx-1)
    static dim3 block_single_row(cuda::blockdim_row, 1);
    static dim3 grid_single_row((Nx + cuda::blockdim_row - 1) / cuda::blockdim_row, 1);

    // Thread layout for accessing a single column (m = 0, My - 1, n = 0...Nx-1)
    static dim3 block_single_col(1, cuda::blockdim_col);
    static dim3 grid_single_col(1, (My + cuda::blockdim_col - 1) / cuda::blockdim_col);

    kernel_arakawa_center<<<u.get_grid(), u.get_block()>>>(u.get_array_d(t_src), u.get_address(),
                                                           v.get_array_d(t_src), v.get_address(),
                                                           res.get_array_d(t_dst), u.get_geom());
    
    // Create address objects to access ghost points 
    kernel_arakawa_single_row<<<grid_single_row, block_single_row>>>(u.get_array_d(t_src), u.get_address(),
                                                                     v.get_array_d(t_src), v.get_address(),
                                                                     res.get_array_d(t_dst), geom, 0);

    kernel_arakawa_single_row<<<grid_single_row, block_single_row>>>(u.get_array_d(t_src), u.get_address(),
                                                                     v.get_array_d(t_src), v.get_address(),
                                                                     res.get_array_d(t_dst), geom, Nx - 1);

    kernel_arakawa_single_col<<<grid_single_col, block_single_col>>>(u.get_array_d(t_src), u.get_address(),
                                                                     v.get_array_d(t_src), v.get_address(),
                                                                     res.get_array_d(t_dst), geom, 0);

    kernel_arakawa_single_col<<<grid_single_col, block_single_col>>>(u.get_array_d(t_src), u.get_address(),
                                                                     v.get_array_d(t_src), v.get_address(),
                                                                     res.get_array_d(t_dst), geom, My - 1);
    cout << "done" << endl;
}


template <typename allocator>
derivs<allocator> :: ~derivs()
{
    cudaFree(d_diag_l);
    cudaFree(d_diag_u);
    cudaFree(d_diag);
    cudaFree(d_tmp_mat);

    delete [] h_diag;
    cublasDestroy(cublas_handle);
    cusparseDestroy(cusparse_handle);
}

template <typename allocator>
void derivs<allocator> :: invert_laplace(cuda_array_bc_nogp<allocator>& dst, cuda_array_bc_nogp<allocator>& src, 
                                         const cuda::bc_t bctype_left, const value_t bval_left,
                                         const cuda::bc_t bctype_right, const value_t bval_right,
                                         const size_t t_src, const size_t t_dst)
{
    // Safe casts because we are fancy :)
    const int My_int{static_cast<int>(src.get_my())};
    const int My21_int{static_cast<int>((src.get_my() + src.get_geom().pad_y) / 2)};
    const int Nx_int{static_cast<int>(src.get_nx())};
    const value_t inv_dx2{1.0 / (geom.get_deltax() * geom.get_deltax())};

    const cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
    const cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);
    cublasStatus_t cublas_status;
    cusparseStatus_t cusparse_status;

    if (dst.get_bvals() != src.get_bvals())
    {
        throw assert_error(string("assert_error: invert_laplace: src and dst must have the same boundary conditions\n"));
    }

    // Call cuBLAS to transpose the memory.
    // cuda_array_bc_nogp stores the Fourier coefficients in each column consecutively.
    // For tridiagonal matrix factorization, cusparseZgtsvStridedBatch, the rows (approx. solution at cell center)
    // need to be consecutive
   
    
    /*
     * To verify whether cublasZgeam does a proper job write the raw input and output 
     * memory from the device into text files to circumvent the memory layout used by
     * cuda_array_gp_nobc when writing to files
     *
     * size_t nelem = src.get_nx() *  (src.get_my() + src.get_geom().pad_y);
     * value_t  tmp_arr[nelem];
     * gpuErrchk(cudaMemcpy(tmp_arr, src.get_array_d(0), nelem * sizeof(value_t), cudaMemcpyDeviceToHost));
     * ofstream ofile("cublas_input.dat");
     * if (!ofile)
     *     cerr << "cannot open file" << endl;

     * for(size_t t = 0;  t < nelem; t++)
     * {
     *     ofile << tmp_arr[t] << " ";
     * } 
     * ofile << endl;
     * ofile.close();
     *
     *
     * Then use the following python snippet to verify:
     *
     * Nx = ...
     * My = ...
     * My21 = My / 2 + 1
     * a1 = np.loadtxt('cublas_input.dat')
     * a2 = np.loadtxt('cublas_output.dat')
     * a1f = (a1[::2] + 1j * a1[1::2]).reshape([Nx, My21])
     * a2f = (a2[::2] + 1j * a2[1::2]).reshape([My21, Nx])
     * a2f should now be a1f if everything is correct
     *
     *
     */

    if((cublas_status = cublasZgeam(cublas_handle,
                                    CUBLAS_OP_T, CUBLAS_OP_N,
                                    Nx_int, My21_int,
                                    &alpha,
                                    (cuDoubleComplex*) src.get_array_d(t_src), My21_int,
                                    &beta,
                                    nullptr, Nx_int,
                                    (cuDoubleComplex*) d_tmp_mat, Nx_int
                                    )) != CUBLAS_STATUS_SUCCESS)
    {
        cout << "cublas_status: " << cublas_status << endl;
        throw cublas_err(cublas_status);
    }

    /* Write output of cublasZgeam to file for debugging purposes
    gpuErrchk(cudaMemcpy(tmp_arr, d_tmp_mat, nelem * sizeof(value_t), cudaMemcpyDeviceToHost));
    ofile.open("cublas_output.dat");
    if(!ofile)
        cerr << "cannot open file" << endl;

    for(size_t t = 0;  t < nelem; t++)
    {
        ofile << tmp_arr[t] << " ";
    } 
    ofile << endl;
    ofile.close();
    */
    
    //gpuErrchk(cudaMemcpy(dst.get_array_d(0), d_tmp_mat, src.get_nx() * My21 * sizeof(cmplx_t),
    //                     cudaMemcpyDeviceToDevice));


    // Next step: Solve the tridiagonal system
    // 1.) Update the first and last element of the main diagonal for ky=0 mode with the boundary values
    //     Add the Fourier coefficient for mode m=0 of f(0.0, y) = u: hat(u) = My * u
    switch(bctype_left)
    {
        case cuda::bc_t::bc_dirichlet:
            h_diag[0] = -3.0 * inv_dx2 + 2.0 * bval_left * cuda::TWOPI * static_cast<value_t>(My) / geom.get_Ly();
            break;
        case cuda::bc_t::bc_neumann:
            h_diag[0] = -3.0 * inv_dx2 - bval_left * cuda::TWOPI * static_cast<value_t>(My) / geom.get_Ly();
            break;
        case cuda::bc_t::bc_periodic:
            cerr << "Periodic boundary conditions not implemented yet." << endl;
            cerr << "Treating as dirichlet, bval=0" << endl;
            break;
    }

    switch(bctype_right)
    {
        case cuda::bc_t::bc_dirichlet:
            h_diag[Nx - 1] = -3.0 * inv_dx2 + 2.0 * bval_right * cuda::TWOPI * static_cast<value_t>(My) / geom.get_Ly();
            break;
        case cuda::bc_t::bc_neumann:
            h_diag[Nx - 1] = -3.0 * inv_dx2 - bval_right * cuda::TWOPI * static_cast<value_t>(My) / geom.get_Ly();
            break;
        case cuda::bc_t::bc_periodic:
            cerr << "Periodic boundary conditions not implemented yet." << endl;
            cerr << "Treating as dirichlet, bval=0" << endl;
            break;
    }

    //gpuErrchk(cudaMemcpy(d_diag, h_diag, Nx * sizeof(cmplx_t), cudaMemcpyHostToDevice));

    if((cusparse_status = cusparseZgtsvStridedBatch(cusparse_handle,
                                                    Nx_int,
                                                    (cuDoubleComplex*)(d_diag_l),
                                                    (cuDoubleComplex*)(d_diag),
                                                    (cuDoubleComplex*)(d_diag_l),
                                                    (cuDoubleComplex*)(d_tmp_mat),
                                                    My21,
                                                    Nx_int)) != CUSPARSE_STATUS_SUCCESS)
    {
        throw cusparse_err(cusparse_status);
    }
    // Convert d_tmp_mat from column-major to row-major order
    if((cublas_status = cublasZgeam(cublas_handle,
                                    CUBLAS_OP_T, CUBLAS_OP_N,
                                    My21_int, Nx_int,
                                    &alpha,
                                    (cuDoubleComplex*) d_tmp_mat, Nx_int,
                                    &beta,
                                    nullptr, My21_int,
                                    (cuDoubleComplex*) dst.get_array_d(t_dst), My21_int
                                    )) != CUBLAS_STATUS_SUCCESS)
    {
        cout << cublas_status << endl;
        throw cublas_err(cublas_status);
    }


#ifdef DEBUG
    // Test the precision of the solution

    // 1.) Build the laplace matrix in csr form
    // Host data
    size_t nnz{Nx + 2 * (Nx - 1)};    // Main diagonal plus 2 side diagonals
    cmplx_t* csrValA_h = new cmplx_t[nnz];  
    int* csrRowPtrA_h = new int[Nx + 1];
    int* csrColIndA_h = new int[nnz];

    // Input columns
    cmplx_t* h_inp_mat_col = new cmplx_t[Nx];
    // Result columns
    cmplx_t* h_tmp_mat_col = new cmplx_t[Nx];

    // Device data
    cmplx_t* csrValA_d{nullptr};
    int* csrRowPtrA_d{nullptr}; 
    int* csrColIndA_d{nullptr};

    cmplx_t* d_inp_mat{nullptr};

    // Some constants we need later on
    //const value_t inv_dx2 = 1.0 / (geom.delta_x * geom.delta_x);
    const cmplx_t inv_dx2_cmplx = cmplx_t(inv_dx2);

    gpuErrchk(cudaMalloc((void**) &csrValA_d, nnz * sizeof(cmplx_t)));
    gpuErrchk(cudaMalloc((void**) &csrRowPtrA_d, (Nx + 1) * sizeof(int)));
    gpuErrchk(cudaMalloc((void**) &csrColIndA_d, nnz * sizeof(int)));
    gpuErrchk(cudaMalloc((void**) &d_inp_mat, Nx * My21 * sizeof(cmplx_t)));

    // Build Laplace matrix structure: ColIndA and RowPtrA
    // Matrix values on main diagonal are updated individually lateron for each ky mode.
    // Side bands (upper/lower diagonal) are computed once here.
    csrColIndA_h[0] = 0;
    csrColIndA_h[1] = 1;

    csrRowPtrA_h[0] = 0;
    csrValA_h[1] = inv_dx2_cmplx;

    for(size_t n = 2; n < nnz - 3; n += 3)
    {
        csrColIndA_h[n    ] = static_cast<int>((n - 2) / 3);
        csrColIndA_h[n + 1] = static_cast<int>((n - 2) / 3) + 1;
        csrColIndA_h[n + 2] = static_cast<int>((n - 2) / 3) + 2;

        csrRowPtrA_h[(n - 2) / 3 + 1] = static_cast<int>(n); 

        csrValA_h[n] = inv_dx2_cmplx;
        csrValA_h[n + 2] = inv_dx2_cmplx;
    }   

    csrColIndA_h[nnz - 2] = static_cast<int>(Nx - 2);
    csrColIndA_h[nnz - 1] = static_cast<int>(Nx - 1);  

    csrRowPtrA_h[Nx - 1] = static_cast<int>(nnz - 2);
    csrRowPtrA_h[Nx] = static_cast<int>(nnz);

    csrValA_h[nnz - 2] = cmplx_t(inv_dx2);

    gpuErrchk(cudaMemcpy(csrRowPtrA_d, csrRowPtrA_h, (Nx + 1) * sizeof(int), cudaMemcpyHostToDevice)); 
    gpuErrchk(cudaMemcpy(csrColIndA_d, csrColIndA_h, nnz * sizeof(int), cudaMemcpyHostToDevice));

    cusparseMatDescr_t mat_type;
    cusparseCreateMatDescr(&mat_type); 

    gpuErrchk(cudaMemcpy(csrColIndA_h, csrColIndA_d, nnz * sizeof(int), cudaMemcpyDeviceToHost));
    cout << endl;

    // 2.) Compare solution for all ky modes
    //     -> Solution of ZgtsvStridedBatch is stored, column-wise in d_tmp_mat. Apply Laplace matrix A to this data.
    //     -> Input to ZgtsvStridedBatch is stored, column-wise in d_inp_mat. Compare A * d_tmp_mat to d_inp_mat
    //     -> Iterate over ky modes 
    //        * update values of csrValA_h for the current mode
    //        * Create Laplace matrix A
    //        * Apply Laplace matrix to d_tmp_mat, column-wise
    //        * Compute ||(A * d_tmp_mat - d_inp_mat)||_2
    
    if((cublas_status = cublasZgeam(cublas_handle,
                                    CUBLAS_OP_T, CUBLAS_OP_N,
                                    Nx_int, My21_int,
                                    &alpha, // 1.0 + 0.0i
                                    (cuDoubleComplex*) src.get_array_d(t_src), My21_int,
                                    &beta,  // 0.0 + 0.0i
                                    nullptr, Nx_int,
                                    (cuDoubleComplex*) d_inp_mat, Nx_int
                                    )) != CUBLAS_STATUS_SUCCESS)
    {
        cerr << "cublas_status: " << cublas_status << endl;
        throw cublas_err(cublas_status);
    } 

    // Pointers to current row in d_inp_mat (the result we check against) and
    //                            d_tmp_mat (the input for Dcsrmv)
    cmplx_t* row_ptr_d_inp{nullptr};
    cmplx_t* row_ptr_d_tmp{nullptr};

    value_t ky2{0.0};
    const value_t Ly{static_cast<value_t>(geom.My) * geom.delta_y};
    for(size_t m = 0; m < My21; m++)
    //for(size_t m = 0; m < 2; m++)
    {
        row_ptr_d_inp = d_inp_mat + m * Nx;
        row_ptr_d_tmp = d_tmp_mat + m * Nx;

        // Copy reference data in d_inp_mat
        // These are fourier coefficients for ky=0 mode at various xn positions
        gpuErrchk(cudaMemcpy(h_inp_mat_col, row_ptr_d_inp, Nx * sizeof(cmplx_t), cudaMemcpyDeviceToHost));
        //for(size_t n = 0; n < Nx; n++)
        //{
        //    cout << n << ":\t h_inp_mat_col = " << h_inp_mat_col[n] << endl;
        //}

        // Update values of csrValA_h
        // Assume Dirichlet=0 boundary conditions for now
        ky2 = cuda::TWOPI * cuda::TWOPI * static_cast<value_t>(m * m) / (Ly * Ly);
        csrValA_h[0] = cmplx_t(-3.0 * inv_dx2 - ky2);
        for(size_t n = 2; n < nnz - 3; n += 3)
        {
            csrValA_h[n + 1] = cmplx_t(-2.0 * inv_dx2 - ky2);
        }
        csrValA_h[nnz - 1] = cmplx_t(-3.0 * inv_dx2 - ky2);

        //for(size_t n = 0; n < nnz; n++)
        //{
        //    cout << n << ":\tcsrValA_h = " << csrValA_h[n] << "\t\t\tcsrColIndA = " << csrColIndA_h[n];
        //    if(n < Nx + 1)
        //        cout << "\t\tcsrRowPtrA_h = " << csrRowPtrA_h[n];
        //    cout << endl;
        //}
        //cout << "===========================================================================================" << endl;

        gpuErrchk(cudaMemcpy(csrValA_d, csrValA_h, nnz * sizeof(cmplx_t), cudaMemcpyHostToDevice));
        
        // Apply Laplace matrix to every column in d_tmp_mat.
        // Overwrite the current column in d_inp_mat
        // Store result in h_tmp_mat
        if((cusparse_status = cusparseZcsrmv(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,   
                                             static_cast<int>(Nx), static_cast<int>(Nx), static_cast<int>(nnz),
                                             &alpha, mat_type,
                                             (cuDoubleComplex*) csrValA_d,
                                             csrRowPtrA_d,
                                             csrColIndA_d,
                                             (cuDoubleComplex*) row_ptr_d_tmp,
                                             &beta,
                                             (cuDoubleComplex*) row_ptr_d_inp)
           ) != CUSPARSE_STATUS_SUCCESS)
        {
            cerr << "cusparse_status = " << cusparse_status << endl;
            throw cusparse_err(cusparse_status);
        }

        gpuErrchk(cudaMemcpy(h_tmp_mat_col, row_ptr_d_inp, Nx * sizeof(cmplx_t), cudaMemcpyDeviceToHost));
        //if(m > 4 && m < 7)
        //{
        //    for(size_t n = 0; n < Nx; n++)
        //    {
        //        cout << n << ": h_inp_mat_col = " << h_inp_mat_col[n] << "\th_tmp_mat_col = " << h_tmp_mat_col[n] << endl;
        //    }
        //}

        // Compute L2 distance between h_inp_mat and h_tmp_mat
        value_t L2_norm{0.0};
        for(size_t n = 0; n < Nx; n++)
        {
            L2_norm += ((h_inp_mat_col[n] - h_tmp_mat_col[n]) * (h_inp_mat_col[n] - h_tmp_mat_col[n])).abs();
        }
        L2_norm = sqrt(L2_norm / static_cast<value_t>(Nx));
        cout << m << ": ky = " << sqrt(ky2) << "\t L2 = " << L2_norm << endl;
    }
    row_ptr_d_inp = nullptr;
    row_ptr_d_tmp = nullptr;

    // 4.) Clean up sparse storage
    cudaFree(d_inp_mat);
    cudaFree(csrRowPtrA_d);
    cudaFree(csrColIndA_d);
    cudaFree(csrValA_d);
    
    // Input columns
    delete [] h_tmp_mat_col;
    delete [] h_inp_mat_col;

    delete [] csrColIndA_h;
    delete [] csrRowPtrA_h;
    delete [] csrValA_h;
#endif
}


#endif //DERIVATIVES_H
