/*
 * Interface to derivation functions
 */

#ifndef DERIVATIVES_H
#define DERIVATIVES_H

#include "cuda_types.h"
#include "cuda_array_bc_nogp.h"
#include "cucmplx.h"
#include <cusolverSp.h>
#include <cublas_v2.h>
#include <cassert>

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
void d_dx1_center(T* in, T* out, const cuda::bvals_t<T> bc, const cuda::slab_layout_t geom)
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
void d_dx1_boundary_left(T* in, T* out, const cuda::bvals_t<T> bc, const cuda::slab_layout_t geom)
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
void d_dx1_boundary_right(T* in, T* out, const cuda::bvals_t<T> bc, const cuda::slab_layout_t geom)
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


template <typename T>
void dx_1(cuda_array_bc_nogp<T>& in, cuda_array_bc_nogp<T>& out, const size_t tlev,
             const cuda::slab_layout_t sl, const cuda::bvals_t<T> bc)
{
    cout << "Computing x derivative\n";

    // Size of the grid for boundary kernels in x-direction
    dim3 gridsize_line(int((sl.My + cuda::blockdim_row - 1) / cuda::blockdim_row));


    d_dx1_center<T> <<<in.get_grid(), in.get_block()>>>(in.get_array_d(tlev), out.get_array_d(0), bc, sl);
    d_dx1_boundary_left<<<gridsize_line, cuda::blockdim_row>>>(in.get_array_d(tlev), out.get_array_d(0), bc, sl);
    d_dx1_boundary_right<<<gridsize_line, cuda::blockdim_row>>>(in.get_array_d(tlev), out.get_array_d(0), bc, sl);
}


/*
 * Datatype that provides derivation routines and solver for QR factorization
 *
 */
template <typename T>
class derivs
{
    public:
        derivs(const cuda::slab_layout_t);
        ~derivs();

        void invert_laplace(cuda_array_bc_nogp<T>&, cuda_array_bc_nogp<T>&, const size_t t_src);

    private:
        const size_t Nx;
        const size_t My;
        const size_t My21;
        // Handles for cusparse library
        cusparseHandle_t cusparse_handle;
        cusparseMatDescr_t cusparse_descr;

        // Handles for cuBLAS library
        cublasHandle_t cublas_handle;

        // Matrix storage for solving tridiagonal equations
        CuCmplx<T>* d_diag;     // Main diagonal
        CuCmplx<T>* d_diag_l;   // lower diagonal
        CuCmplx<T>* d_diag_u;   // upper diagonal, for Laplace equatoin this is the same as the lower diagonal

        CuCmplx<T>* d_tmp_mat;  // workspace, used to transpose matrices for invert_laplace
};


template <typename T>
derivs<T> :: derivs(const cuda::slab_layout_t _sl) :
    Nx(_sl.Nx), My(_sl.My), My21(int(My) / 2 + 1),
    d_diag{nullptr}, d_diag_l{nullptr}, d_diag_u{nullptr}
{
    // Host copy of main and lower diagonal
    CuCmplx<T>* h_diag = new CuCmplx<T>[Nx];
    CuCmplx<T>* h_diag_l = new CuCmplx<T>[Nx];

    // Initialize cusparse
    cusparseStatus_t cusparse_status;
    if((cusparse_status = cusparseCreate(&cusparse_handle)) != CUSPARSE_STATUS_SUCCESS)
    {
        cerr << "Could not initialize CUSPARSE" << endl;
    }

    if((cusparse_status = cusparseCreateMatDescr(&cusparse_descr)) != CUSPARSE_STATUS_SUCCESS)
    {
        cerr << "Could not create CUSPARSE matrix description" << endl;
    }
    
    cusparseSetMatType(cusparse_descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(cusparse_descr, CUSPARSE_INDEX_BASE_ZERO);

    // Initialize cublas
    cublasStatus_t cublas_status;
    if((cublas_status = cublasCreate(&cublas_handle)) != CUBLAS_STATUS_SUCCESS)
    {
        cerr << "Could not initialize cuBLAS" << endl;
    }

    // Allocate memory for the lower and main diagonal for tridiagonal matrix factorization
    // The upper diagonal is equal to the lower diagonal
    gpuErrchk(cudaMalloc((void**) &d_diag, Nx * My21 * sizeof(CuCmplx<T>)));
    gpuErrchk(cudaMalloc((void**) &d_diag_l, Nx * My21 * sizeof(CuCmplx<T>)));
    gpuErrchk(cudaMalloc((void**) &d_tmp_mat, Nx * My21 * sizeof(CuCmplx<T>)));

    d_diag_u = d_diag_l;

    // Initialize the main diagonal separately for every ky
    T ky{0.0};
    for(size_t m = 0; m < My21; m++)
    {
        ky = cuda::TWOPI * double(m) / (_sl.y_lo + _sl.My * _sl.delta_y);
        for(size_t n = 0; n < Nx; n++)
        {
            h_diag[n] = -2.0 - _sl.delta_x * _sl.delta_x * ky * ky;
        }
        h_diag[0] = h_diag[0] - 1.0;
        h_diag[Nx - 1] = h_diag[Nx - 1] - 1.0;
        gpuErrchk(cudaMemcpy(d_diag + m * Nx, h_diag, Nx * sizeof(CuCmplx<T>), cudaMemcpyHostToDevice));
    }

    // Initialize the lower diagonal with all ones
    for(size_t n = 0; n < Nx; n++)
        h_diag_l[n] = 1.0;
    gpuErrchk(cudaMemcpy(d_diag_l, h_diag, Nx * sizeof(CuCmplx<T>), cudaMemcpyHostToDevice)); 

    /*
     * Check if the main diagonal elements are copied correctly
     * CuCmplx<T>* tmp = new CuCmplx<T>[Nx * My21];
     * gpuErrchk(cudaMemcpy(tmp, d_diag, Nx * My21 * sizeof(CuCmplx<T>), cudaMemcpyDeviceToHost));
     * for(size_t m = 0; m < My21; m++)
     * {
     *     for(size_t n = 0; n < Nx; n++)
     *     {
     *         cout << "m = " << m << " n = " << n << ": " << tmp[m * Nx + n] << endl;
     *     }
     * }
     * delete [] tmp;
     */

    delete [] h_diag_l;
    delete [] h_diag;
}


template <typename T>
derivs<T> :: ~derivs()
{
    cudaFree(d_tmp_mat);
    cudaFree(d_diag_l);
    cudaFree(d_diag);

    cublasDestroy(cublas_handle);
    cusparseDestroy(cusparse_handle);
}

template <typename T>
void derivs<T> :: invert_laplace(cuda_array_bc_nogp<T>& dst, cuda_array_bc_nogp<T>& src, const size_t t_src){
    cout << "Inverting Laplace" << endl;
    const int My21{src.get_my() / 2 + 1};

    // Copy data from src into dst. This is overwritten when calling cusparse<T>gtsv
    gpuErrchk(cudaMemcpy(dst.get_array_d(0), src.get_array_d(t_src), src.get_nx() * My21 * sizeof(CuCmplx<T>),
                         cudaMemcpyDeviceToDevice));
    // Call cuBLAS to transpose the memory.
    // cuda_array_bc_nogp stores the Fourier coefficients in each column consecutively.
    // For tridiagonal matrix factorization, cusparseZgtsvStridedBatch, the rows (approx. sol. at cell center)
    // need to be consecutive
    const cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
    const cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);
    cublasZgeam(cublas_handle,
                CUBLAS_OP_T, 
                CUBLAS_OP_N,
                My21, src.get_nx(),
                &alpha,
                (cuDoubleComplex*) dst.get_array_d(0),
                Nx,
                &beta,
                (cuDoubleComplex*) dst.get_array_d(0),
                Nx,
                (cuDoubleComplex*) dst.get_array_d(0),
                Nx);


    /*
    cusparseStatus_t 
        cusparseZgtsvStridedBatch(cusparseHandle_t handle, int m,         
                const cuDoubleComplex *dl, 
                const cuDoubleComplex  *d,  
                const cuDoubleComplex *du, cuDoubleComplex *x,     
                int batchCount, int batchStride)
    */

}

/******** temporary code used with cusolve library ******************
        // Provide CSR description of the 2d laplace matrix
        // See http://docs.nvidia.com/cuda/cusolver/index.html#format-csr
        // 1.) Device vectors 
        //int* d_csrRowPtrA;
        //int* d_csrColIndA;
        //T* d_csrValA;
        //T* d_b;
        //T* d_x;
        // 2.) Host vectors
        //int m;        // Dimension of the Matrix
        //int nnzA;     // Number of non-zero elements
        //int* csrRowPtrA;
        //int* csrColIndA;
        //T* csrValA;
        //T* b;
        //int batch_size;

        // Size information of QR factorization
        //size_t size_qr;
        //size_t size_internal;
        //void* buffer_qr;
};


//template <typename T>
//derivs<T> :: derivs(const cuda::slab_layout_t _sl) :
//    info{nullptr},
//    descrA{nullptr},
//    d_csrRowPtrA{nullptr}, d_csrColIndA{nullptr}, d_csrValA{nullptr}, d_b{nullptr}, d_x{nullptr},
//    size_qr{0}, size_internal{0}, buffer_qr{nullptr},
//    m{int(_sl.My)}, nnzA{3 * m}, batch_size{1},
//    csrRowPtrA{new int[nnzA]}, csrColIndA{new int[nnzA]}, csrValA{new T[nnzA]}, b{new T[_sl.My]}
//{
//    cout << "derivs<T> ::derivs" << endl;
//    // Initialize cusolver and create a handle for the cuSolver context
//    if (cusolverSpCreate(&cusolver_handle) != CUSOLVER_STATUS_SUCCESS)
//    {
//        cerr << "Error creating cusolver handle" << endl; 
//    };
//    // Initialize cusparse
//    if(cusparseCreateMatDescr(&descrA) != CUSPARSE_STATUS_SUCCESS)
//    {
//        cerr << "Error creating cusparse Matrix description" << endl;
//    }
//    // Set matrix type
//    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
//    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE);
//
//    if(cusolverSpCreateCsrqrInfo(&info) != CUSOLVER_STATUS_SUCCESS)
//    {
//        cerr << "Error creating QR solver info" << endl;
//    }
//
//}




//template <typename T>
//derivs<T> :: ~derivs()
//{
//    cout << "~derivs<T>" << endl;
//    cusolverSpDestroy(cusolver_handle);
//
//    // Free GPU arrays
//    //
//
//    // Free CPU arrays
//    delete [] b;
//    delete [] csrValA;
//    delete [] csrColIndA;
//    delete [] csrRowPtrA;
//}

**************** End of temporary code used with cusparse library *********/



//
//template <typename T>
//class derivs
//{
//    public:
//        derivs(const cuda::slab_layout_t);
//        ~derivs();
//
//        /// @brief Compute first order x- and y-derivatives
//        /// @detailed Allocates memory for Fourier coefficients.
//        /// @detailed If spectral representation is available, use
//        /// @d_dx1_dy1 where they are passed as arguments instead
//        void d_dx1_dy1(cuda_array<T>&, cuda_array<T>&, cuda_array<T>&);
//        void d_dx1_dy1(const cuda_array<CuCmplx<T> >&,  cuda_array<CuCmplx<T> >&, cuda_array<CuCmplx<T> >&, const uint);
//        void d_dx1_dy1(const cuda_array<CuCmplx<T> >*,  cuda_array<CuCmplx<T> >*, cuda_array<CuCmplx<T> >*, const uint);
//        /// @brief Compute second order x- and y-derivatives
//        void d_dx2_dy2(cuda_array<T>&, cuda_array<T>&, cuda_array<T>&);
//        void d_dx2_dy2(cuda_array<CuCmplx<T> >&,  cuda_array<CuCmplx<T> >&, cuda_array<CuCmplx<T> >&, const uint);
//        /// @brief Compute Laplacian
//        void d_laplace(cuda_array<T>&, cuda_array<T>&, const uint);
//        void d_laplace(cuda_array<CuCmplx<T> >&, cuda_array<CuCmplx<T> >&, const uint);
//        void d_laplace(cuda_array<CuCmplx<T> >*, cuda_array<CuCmplx<T> >*, const uint);
//        /// @brief Invert Laplace equation
//        void inv_laplace(cuda_array<T>&, cuda_array<T>&, const uint);
//        void inv_laplace(cuda_array<CuCmplx<T> >&, cuda_array<CuCmplx<T> >&, const uint);
//        void inv_laplace(cuda_array<CuCmplx<T> >*, cuda_array<CuCmplx<T> >*, const uint);
//
//        void dft_r2c(T* in, CuCmplx<T>* out);
//        void dft_c2r(CuCmplx<T>* in, T* out);
//
//    private:
//        const unsigned int Nx;
//        const unsigned int My;
//        const T Lx;
//        const T Ly;
//        const T dx;
//        const T dy;
//
//        dim3 grid_my_nx21;
//        dim3 block_my_nx21;
//
//        cuda_array<CuCmplx<T> > kmap_dx1_dy1;
//        cuda_array<CuCmplx<T> > kmap_dx2_dy2;
//
//        cufftHandle plan_r2c;
//        cufftHandle plan_c2r;
//
//        void init_dft();
//};
//
//
//#ifdef __CUDACC__
//
//template <typename T>
//derivs<T> :: derivs(const cuda::slab_layout_t sl) :
//    Nx(sl.Nx), My(sl.My),
//    Lx(T(sl.Nx) * T(sl.delta_x)),
//    Ly(T(sl.My) * T(sl.delta_y)),
//    dx(T(sl.delta_x)), dy(T(sl.delta_y)),
//    kmap_dx1_dy1(1, My, Nx / 2 + 1),
//    kmap_dx2_dy2(1, My, Nx / 2 + 1)
//{
//    init_dft();
//    // Generate first and second derivative map;
//    gen_kmap_dx1_dy1<<<kmap_dx1_dy1.get_grid(), kmap_dx1_dy1.get_block()>>>(kmap_dx1_dy1.get_array_d(), cuda::TWOPI / Lx,
//                                                                            cuda::TWOPI / Ly, My, Nx / 2 + 1);
//    gen_kmap_dx2_dy2<<<kmap_dx2_dy2.get_grid(), kmap_dx2_dy2.get_block()>>>(kmap_dx2_dy2.get_array_d(), cuda::TWOPI / Lx,
//                                                                            cuda::TWOPI / Ly, My, Nx / 2 + 1);
//    //ostream of;
//    //of.open("k2map.dat");
//    //of << kmap_dx2_dy2 << endl;
//    //of.close()
//    gpuStatus();
//}


#endif //DERIVATIVES_H
