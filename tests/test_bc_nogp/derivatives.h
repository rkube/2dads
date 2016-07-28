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
        const cuda::slab_layout_t sl;
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
    Nx(_sl.Nx), My(_sl.My), My21(static_cast<int>(My / 2 + 1)), sl(_sl),
    d_diag{nullptr}, d_diag_l{nullptr}, d_diag_u{nullptr}
{
    // Host copy of main and lower diagonal
    CuCmplx<T>* h_diag = new CuCmplx<T>[Nx];
    CuCmplx<T>* h_diag_u = new CuCmplx<T>[Nx];
    CuCmplx<T>* h_diag_l = new CuCmplx<T>[Nx];

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

    // Initialize cublas
    cublasStatus_t cublas_status;
    if((cublas_status = cublasCreate(&cublas_handle)) != CUBLAS_STATUS_SUCCESS)
    {
        throw cublas_err(cublas_status);
    }

    // Allocate memory for temporary matrix storage
    gpuErrchk(cudaMalloc((void**) &d_tmp_mat, Nx * My21 * sizeof(CuCmplx<T>)));

    // Allocate memory for the lower and main diagonal for tridiagonal matrix factorization
    // The upper diagonal is equal to the lower diagonal
    gpuErrchk(cudaMalloc((void**) &d_diag, Nx * My21 * sizeof(CuCmplx<T>)));
    gpuErrchk(cudaMalloc((void**) &d_diag_u, Nx * My21 * sizeof(CuCmplx<T>)));
    gpuErrchk(cudaMalloc((void**) &d_diag_l, Nx * My21 * sizeof(CuCmplx<T>)));

    T ky2{0.0};                             // ky^2
    const T inv_dx{1.0 / sl.delta_x};      // 1 / delta_x
    const T inv_dx2{inv_dx * inv_dx};       // 1 / delta_x^2
    const T Ly{static_cast<T>(sl.My) * sl.delta_y};

    // Initialize the main diagonal separately for every ky
    for(size_t m = 0; m < My21; m++)
    {
        ky2 = cuda::TWOPI * cuda::TWOPI * static_cast<T>(m * m) / (Ly * Ly);
        for(size_t n = 0; n < Nx; n++)
        {
            h_diag[n] = -2.0 * inv_dx2 - ky2;
        }
        h_diag[0] = h_diag[0] - inv_dx2;
        h_diag[Nx - 1] = h_diag[Nx - 1] - inv_dx2;
        gpuErrchk(cudaMemcpy(d_diag + m * Nx, h_diag, Nx * sizeof(CuCmplx<T>), cudaMemcpyHostToDevice));
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

    // Paste copies of this vector together (required by cusparseZgtsvStridedBatch
    for(size_t m = 0; m < My21; m++)
    {
        gpuErrchk(cudaMemcpy(d_diag_l + m * Nx, h_diag_l, Nx * sizeof(CuCmplx<T>), cudaMemcpyHostToDevice)); 
        gpuErrchk(cudaMemcpy(d_diag_u + m * Nx, h_diag_u, Nx * sizeof(CuCmplx<T>), cudaMemcpyHostToDevice)); 
    }

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
    delete [] h_diag_u;
    delete [] h_diag;
}


template <typename T>
derivs<T> :: ~derivs()
{
    cudaFree(d_tmp_mat);
    cudaFree(d_diag_l);
    cudaFree(d_diag_u);
    cudaFree(d_diag);

    cublasDestroy(cublas_handle);
    cusparseDestroy(cusparse_handle);
}

template <typename T>
void derivs<T> :: invert_laplace(cuda_array_bc_nogp<T>& dst, cuda_array_bc_nogp<T>& src, const size_t t_src){
    // Safe casts because I am feeling fancy today :)
    const int My_int{static_cast<int>(src.get_my())};
    const int My21_int{static_cast<int>((src.get_my() + src.get_geom().pad_y) / 2)};
    const int Nx_int{static_cast<int>(src.get_nx())};

    const cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
    const cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);
    cublasStatus_t cublas_status;
    cusparseStatus_t cusparse_status;

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
     * T tmp_arr[nelem];
     * gpuErrchk(cudaMemcpy(tmp_arr, src.get_array_d(0), nelem * sizeof(T), cudaMemcpyDeviceToHost));
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
                                    (cuDoubleComplex*) src.get_array_d(0), My21_int,
                                    &beta,
                                    nullptr, Nx_int,
                                    (cuDoubleComplex*) d_tmp_mat, Nx_int
                                    )) != CUBLAS_STATUS_SUCCESS)
    {
        cout << "cublas_status: " << cublas_status << endl;
        throw cublas_err(cublas_status);
    }
    cout << "cublas_status: " << cublas_status << endl;

    /* Write output of cublasZgeam to file for debugging purposes
    gpuErrchk(cudaMemcpy(tmp_arr, d_tmp_mat, nelem * sizeof(T), cudaMemcpyDeviceToHost));
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
    
    //gpuErrchk(cudaMemcpy(dst.get_array_d(0), d_tmp_mat, src.get_nx() * My21 * sizeof(CuCmplx<T>),
    //                     cudaMemcpyDeviceToDevice));


    // Next step: Solve the tridiagonal system

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
    cout << "cusparse_status: " << cusparse_status << endl;
    // Convert d_tmp_mat from column-major to row-major order
    if((cublas_status = cublasZgeam(cublas_handle,
                                    CUBLAS_OP_T, CUBLAS_OP_N,
                                    My21_int, Nx_int,
                                    &alpha,
                                    (cuDoubleComplex*) d_tmp_mat, Nx_int,
                                    &beta,
                                    nullptr, My21_int,
                                    (cuDoubleComplex*) dst.get_array_d(0), My21_int
                                    )) != CUBLAS_STATUS_SUCCESS)
    {
        cout << cublas_status << endl;
        throw cublas_err(cublas_status);
    }
    //gpuErrchk(cudaMemcpy(dst.get_array_d(0), d_tmp_mat, src.get_nx() * My21 * sizeof(CuCmplx<T>), cudaMemcpyDeviceToDevice));
    cout << "cublas_status: " << cublas_status << endl;


#ifdef DEBUG
    // Test the precision of the solution

    // 1.) Build the laplace matrix in csr form
    // Host data
    size_t nnz{Nx + 2 * (Nx - 1)};    // Main diagonal plus 2 side diagonals
    CuCmplx<T>* csrValA_h = new CuCmplx<T>[nnz];  
    int* csrRowPtrA_h = new int[Nx + 1];
    int* csrColIndA_h = new int[nnz];

    // Input columns
    CuCmplx<T>* h_inp_mat_col = new CuCmplx<T>[Nx];
    // Result columns
    CuCmplx<T>* h_tmp_mat_col = new CuCmplx<T>[Nx];

    // Device data
    CuCmplx<T>* csrValA_d{nullptr};
    int* csrRowPtrA_d{nullptr}; 
    int* csrColIndA_d{nullptr};

    CuCmplx<T>* d_inp_mat{nullptr};

    // Some constants we need later on
    const T inv_dx2 = 1.0 / (sl.delta_x * sl.delta_x);
    const CuCmplx<T> inv_dx2_cmplx = CuCmplx<T>(inv_dx2);

    gpuErrchk(cudaMalloc((void**) &csrValA_d, nnz * sizeof(CuCmplx<T>)));
    gpuErrchk(cudaMalloc((void**) &csrRowPtrA_d, (Nx + 1) * sizeof(int)));
    gpuErrchk(cudaMalloc((void**) &csrColIndA_d, nnz * sizeof(int)));
    gpuErrchk(cudaMalloc((void**) &d_inp_mat, Nx * My21 * sizeof(CuCmplx<T>)));

    // Build Laplace matrix structure: ColIndA and RowPtrA
    // Matrix values on main diagonal are later updated individually for each ky mode.
    // Side bands (upper/lower diagonal) are computed once here.
    csrColIndA_h[0] = 0;
    csrColIndA_h[1] = 0;

    csrRowPtrA_h[0] = 0;
    csrValA_h[1] = inv_dx2_cmplx;

    for(size_t n = 2; n < nnz - 3; n += 3)
    {
        csrColIndA_h[n    ] = static_cast<int>((n - 2) / 3);
        csrColIndA_h[n + 1] = static_cast<int>((n - 2) / 3) + 1;
        csrColIndA_h[n + 2] = static_cast<int>((n - 2) / 3) + 2;

        csrRowPtrA_h[(n - 2) / 3 + 1] = n; 

        csrValA_h[n] = inv_dx2_cmplx;
        csrValA_h[n + 2] = inv_dx2_cmplx;
    }   

    csrColIndA_h[nnz - 2] = Nx - 2;
    csrColIndA_h[nnz - 1] = Nx - 1;  

    csrRowPtrA_h[Nx - 1] = nnz;

    csrValA_h[nnz - 2] = CuCmplx<T>(inv_dx2);

    gpuErrchk(cudaMemcpy(csrRowPtrA_d, csrRowPtrA_h, (Nx + 1) * sizeof(int), cudaMemcpyHostToDevice)); 
    gpuErrchk(cudaMemcpy(csrColIndA_d, csrColIndA_h, nnz * sizeof(int), cudaMemcpyHostToDevice));

    cusparseMatDescr_t mat_type;
    cusparseCreateMatDescr(&mat_type); 

    
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
                                    (cuDoubleComplex*) src.get_array_d(0), My21_int,
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
    CuCmplx<T>* row_ptr_d_inp{nullptr};
    CuCmplx<T>* row_ptr_d_tmp{nullptr};

    T ky2{0.0};
    const T Ly{static_cast<T>(sl.My) * sl.delta_y};
    for(size_t m = 0; m < My21; m++)
    {
        row_ptr_d_inp = d_inp_mat + m * Nx;
        row_ptr_d_tmp = d_tmp_mat + m * Nx;

        // Copy reference data in h_inp_mat
        gpuErrchk(cudaMemcpy(h_inp_mat_col, row_ptr_d_inp, Nx * sizeof(CuCmplx<T>), cudaMemcpyDeviceToHost));
        //for(size_t n = 0; n < Nx; n++)
        //{
        //    cout << n << ":\t h_inp_mat_col = " << h_inp_mat_col[n] << endl;
        //}

        // Update values of csrValA_h
        // Assume Dirichlet=0 boundary conditions for now
        ky2 = cuda::TWOPI * cuda::TWOPI * static_cast<T>(m * m) / (Ly * Ly);
        csrValA_h[0] = CuCmplx<T>(-3.0 * inv_dx2 - ky2);
        //cout << csrValA_h[0] << "\t" << csrValA_h[1] << "\t";
        for(size_t n = 2; n < nnz - 3; n += 3)
        {
            csrValA_h[n + 1] = CuCmplx<T>(-2.0 * inv_dx2 - ky2);
            //cout << csrValA_h[n] << "\t" << csrValA_h[n + 1] << "\t" << csrValA_h[n + 1] << "\t";
        }
        csrValA_h[nnz - 1] = CuCmplx<T>(-3.0 * inv_dx2 - ky2);
        //cout << csrValA_h[nnz - 2] << "\t" << csrValA_h[nnz - 1] << "\t";

        gpuErrchk(cudaMemcpy(csrValA_d, csrValA_h, nnz * sizeof(CuCmplx<T>), cudaMemcpyHostToDevice));

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
        gpuErrchk(cudaMemcpy(h_tmp_mat_col, row_ptr_d_inp, Nx * sizeof(CuCmplx<T>), cudaMemcpyDeviceToHost));

        // Compute L2 distance between h_inp_mat and h_tmp_mat
      T L2_norm{0.0};
      //CuCmplx<T> tmp(0.0, 0.0);
      for(size_t n = 0; n < Nx; n++)
        {
            //cout << n << ": h_inp_mat_col = " << h_inp_mat_col[n] << "\th_tmp_mat_col = " << h_tmp_mat_col[n] << endl;
            L2_norm += ((h_inp_mat_col[n] - h_tmp_mat_col[n]) * (h_inp_mat_col[n] - h_tmp_mat_col[n])).abs();
        }
        L2_norm = sqrt(L2_norm);
        cout << m << ": ky = " << sqrt(ky2) << "\t L2 = " << L2_norm << endl;
    }
     

    // 4.) Clean up sparse storage
    cudaFree(d_inp_mat);
    cudaFree(csrColIndA_d);
    cudaFree(csrRowPtrA_d);
    cudaFree(csrValA_d);
    
    // Input columns
    delete [] h_inp_mat_col;
    delete [] h_tmp_mat_col;

    delete [] csrColIndA_h;
    delete [] csrRowPtrA_h;
    delete [] csrValA_h;
#endif


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
