/*
 * Solver for elliptical problems
 */

#include <iostream>
#include <sstream>
#include <cusolverSp.h>
#include <cublas_v2.h>
#include "cucmplx.h"
#include "error.h"
#include "2dads_types.h"
#include "cuda_types.h"

#ifndef SOLVERS_H
#define SOLVERS_H

// Structures for solver libraries
// 
// Handles to cublas and cusparse are wrapped in static objects, static factory pattern
// https://github.com/botonchou/libcumatrix/blob/master/include/device_matrix.h
// http://stackoverflow.com/questions/2062560/what-is-the-use-of-making-constructor-private-in-a-class
//
namespace solvers
{
    // Wrapper data type for cublasHandle_t
    class cublas_handle_t
    {
        private: 
            cublas_handle_t()
            {
                std::cout << "Creating cublas_handle" << std::endl;
                cublasStatus_t cublas_status;
                if((cublas_status = cublasCreate(&cublas_handle)) != CUBLAS_STATUS_SUCCESS)
                {
                    throw cublas_err(cublas_status);
                }
            };

            ~cublas_handle_t() {cublasDestroy(cublas_handle);};
            cublas_handle_t(const cublas_handle_t& source);
            cublas_handle_t& operator= (const cublas_handle_t rhs);
            cublasHandle_t cublas_handle;

        public:
            static cublasHandle_t& get_handle() 
            {
                static cublas_handle_t h;
                return (h.cublas_handle);
            }

    };

    // Wrapper data type for cusparseHandle_t
    class cusparse_handle_t
    {
        private: 
            cusparse_handle_t()
            {
                std::cout << "Initializing cusparse " << cusparse_handle << std::endl;
                cusparseStatus_t cusparse_status;
                if((cusparse_status = cusparseCreate(&cusparse_handle)) != CUSPARSE_STATUS_SUCCESS)
                {
                    throw cusparse_err(cusparse_status);
                }
            };

            ~cusparse_handle_t() { cusparseDestroy(cusparse_handle); };
            cusparse_handle_t(const cusparse_handle_t& source);
            cusparse_handle_t& operator= (const cusparse_handle_t rhs);
            cusparseHandle_t cusparse_handle;

        public :
            static cusparseHandle_t& get_handle() 
            {
                static cusparse_handle_t h;
                return (h.cusparse_handle);
            }
    };


    class elliptic
    {
        private:
            const int My_int;
            const int My21_int;
            const int Nx_int;
            const double inv_dx2;
            cuDoubleComplex* d_tmp_mat;

        public:
            elliptic(const twodads::slab_layout_t _geom) : My_int(static_cast<int>(_geom.get_my())),
                                                           My21_int{static_cast<int>((_geom.get_my() + _geom.get_pad_y()) / 2)},
                                                           Nx_int{static_cast<int>(_geom.get_nx())},
                                                           inv_dx2{1.0 / (_geom.get_deltax() * _geom.get_deltax())}
             {
                cudaError_t err;
                if( (err = cudaMalloc((void**) &d_tmp_mat, get_nx() * get_my21() * sizeof(cuDoubleComplex))) != cudaSuccess)
                {
                    std::cerr << "elliptic::elliptic: Failed to allocate " << get_nx() * get_my21() * sizeof(cuDoubleComplex) << " bytes" << std::endl;
                }
            };


            ~elliptic()
            {
                cudaFree(get_d_tmp_mat());
            };    

            inline int get_my() const {return(My_int);};
            inline int get_my21() const {return(My21_int);};
            inline int get_nx() const {return(Nx_int);};
            inline double get_invdx2() const {return(inv_dx2);};
            inline cuDoubleComplex* get_d_tmp_mat() {return(d_tmp_mat);};


            void solve(cuDoubleComplex* dst, cuDoubleComplex* src,
                       cuDoubleComplex* d_diag_l, cuDoubleComplex* d_diag, cuDoubleComplex* d_diag_u)

            {
                const cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
                const cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);

                cublasStatus_t cublas_status;
                cusparseStatus_t cusparse_status;

                // Transpose matrix
                if((cublas_status = cublasZgeam(solvers::cublas_handle_t::get_handle(),
                                                CUBLAS_OP_T, CUBLAS_OP_N,
                                                get_nx(), get_my21(),
                                                &alpha,
                                                src,
                                                get_my21(),
                                                &beta,
                                                nullptr,
                                                get_nx(), 
                                                get_d_tmp_mat(),
                                                get_nx())) != CUBLAS_STATUS_SUCCESS)
                {
                    throw cublas_err(cublas_status);
                }
                // Solve banded system 
                if((cusparse_status = cusparseZgtsvStridedBatch(solvers::cusparse_handle_t::get_handle(),
                                                                get_nx(),
                                                                d_diag_l,
                                                                d_diag,
                                                                d_diag_l,
                                                                get_d_tmp_mat(),
                                                                get_my21(),
                                                                get_nx())) != CUSPARSE_STATUS_SUCCESS)
                {
                    throw cusparse_err(cusparse_status);
                }

                // Tranpose back
                if((cublas_status = cublasZgeam(solvers::cublas_handle_t::get_handle(),
                                                CUBLAS_OP_T, 
                                                CUBLAS_OP_N,
                                                get_my21(),
                                                get_nx(),
                                                &alpha,
                                                get_d_tmp_mat(),
                                                get_nx(),
                                                &beta,
                                                nullptr, 
                                                get_my21(),
                                                dst,
                                                get_my21())) != CUBLAS_STATUS_SUCCESS)
                {
                    throw cublas_err(cublas_status);
                }
                std::cout << "Iam in solvers::elliptic" << std::endl;
            }
    };
}

#endif // SOLVERS_H

// End of file solvers.h
