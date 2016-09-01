/*
 * Solver for elliptical problems
 */

#include <iostream>
#include <sstream>
#include <string.h>

#include "2dads_types.h"
#include "error.h"
#include "cucmplx.h"

#ifdef HOST
#include "mkl.h"
#endif //HOST

#ifdef __CUDACC__
#include <cusolverSp.h>
#include <cublas_v2.h>
#include "cuda_types.h"
#endif //__CUDACC__

#ifndef SOLVERS_H
#define SOLVERS_H

// Structures for solver libraries
// 
// Handles to cublas and cusparse are wrapped in static objects, static factory pattern
// https://github.com/botonchou/libcumatrix/blob/master/include/device_matrix.h
// http://stackoverflow.com/questions/2062560/what-is-the-use-of-making-constructor-private-in-a-class
//
//
// elliptic_base_t is defined as an abstract base class that just defines the sizes
// of the linear system. The call to the actual solver routines is defined in the
// solve member of the derived classes.
// Implementations
//
// * MKL (zgtsv)
// * Numerical recipies
// * cuSparse (zgtsv)

namespace solvers
{
    // Wrapper data type for cublasHandle_t

#ifdef __CUDACC__
    class cublas_handle_t
    {
        private: 
            cublas_handle_t()
            {
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


#endif //__CUDAC__

    class elliptic_base_t
    {
        public:
            elliptic_base_t(const twodads::slab_layout_t _geom) : My_int{static_cast<int>(_geom.get_my())},
                                                                  My21_int{static_cast<int>((_geom.get_my() + _geom.get_pad_y()) / 2)},
                                                                  Nx_int{static_cast<int>(_geom.get_nx())}
            {}
            virtual ~elliptic_base_t() {}

            virtual void solve(CuCmplx<twodads::real_t>*, 
                               CuCmplx<twodads::real_t>*, 
                               CuCmplx<twodads::real_t>*, 
                               CuCmplx<twodads::real_t>*, 
                               CuCmplx<twodads::real_t>*) = 0;

            int get_my_int() const {return(My_int);};
            int get_my21_int() const {return(My21_int);};
            int get_nx_int() const {return(Nx_int);};
        private:
            const int My_int;
            const int My21_int;
            const int Nx_int;
    };


#ifdef __CUDACC__
    class elliptic_cublas_t : public elliptic_base_t
    {
        using elliptic_base_t :: get_my_int;
        using elliptic_base_t :: get_my21_int;
        using elliptic_base_t :: get_nx_int;

        private:
            cuDoubleComplex* d_tmp_mat;

        public:
            elliptic_cublas_t(const twodads::slab_layout_t _geom) : elliptic_base_t(_geom)
             {
                cudaError_t err;
                if( (err = cudaMalloc((void**) &d_tmp_mat, static_cast<size_t>(get_nx_int() * get_my21_int()) * sizeof(cuDoubleComplex))) != cudaSuccess)
                {
                    std::cerr << "elliptic::elliptic: Failed to allocate " << static_cast<size_t>(get_nx_int() * get_my21_int()) * sizeof(cuDoubleComplex) << " bytes" << std::endl;
                }
            };


            ~elliptic_cublas_t()
            {
                cudaFree(get_d_tmp_mat());
            };    

            inline cuDoubleComplex* get_d_tmp_mat() {return(d_tmp_mat);};

            //void solve(cuDoubleComplex* src, cuDoubleComplex* dst,
            //           cuDoubleComplex* d_diag_l, cuDoubleComplex* d_diag, cuDoubleComplex* d_diag_u)
            virtual void solve(CuCmplx<twodads::real_t>* dummy_src, 
                               CuCmplx<twodads::real_t>* dummy_dst,
                               CuCmplx<twodads::real_t>* dummy_diag_l, 
                               CuCmplx<twodads::real_t>* dummy_diag, 
                               CuCmplx<twodads::real_t>* dummy_diag_u)
            {
                cuDoubleComplex* src = reinterpret_cast<cuDoubleComplex*>(dummy_src);
                cuDoubleComplex* dst = reinterpret_cast<cuDoubleComplex*>(dummy_dst);
                cuDoubleComplex* diag_l = reinterpret_cast<cuDoubleComplex*>(dummy_diag_l);
                cuDoubleComplex* diag = reinterpret_cast<cuDoubleComplex*>(dummy_diag);
                cuDoubleComplex* diag_u = reinterpret_cast<cuDoubleComplex*>(dummy_diag_u);

                const cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
                const cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);

                cublasStatus_t cublas_status;
                cusparseStatus_t cusparse_status;

                // Transpose matrix
                if((cublas_status = cublasZgeam(solvers::cublas_handle_t::get_handle(),
                                                CUBLAS_OP_T, CUBLAS_OP_N,
                                                get_nx_int(), get_my21_int(),
                                                &alpha,
                                                src,
                                                get_my21_int(),
                                                &beta,
                                                nullptr,
                                                get_nx_int(), 
                                                get_d_tmp_mat(),
                                                get_nx_int())) != CUBLAS_STATUS_SUCCESS)
                {
                    throw cublas_err(cublas_status);
                }

                // Solve banded system 
                if((cusparse_status = cusparseZgtsvStridedBatch(solvers::cusparse_handle_t::get_handle(),
                                                                get_nx_int(),
                                                                diag_l,
                                                                diag,
                                                                diag_u,
                                                                get_d_tmp_mat(),
                                                                get_my21_int(),
                                                                get_nx_int())) != CUSPARSE_STATUS_SUCCESS)
                {
                    throw cusparse_err(cusparse_status);
                }

                // Tranpose back
                if((cublas_status = cublasZgeam(solvers::cublas_handle_t::get_handle(),
                                                CUBLAS_OP_T, 
                                                CUBLAS_OP_N,
                                                get_my21_int(),
                                                get_nx_int(),
                                                &alpha,
                                                get_d_tmp_mat(),
                                                get_nx_int(),
                                                &beta,
                                                nullptr, 
                                                get_my21_int(),
                                                dst,
                                                get_my21_int())) != CUBLAS_STATUS_SUCCESS)
                {
                    throw cublas_err(cublas_status);
                }
            }
    };

#endif //__CUDACC__

#ifndef __CUDACC__
    class elliptic_mkl_t : public elliptic_base_t
    // Class wrapper for zgtsv routine
    {
        using elliptic_base_t :: get_my_int;
        using elliptic_base_t :: get_my21_int;
        using elliptic_base_t :: get_nx_int;

        public:
            elliptic_mkl_t(const twodads::slab_layout_t& _geom) : elliptic_base_t(_geom) 
            {};
            
            virtual void solve(CuCmplx<twodads::real_t>* dummy, 
                               CuCmplx<twodads::real_t>* dummy_dst,
                               CuCmplx<twodads::real_t>* dummy_diag_l, 
                               CuCmplx<twodads::real_t>* dummy_diag, 
                               CuCmplx<twodads::real_t>* dummy_diag_u)
                       {
                           lapack_complex_double* dst = reinterpret_cast<lapack_complex_double*>(dummy_dst);
                           lapack_complex_double* diag = reinterpret_cast<lapack_complex_double*>(dummy_diag);
                           lapack_complex_double* diag_u = reinterpret_cast<lapack_complex_double*>(dummy_diag_u);
                           lapack_complex_double* diag_l = reinterpret_cast<lapack_complex_double*>(dummy_diag_l);
                           // In contrast to the cublas library, it accepts the input in row-major
                           // format. Thus do not transpose but solve directly.
                            
                            // Temporary copy of the diagonals. They get overwritten when calling LAPACKE_zgtsv
                            // Update the diagonal values into the dummy copies in each iteration of the solver.
                            lapack_complex_double* diag_l_copy = new lapack_complex_double[get_nx_int()];
                            lapack_complex_double* diag_u_copy = new lapack_complex_double[get_nx_int()];
                            lapack_complex_double* diag_copy = new lapack_complex_double[get_nx_int()];

                            for(size_t m = 0; m < static_cast<size_t>(get_my21_int()); m++)
                            { 
                                lapack_int res{0};
                                memcpy(diag_l_copy, diag_l, static_cast<size_t>(get_nx_int()) * sizeof(lapack_complex_double));
                                memcpy(diag_u_copy, diag_u, static_cast<size_t>(get_nx_int()) * sizeof(lapack_complex_double));
                                memcpy(diag_copy, diag + m * static_cast<size_t>(get_nx_int()), static_cast<size_t>(get_nx_int()) * sizeof(lapack_complex_double));

                                if((res = LAPACKE_zgtsv(LAPACK_ROW_MAJOR,
                                                        get_nx_int(),
                                                        1, 
                                                        diag_l_copy,
                                                        diag_copy,
                                                        diag_u_copy,
                                                        dst + m, 
                                                        get_my21_int())) != 0)
                                {
                                    std::cerr << "Error from MKL LAPACK_zgtsv: " << res << std::endl;
                                }
                            } 
                            delete [] diag_copy;
                            delete [] diag_u_copy;
                            delete [] diag_l_copy;
                       }
    };
#endif //__CUDACC__

    // Implementation of tridiagonal solver from numerical recipes
    // $2.4, p.53ff
    class elliptic_nr_t : public elliptic_base_t
    {
        using elliptic_base_t :: get_my_int;
        using elliptic_base_t :: get_my21_int;
        using elliptic_base_t :: get_nx_int;

        public:
            elliptic_nr_t(const twodads::slab_layout_t& _geom) : elliptic_base_t(_geom)
            {};

            virtual void solve(CuCmplx<twodads::real_t>* src, CuCmplx<twodads::real_t>* dst,
                       CuCmplx<twodads::real_t>* diag_l, CuCmplx<twodads::real_t>* diag, CuCmplx<twodads::real_t>* diag_u)
            {
                // Pointers to start of the current system, nomenclature see numerical recipes
                CuCmplx<twodads::real_t>* a{nullptr};
                CuCmplx<twodads::real_t>* b{nullptr};
                CuCmplx<twodads::real_t>* c{nullptr};
                CuCmplx<twodads::real_t>* u{nullptr};
                CuCmplx<twodads::real_t>* r{nullptr};

                size_t j{0};
                CuCmplx<twodads::real_t> beta;
                std::vector<CuCmplx<twodads::real_t>> gamma(get_nx_int());

                for(size_t m = 0; m < static_cast<size_t>(get_my21_int()); m++)
                {
                    a = diag_l + m * get_nx_int();
                    b = diag + m * get_nx_int();
                    c = diag_u + m * get_nx_int();
                    u = dst + m * get_nx_int();
                    r = src + m * get_nx_int();

                    u[0] = r[0] / (beta = b[0]);
                    for(j = 1; j < static_cast<size_t>(get_nx_int()); j++)
                    {
                        gamma[j] = c[j - 1] / beta;
                        beta = b[j] - a[j] * gamma[j];
                        if(beta.abs() < twodads::epsilon)
                            throw numerics_error("Error 2 in elliptic_nr_t :: solve");
                        u[j] = (r[j] - a[j] * u[j-1]) / beta;
                    }
                    for(int j = get_nx_int() - 2; j >= 0; j--)
                    {
                        u[j] -= gamma[j + 1] * u[j + 1];
                    }
                }
            }
    };

}

#endif // SOLVERS_H

// End of file solvers.h