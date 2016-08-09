/*
 * Solver for elliptical problems
 */

#include <iostream>
#include <cusolverSp.h>
#include <cublas_v2.h>
#include "error.h"

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

    // Wrapper for cusparseMatDescr
    //class cusparse_matrix_desc_t
    //{
    //    private:
    //        cusparse_matrix_descr_t()
    //        {
    //            cusparseStatus_t cusparse_status;
    //            if((cusparse_status = cusparseCreateMatDescr(&cusparse_descr)) != CUSPARSE_STATUS_SUCCESS)
    //            {
    //                throw cusparse_err(cusparse_status);
    //            }
    //        };

    //        ~cusparse_matrix_desc_t(){};

    //}


    class elliptical
    {
        public:
            elliptical(){};
            ~elliptical(){};    

        private:
    };
}

#endif // SOLVERS_H

// End of file solvers.h
