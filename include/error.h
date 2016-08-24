/*
 *  error.h
 *  2dads-oo
 *
 *  Created by Ralph Kube on 20.01.11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef ERROR_H
#define ERROR_H

#include <string>
#include <exception>
#include <map>

#ifdef __CUDACC__
#include "cublas_v2.h"
#include "cusparse.h"
#endif //__CUDACC__




#ifdef __CUDACC__
class cublas_err : public std::exception
{
    public:
        /// Error code encoded as string
        cublas_err(const cublasStatus_t err_code) : error(cublas_status_str.at(err_code)) {}; 
        ~cublas_err() throw() {};
        virtual const char* what() const throw() {return error.data();};
    private:
        std::map<cublasStatus_t, std::string> cublas_status_str{
            {CUBLAS_STATUS_SUCCESS, std::string("CUBLAS_STATUS_SUCCESS")},
            {CUBLAS_STATUS_NOT_INITIALIZED, std::string("CUBLAS_STATUS_NOT_INITIALIZED")},
            {CUBLAS_STATUS_ALLOC_FAILED, std::string("CUBLAS_STATUS_ALLOC_FAILED")},
            {CUBLAS_STATUS_INVALID_VALUE, std::string("CUBLAS_STATUS_INVALID_VALUE")},
            {CUBLAS_STATUS_ARCH_MISMATCH, std::string("CUBLAS_STATUS_ARCH_MISMATCH")}, 
            {CUBLAS_STATUS_MAPPING_ERROR, std::string("CUBLAS_STATUS_MAPPING_ERROR")},
            {CUBLAS_STATUS_EXECUTION_FAILED, std::string("CUBLAS_STATUS_EXECUTION_FAILED")},
            {CUBLAS_STATUS_INTERNAL_ERROR, std::string("CUBLAS_STATUS_INTERNAL_ERROR")},
            {CUBLAS_STATUS_NOT_SUPPORTED, std::string("CUBLAS_STATUS_NOT_SUPPORTED")},
            {CUBLAS_STATUS_LICENSE_ERROR, std::string("CUBLAS_STATUS_LICENSE_ERROR")}
        };

        const std::string error;
};


class cusparse_err : public std::exception 
{
    public:
        /// Error code encoded as string
        cusparse_err(const cusparseStatus_t err_code) : error(cusparse_status_str.at(err_code)) {};
        ~cusparse_err() throw() {};
        virtual const char* what() const throw() {return error.data();};
    private:
        std::map<cusparseStatus_t, std::string> cusparse_status_str{
            {CUSPARSE_STATUS_SUCCESS, std::string("CUSPARSE_STATUS_SUCCESS")},
            {CUSPARSE_STATUS_NOT_INITIALIZED, std::string("CUSPARSE_STATUS_NOT_INITIALIZED")},
            {CUSPARSE_STATUS_ALLOC_FAILED, std::string("CUSPARSE_STATUS_ALLOC_FAILED")},
            {CUSPARSE_STATUS_INVALID_VALUE, std::string("CUSPARSE_STATUS_INVALID_VALUE")},
            {CUSPARSE_STATUS_ARCH_MISMATCH, std::string("CUSPARSE_STATUS_ARCH_MISMATCH")},
            {CUSPARSE_STATUS_MAPPING_ERROR, std::string("CUSPARSE_STATUS_MAPPING_ERROR")}, 
            {CUSPARSE_STATUS_EXECUTION_FAILED, std::string("CUSPARSE_STATUS_EXECUTION_FAILED")},
            {CUSPARSE_STATUS_INTERNAL_ERROR, std::string("CUSPARSE_STATUS_INTERNAL_ERROR")},
            {CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED, std::string("CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED")}
        };

        const std::string error;
};


/// GPU call errorss
class gpu_error : public std::exception
{
    public:
        gpu_error(const std::string& err) : error(err) {};
        ~gpu_error() throw() {};
        virtual const char* what() const throw() {return error.data();};

    private:
        const std::string error;
};

#endif //__CUDACC__

class out_of_bounds_err : public std::exception
{
    public:
        /// Error code encoded as a string
        out_of_bounds_err(const std::string& err) : error(err) {};
        ~out_of_bounds_err() throw() {};
        virtual const char* what() const throw() {return error.data();};	

    private:
        const std::string error;
};


/// Memory error in operators
class operator_err : public std::exception
{
    public:
        operator_err(const std::string& err) : error(err) {};
        ~operator_err() throw() {};
        virtual const char* what() const throw() {return error.data();};

    private:
        const std::string error;
};


/// Invalid size for array construction
class invalid_size : public std::exception
{
    public:
        /// Write error message 
        invalid_size(const std::string& err) : error(err) {};
        ~invalid_size() throw() {};
        virtual const char* what() const throw() {return error.data();};

    private:
        //int min_size;
        //int max_size;
        //int selected_size;
        const std::string error;
};


/// Configuration file errors
class config_error : public std::exception{
    public:
        config_error(const std::string& err) : error(err) {};
        ~config_error() throw() {};
        virtual const char* what() const throw() {return error.data();};

    private:
        const std::string error;
};


/// Errors from the diagnostic unit
class diagnostics_error : public std::exception
{
    public:
        diagnostics_error(const std::string& err) : error(err) {};
        ~diagnostics_error() throw() {};
        virtual const char* what() const throw() {return error.data();};

    private:
        const std::string error;
};


/// Invalid field name
class name_error : public std::exception
{
    public:
        name_error(const std::string& err) : error(err) {};
        ~name_error() throw() {};
        virtual const char* what() const throw() {return error.data();};

    private:
        const std::string error;
};




/// things not implemented yet
class not_implemented_error : public std::exception
{
    public:
        not_implemented_error(const std::string& err) : error(err) {};
        ~not_implemented_error() throw() {};
        virtual const char* what() const throw() {return error.data();};

        private:
            const std::string error;
};


#endif //__ERROR_H
