/*
 *  error.h
 *  2dads-oo
 *
 *  Created by Ralph Kube on 20.01.11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

//#include "2dads.h"
#include <string>
#include <exception>

#ifndef ERROR_H
#define ERROR_H

/// Out of bounds error code: Attempted access for element out of array bounds

using namespace std;

class out_of_bounds_err : public exception
{
    public:
        /// Error code encoded as a string
        out_of_bounds_err(const string err) : error(err) {};
        ~out_of_bounds_err() throw() {};
        virtual const char* what() const throw() {return error.data();};	

    private:
        string error;
};


/// Memory error in operators
class operator_err : public exception
{
    public:
        operator_err(const string err) : error(err) {};
        ~operator_err() throw() {};
        virtual const char* what() const throw() {return error.data();};

    private:
        string error;
};


/// Invalid size for array construction
class invalid_size : public exception
{
    public:
        /// Write error message 
        invalid_size(const string err) : error(err) {};
        ~invalid_size() throw() {};
        virtual const char* what() const throw() {return error.data();};

    private:
        int min_size;
        int max_size;
        int selected_size;
        string error;
};


/// Configuration file errors
class config_error : public exception{
    public:
        config_error(const string err) : error(err) {};
        ~config_error() throw() {};
        virtual const char* what() const throw() {return error.data();};

    private:
        string error;
};


/// Errors from the diagnostic unit
class diagnostics_error : public exception
{
    public:
        diagnostics_error(const string err) : error(err) {};
        ~diagnostics_error() throw() {};
        virtual const char* what() const throw() {return error.data();};

    private:
        string error;
};


/// Invalid field name
class name_error : public exception
{
    public:
        name_error(const string err) : error(err) {};
        ~name_error() throw() {};
        virtual const char* what() const throw() {return error.data();};

    private:
        string error;
};


/// GPU call errorss
class gpu_error : public exception
{
    public:
        gpu_error(const string err) : error(err) {};
        ~gpu_error() throw() {};
        virtual const char* what() const throw() {return error.data();};

    private:
        string error;
};


class assert_error : public exception
{
    public:
        assert_error(const string err) : error(err) {};
        ~assert_error() throw() {};
        virtual const char* what() const throw() {return error.data();};

    private:
        string error;
};

#endif //__ERROR_H
