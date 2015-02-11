/*!
 * Complex data type for cuda
 * Define all operator arguments and return types as by-value, see Stroustrup(2013), 18.2.4
 * \param T: underlying data type, most probably double
 */


#ifndef CUCMPLX_H
#define CUCMPLX_H

#include <iostream>
#include <iomanip>
#include "cucmplx.h"

#ifdef __CUDACC__
#define CUDA_MEMBER __host__ __device__
#else
#define CUDA_MEMBER
#endif



template<typename T>
class CuCmplx
{
public:
    CUDA_MEMBER CuCmplx() : real(T(0.0)), imag(T(0.0)) {};
    CUDA_MEMBER CuCmplx(T re) : real(re), imag(0.0) {};
    CUDA_MEMBER CuCmplx(int re) : real(T(re)), imag(0.0) {};
    CUDA_MEMBER CuCmplx(unsigned int re) : real(T(re)), imag(0.0) {};
    CUDA_MEMBER CuCmplx(T re, T im) : real(re), imag(im) {};
    //CuCmplx(const CuCmplx<T> rhs) : real(rhs.real), imag(rhs.imag) {};
    CUDA_MEMBER CuCmplx(const CuCmplx<T>& rhs) : real(rhs.real), imag(rhs.imag) {};
    CUDA_MEMBER CuCmplx(const CuCmplx<T>* rhs) : real(rhs -> real), imag(rhs -> imag) {};

    CUDA_MEMBER inline T abs() const {return (real*real + imag * imag);};
    // (a + ib) + (c + id) = (a + c) + i * (b + d)


    CUDA_MEMBER inline CuCmplx<T> operator+=(const CuCmplx<T> & rhs)
    {
        real += rhs.real;
        imag += rhs.imag;
        return (*this);
    }


    CUDA_MEMBER CuCmplx<T> operator+(const CuCmplx<T>& rhs) const
    {
        CuCmplx<T> result(this);
        result += rhs;
        return result;
    }


    CUDA_MEMBER inline CuCmplx<T> operator+=(const T& rhs)
    {
    	real += rhs;
    	return (*this);
    }


    CUDA_MEMBER inline CuCmplx<T> operator+(const T& rhs) const
    {
    	CuCmplx<T> result(this);
    	result.real += rhs;
    	return(result);
    }


    // (a + ib) - (c + id) = (a - c) + i(b - d)
    CUDA_MEMBER inline CuCmplx<T> operator-=(const CuCmplx<T>& rhs)
    {
    	    real -= rhs.real;
    	    imag -= rhs.imag;
    	    return (*this);
	}


    CUDA_MEMBER inline CuCmplx<T> operator-(const CuCmplx<T>& rhs) const
    {
        CuCmplx<T> result(this);
        result -= rhs;
        return result;
    }

    CUDA_MEMBER inline CuCmplx<T> operator-=(const T& rhs)
    {
        real -= rhs;
        return (*this);
    }


    CUDA_MEMBER inline CuCmplx<T> operator-(const T& rhs) const
    {
        CuCmplx<T> result(this);
        result += rhs;
        return result;
    }

    // (a + ib) * (c + id) = (ac - bd) + i(ad + bc)
    CUDA_MEMBER inline CuCmplx<T> operator*=(const CuCmplx<T>& rhs)
	{
    	// Create temporary variables, so that we don't overwrite temporary variables
        T new_real = real * rhs.real - imag * rhs.imag;
        T new_imag = imag * rhs.real + real * rhs.imag;
        real = new_real;
        imag = new_imag;
        // NO!!! NOT LIKE THIS!! This overwrites (*this).real in the first line!
    	//real = real * rhs.real - imag * rhs.imag;
    	//imag = imag * rhs.real + real * rhs.imag;
        return (*this);
	}


    CUDA_MEMBER inline CuCmplx<T> operator*(const CuCmplx<T>& rhs) const
    {
        CuCmplx<T> result(this);
        result *= rhs;
        return result;
    }


    CUDA_MEMBER inline CuCmplx<T> operator*=(const T& rhs)
    {
        real *= rhs;
        imag *= rhs;
        return (*this);
    }


    CUDA_MEMBER inline CuCmplx<T> operator*(const T& rhs) const
    {
    	CuCmplx<T> result(this);
    	result *= rhs;
    	return result;
    }

    // (a + ib) / (c + id) = 
    CUDA_MEMBER inline CuCmplx<T> operator/=(const CuCmplx<T>& rhs)
	{
    	// Create temporary variables so that we don't overwrite this
    	// with temporary results
        T new_real = (real * rhs.real + imag * rhs.imag) / rhs.abs();
        T new_imag = (imag * rhs.real - real * rhs.imag) / rhs.abs();
        real = new_real;
        imag = new_imag;
        return (*this);
	}


    CUDA_MEMBER inline CuCmplx<T> operator/(const CuCmplx<T> & rhs) const
    {
    	CuCmplx<T> result(this);
    	result /= rhs;
    	return(result);
    }

    CUDA_MEMBER inline CuCmplx<T> operator/=(const T& rhs)
    {
        //T inv_rhs = 1.0 / rhs;
        real /= rhs;
        imag /= rhs;
        return (*this);
    }


    CUDA_MEMBER inline CuCmplx<T> operator/(const T& rhs) const
    {
    	CuCmplx<T> result(this);
    	result /= rhs;
    	return(result);
    }

    CUDA_MEMBER inline CuCmplx<T> operator=(const CuCmplx<T>& rhs)
    {
    	real = rhs.real;
    	imag = rhs.imag;
    	return (*this);
    }
    CUDA_MEMBER inline CuCmplx<T> operator=(const T& rhs)
    {
    	real = rhs;
    	imag = T(0.0);
    	return (*this);
    }
    //CUDA_MEMBER inline CuCmplx<T>& operator=(T);

    CUDA_MEMBER void dump();
    friend std::ostream& operator<<(std::ostream& os, CuCmplx<T> rhs)
    {
        os << "(" << std::setw(7) << std::setprecision(4) << rhs.re() << ", ";
        os << std::setw(7) << std::setprecision(4) << rhs.im() << ")";
        return (os);
    }

    CUDA_MEMBER inline void set(T re, T im) {real = re; imag = im;};
    CUDA_MEMBER inline T re() const {return real;}
    CUDA_MEMBER inline T im() const {return imag;}


private:
    T real;
    T imag;
};


#endif //CUCMPLX

