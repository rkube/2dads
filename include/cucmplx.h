/*!
 * Complex data type for cuda
 * Define all operator arguments and return types as by-value, see Stroustrup(2013), 18.2.4
 * \param T: underlying data type, most probably double
 */


#ifndef CUCMPLX_H
#define CUCMPLX_H

#include <iostream>
#include <iomanip>
#include <cmath>
//#include "cucmplx.h"

#ifdef __CUDACC__
#define CUDA_MEMBER __host__ __device__
#else
#define CUDA_MEMBER
#endif



template<typename T>
class CuCmplx
{
public:
    CUDA_MEMBER CuCmplx() : data{T(0.0), T(0.0)} {};
    CUDA_MEMBER CuCmplx(T re) : data{re, T(0.0)} {};
    CUDA_MEMBER CuCmplx(int re) : data{T(re), T(0.0)} {};
    CUDA_MEMBER CuCmplx(unsigned int re) : data{T(re), T(0.0)} {};
    CUDA_MEMBER CuCmplx(T re, T im) : data{re, im} {};
    CUDA_MEMBER CuCmplx(const CuCmplx<T>& rhs) : data{rhs.re(), rhs.im()} {};
    CUDA_MEMBER CuCmplx(const CuCmplx<T>* rhs) : data{rhs -> re(), rhs -> im()} {};

    CUDA_MEMBER inline T abs() const {return (sqrt(re() * re() + im() * im()));};
    // (a + ib) + (c + id) = (a + c) + i * (b + d)

    CUDA_MEMBER inline CuCmplx<T> conj() const {return (CuCmplx(re(), -1.0 * im()));};


    CUDA_MEMBER inline CuCmplx<T> operator+=(const CuCmplx<T> & rhs)
    {
    	data[0] += rhs.re();
    	data[1] += rhs.im();
        return (*this);
    }


    CUDA_MEMBER CuCmplx<T> operator+(const CuCmplx<T>& rhs) const
    {
        CuCmplx<T> result(re() + rhs.re(), im() + rhs.im());
        return result;
    }


    CUDA_MEMBER inline CuCmplx<T> operator+=(const T& rhs)
    {
    	data[0] += rhs;
    	return (*this);
    }


    CUDA_MEMBER inline CuCmplx<T> operator+(const T& rhs) const
    {
    	CuCmplx<T> result(re() + rhs, im());
    	return(result);
    }


    // (a + ib) - (c + id) = (a - c) + i(b - d)
    CUDA_MEMBER inline CuCmplx<T> operator-=(const CuCmplx<T>& rhs)
    {
    	    data[0] -= rhs.re();
    	    data[1] -= rhs.im();
    	    return (*this);
	}


    CUDA_MEMBER inline CuCmplx<T> operator-(const CuCmplx<T>& rhs) const
    {
        CuCmplx<T> result(re() - rhs.re(), im() - rhs.im());
        return result;
    }

    CUDA_MEMBER inline CuCmplx<T> operator-=(const T& rhs)
    {
        data[0] -= rhs;
        return (*this);
    }


    CUDA_MEMBER inline CuCmplx<T> operator-(const T& rhs) const
    {
        CuCmplx<T> result(re() - rhs, im());
        return result;
    }

    // (a + ib) * (c + id) = (ac - bd) + i(ad + bc)
    CUDA_MEMBER inline CuCmplx<T> operator*=(const CuCmplx<T>& rhs)
	{
    	// Create temporary variables, so that we don't overwrite temporary variables
        T new_real = data[0] * rhs.re() - data[1] * rhs.im();
        T new_imag = data[1] * rhs.re() + data[0] * rhs.im();
        set_re(new_real);
        set_im(new_imag);
        //real = new_real;
        //imag = new_imag;
        // NO!!! NOT LIKE THIS!! This overwrites (*this).real in the first line!
    	//real = real * rhs.real - imag * rhs.imag;
    	//imag = imag * rhs.real + real * rhs.imag;
        return (*this);
	}


    CUDA_MEMBER inline CuCmplx<T> operator*(const CuCmplx<T>& rhs) const
    {
        CuCmplx<T> result(data[0] * rhs.re() - data[1] * rhs.im(),
                          data[1] * rhs.re() + data[0] * rhs.im());
        return result;
    }


    CUDA_MEMBER inline CuCmplx<T> operator*=(const T& rhs)
    {
    	data[0] *= rhs;
    	data[1] *= rhs;
        return (*this);
    }


    CUDA_MEMBER inline CuCmplx<T> operator*(const T& rhs) const
    {
    	CuCmplx<T> result(re() * rhs, im() * rhs);
    	return result;
    }

    // (a + ib) / (c + id) = 
    CUDA_MEMBER inline CuCmplx<T> operator/=(const CuCmplx<T>& rhs)
	{
    	// Create temporary variables so that we don't overwrite this
    	// with temporary results
        T new_real = (re() * rhs.re() + im() * rhs.im()) / rhs.abs();
        T new_imag = (im() * rhs.re() - re() * rhs.im()) / rhs.abs(
        		);
        set_re(new_real);
        set_im(new_imag);

        return (*this);
	}


    CUDA_MEMBER inline CuCmplx<T> operator/(const CuCmplx<T> & rhs) const
    {
    	CuCmplx<T> result((re() * rhs.re() + im() * rhs.im()) / rhs.abs(),
    			          (im() * rhs.re() - re() * rhs.im()) / rhs.abs());
    	return(result);
    }

    CUDA_MEMBER inline CuCmplx<T> operator/=(const T& rhs)
    {
        data[0] /= rhs;
        data[1] /= rhs;

        return (*this);
    }


    CUDA_MEMBER inline CuCmplx<T> operator/(const T& rhs) const
    {
    	CuCmplx<T> result(re() / rhs, im() / rhs);
    	return(result);
    }

    CUDA_MEMBER inline CuCmplx<T> operator=(const CuCmplx<T>& rhs)
    {
    	set_re(rhs.re());
    	set_im(rhs.im());
    	return (*this);
    }
    CUDA_MEMBER inline CuCmplx<T> operator=(const T& rhs)
    {
    	set_re(rhs);
    	set_im(T(0.0));
    	return (*this);
    }

    CUDA_MEMBER void dump();
    friend std::ostream& operator<<(std::ostream& os, CuCmplx<T> rhs)
    {
        os << "(" << std::setw(7) << std::setprecision(4) << rhs.re() << ", ";
        os << std::setw(7) << std::setprecision(4) << rhs.im() << ")";
        return (os);
    }

    //CUDA_MEMBER inline void set(T re, T im) {real = re; imag = im;};
    CUDA_MEMBER inline T re() const {return(data[0]);}
    CUDA_MEMBER inline T& re() {return(data[0]);}
    CUDA_MEMBER inline void set_re(const T& re) {data[0] = re;}

    CUDA_MEMBER inline T im() const {return(data[1]);}
    CUDA_MEMBER inline T& im() {return(data[1]);}
    CUDA_MEMBER inline void set_im(const T& im) {data[1] = im;}

private:
    T data[2];
};


#endif //CUCMPLX

