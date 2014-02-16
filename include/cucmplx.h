/*
 * Complex data type for cuda
 */


#ifndef CUCMPLX
#define CUCMPLX

#include <iostream>
#include <iomanip>

#ifdef __CUDACC__
#define CUDA_MEMBER __host__ __device__
#else
#define CUDA_MEMBER
#endif



/// Template parameters:
/// T: underlying data type, most probably double
/// Define all operator arguments and return types as by-value, see Stroustrup(2013), 18.2.4
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

    CUDA_MEMBER inline T abs();
    // (a + ib) + (c + id) = (a + c) + i * (b + d)
    CUDA_MEMBER inline CuCmplx<T> operator+=(CuCmplx<T>); 
    CUDA_MEMBER inline CuCmplx<T> operator+(CuCmplx<T>);

    CUDA_MEMBER inline CuCmplx<T> operator+=(T); 
    CUDA_MEMBER inline CuCmplx<T> operator+(T);

    // (a + ib) - (c + id) = (a - c) + i(b - d)
    CUDA_MEMBER inline CuCmplx<T> operator-=(CuCmplx<T>);
    CUDA_MEMBER inline CuCmplx<T> operator-(CuCmplx<T>); 

    CUDA_MEMBER inline CuCmplx<T> operator-=(T);
    CUDA_MEMBER inline CuCmplx<T> operator-(T); 

    // (a + ib) * (c + id) = (ac - bd) + i(ad + bc)
    CUDA_MEMBER inline CuCmplx<T> operator*=(CuCmplx<T>);
    CUDA_MEMBER inline CuCmplx<T> operator*(CuCmplx<T>); 

    CUDA_MEMBER inline CuCmplx<T> operator*=(T);
    CUDA_MEMBER inline CuCmplx<T> operator*(T); 

    // (a + ib) / (c + id) = 
    CUDA_MEMBER inline CuCmplx<T> operator/=(CuCmplx<T>);
    CUDA_MEMBER inline CuCmplx<T> operator/(CuCmplx<T>);

    CUDA_MEMBER inline CuCmplx<T> operator/=(T);
    CUDA_MEMBER inline CuCmplx<T> operator/(T);

    CUDA_MEMBER inline CuCmplx<T> operator=(CuCmplx<T> );
    CUDA_MEMBER inline CuCmplx<T> operator=(T);
    //CUDA_MEMBER inline CuCmplx<T>& operator=(T);

    CUDA_MEMBER void dump();
    friend std::ostream& operator<<(std::ostream& os, CuCmplx<T> rhs)
    {
        //os << std::setw(6) << std::setprecision(4);
        os << "(" << rhs.re() << ", " << rhs.im() << ")";
        return (os);
    }

    CUDA_MEMBER inline void set(T re, T im) {real = re; imag = im;};
    CUDA_MEMBER inline T re() {return real;}
    CUDA_MEMBER inline T im() {return imag;}


private:
    T real;
    T imag;
};



template <typename T>
#ifdef __CUDACC__
__host__ __device__
#endif
inline T CuCmplx<T>:: abs()
{
    return (real*real + imag * imag);
}

template <typename T>
#ifdef __CUDACC__
__host__ __device__
#endif
inline CuCmplx<T> CuCmplx<T> :: operator+=(CuCmplx<T> rhs)
{
    real += rhs.real;
    imag += rhs.imag;
    return (*this);
}

template <typename T>
#ifdef __CUDACC__
__host__ __device__
#endif
inline CuCmplx<T> CuCmplx<T> :: operator+=(T rhs)
{
    real += rhs;
    return (*this);
}


template <typename T>
#ifdef __CUDACC__
__host__ __device__
#endif
inline CuCmplx<T> CuCmplx<T> :: operator-=(CuCmplx<T> rhs)
{
    real -= rhs.real;
    imag -= rhs.imag;
    return (*this);
}


template <typename T>
#ifdef __CUDACC__
__host__ __device__
#endif
inline CuCmplx<T> CuCmplx<T> :: operator-=(T rhs)
{
    real -= rhs;
    return (*this);
}


template <typename T>
#ifdef __CUDACC__
__host__ __device__
#endif
inline CuCmplx<T> CuCmplx<T> :: operator*=(CuCmplx<T> rhs)
{
    T new_real = real * rhs.real - imag * rhs.imag;
    T new_imag = imag * rhs.real + real * rhs.imag;
    real = new_real;
    imag = new_imag;
    return (*this);
}


template <typename T>
#ifdef __CUDACC__
__host__ __device__
#endif
inline CuCmplx<T> CuCmplx<T> :: operator*=(T rhs)
{
    real *= rhs;
    imag *= rhs;
    return (*this);
}


template <typename T>
#ifdef __CUDACC__
__host__ __device__
#endif
inline CuCmplx<T> CuCmplx<T> :: operator /=(CuCmplx<T> rhs)
{
    T new_real = (real * rhs.real + imag * rhs.imag) / rhs.abs();
    T new_imag = (imag * rhs.real - real * rhs.imag) / rhs.abs();
    real = new_real;
    imag = new_imag;
    return (*this);
}


template <typename T>
#ifdef __CUDACC__
__host__ __device__
#endif
inline CuCmplx<T> CuCmplx<T> :: operator /=(T rhs)
{
    T inv_rhs = 1.0 / rhs;
    real *= inv_rhs;
    imag *= inv_rhs;
    return (*this);
}


template <typename T>
#ifdef __CUDACC__
__host__ __device__
#endif
inline CuCmplx<T> CuCmplx<T> :: operator+(CuCmplx<T> rhs)
{
    CuCmplx<T> result(this);
    result += rhs;
    return result;
}


template <typename T>
#ifdef __CUDACC__
__host__ __device__
#endif
inline CuCmplx<T> CuCmplx<T> :: operator+(T rhs)
{
    CuCmplx<T> result(this);
    result += rhs;
    return result;
}


template <typename T>
#ifdef __CUDACC__
__host__ __device__
#endif
inline CuCmplx<T> CuCmplx<T> :: operator-(CuCmplx<T> rhs)
{
    CuCmplx<T> result(this);
    result -= rhs;
    return result;
}


template <typename T>
#ifdef __CUDACC__
__host__ __device__
#endif
inline CuCmplx<T> CuCmplx<T> :: operator-(T rhs)
{
    CuCmplx<T> result(this);
    result -= rhs;
    return result;
}


template <typename T>
#ifdef __CUDACC__
__host__ __device__
#endif
inline CuCmplx<T> CuCmplx<T> :: operator*(CuCmplx<T> rhs)
{
    CuCmplx<T> result(this);
    result *= rhs;
    return (result);
}


template <typename T>
#ifdef __CUDACC__
__host__ __device__
#endif
inline CuCmplx<T> CuCmplx<T> :: operator*(T rhs)
{
    CuCmplx<T> result(this);
    result *= rhs;
    return (result);
}


template <typename T>
#ifdef __CUDACC__
__host__ __device__
#endif
inline CuCmplx<T> CuCmplx<T> :: operator/(CuCmplx<T> rhs)
{
    CuCmplx<T> result(this);
    result /= rhs;
    return (result);
}

template <typename T>
#ifdef __CUDACC__
__host__ __device__
#endif
inline CuCmplx<T> CuCmplx<T> :: operator/(T rhs)
{
    CuCmplx<T> result(this);
    result /= rhs;
    return (result);
}


template <typename T>
#ifdef __CUDACC__
__host__ __device__
#endif
inline CuCmplx<T> CuCmplx<T> :: operator=(CuCmplx<T> rhs)
{
    real = rhs.real;
    imag = rhs.imag;
    return (*this);
}

/*
template <typename T>
#ifdef __CUDACC__
__host__ __device__
#endif
inline CuCmplx<T>& CuCmplx<T> :: operator=(T &rhs)
{
    real = rhs;
    imag = 0.0;
    return (*this);
}
*/

/// To do stuff like U** foo;foo[i][j] = 0.0;
template <typename T>
#ifdef __CUDACC__
__host__ __device__
#endif
inline CuCmplx<T> CuCmplx<T> :: operator=(T rhs)
{
    real = rhs;
    imag = 0.0;
    return (*this);
}






#endif //CUCMPLX

