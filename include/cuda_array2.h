/*
 * cuda_array2.h
 *
 *  Created on: Oct 22, 2013
 *      Author: rku000
 */


#ifndef CUDA_ARRAY2_H_
#define CUDA_ARRAY2_H_

/*
 * cuda_array.h
 *
 * Datatype to hold 2d CUDA arrays with three time levels
 *
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <complex>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuComplex.h>
#include <cufft.h>
#include <string>
#include "check_bounds.h"
#include "error.h"
#include "cuda_types.h"


// Error checking macro for cuda calls
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line)
{ 
    if (code != cudaSuccess) 
    {
        stringstream err_str;
        err_str << "GPUassert: " << cudaGetErrorString(code) << "\t file: " << file << ", line: " << line << "\n";
        throw gpu_error(err_str.str());
    }
}


// Define a class for the template parameter T that goes into cuda_array
// The idea is that the CUDA kernels need different function calls
// for assigning a field index to a value, depending on the template parameter
// i.e.
// array[index] = doule(index)
// array[index] = make_CuDoubleComplex(double(index), double(index)
//
// Using the cuda_array_value_type class this can be written as
// array[index] = ca_val(index) and we can happily use a templates
// in the implementation of cuda_array


template<class T>
class ca_val
{
    public:
        __host__ __device__ ca_val() {} ;
        __host__ __device__ inline void set(cuda::real_t);
        __host__ __device__ inline void set(cuda::cmplx_t);
        __host__ __device__ inline T get() const {return x;};
    private:
        T x;
};


template<>
inline void ca_val<cuda::real_t> :: set(cuda::real_t a)
{
    x = a;
}


template<>
inline void ca_val<cuda::real_t> :: set(cuda::cmplx_t a)
{
    x = a.x;
}


template<>
inline void ca_val<cuda::cmplx_t> :: set(cuda::real_t a)
{
    x = make_cuDoubleComplex(a, a);
}

template<>
inline void ca_val<cuda::cmplx_t> :: set(cuda::cmplx_t a)
{
    x = a;
}

template class ca_val<cuda::real_t>;
template class ca_val<cuda::cmplx_t>;


template <class T>
class cuda_array{
    public:
        // Explicitly declare construction operators that allocate memory
        // on the device
        cuda_array(uint, uint, uint);
        cuda_array(const cuda_array&);
        ~cuda_array();

        T* get_array_host(int) const;

        // Test function
        void enumerate_array(const uint);
        void enumerate_array_t(const uint);

        // Operators
        cuda_array<T>& operator=(const cuda_array<T>&);
        cuda_array<T>& operator=(const T&);

        cuda_array<T>& operator+=(const cuda_array<T>&);
        cuda_array<T>& operator+=(const T&);

        cuda_array<T>& operator-=(const cuda_array<T>&);
        cuda_array<T>& operator-=(const T&);

        cuda_array<T>& operator*=(const cuda_array<T>&);
        cuda_array<T>& operator*=(const T&);

        // Similar to operator=, but operates on all time levels
        cuda_array<T>& set_all(const double&);
        cuda_array<T>& set_all(const cuDoubleComplex&);
        // Set array to constant value for specified time level
        cuda_array<T>& set_t(const T&, uint);
        // Access operator to host array
        T& operator()(uint, uint, uint);
        T operator()(uint, uint, uint) const;

        // Problem: cuDoubleComplex doesn't like cout. Implement a wrapper that
        // returns a string representation of a single element of the host data array
        //string dump_to_str();
        string cout_wrapper(const T &val) const;

        // Copy device memory to host and print to stdout
        friend std::ostream& operator<<(std::ostream& os, cuda_array<T>& src)
        {
            const uint tl = src.get_tlevs();
            const uint nx = src.get_nx();
            const uint my = src.get_my();
            src.copy_device_to_host();
            os << std::setw(10);
            os << "\n";
            for(uint t = 0; t < tl; t++)
            {
                os << "t: " << t << "\n";
                for(uint n = 0; n < nx; n++)
                {
                    for(uint m = 0; m < my; m++)
                    {
                        os << std::setw(6) << std::setprecision(4);
                        os << src.cout_wrapper(src(t,n,m)) << "\t";
                    }
                os << "\n";
                }
                os << "\n\n";
            }
            return (os);
        }

        void copy_device_to_host();
        void copy_device_to_host(uint);

        // Transfer from host to device
        void copy_host_to_device();

        // Advance time levels
        void advance();
        // copy(dst, src);
        void copy(uint, uint);
        void copy(uint, const cuda_array<T>&, uint);
        void move(uint, uint);
        void swap(uint, uint);
        void normalize();

        // Access to private members
        inline uint get_nx() const {return Nx;};
        inline uint get_my() const {return My;};
        inline uint get_tlevs() const {return tlevs;};
        inline int address(uint n, uint m) const {return (n * My + m);};
        inline dim3 get_grid() const {return grid;};
        inline dim3 get_block() const {return block;};

        // Pointer to host copy of device data
        inline T* get_array_h() const {return array_h;};
        inline T* get_array_h(uint t) const {return array_h_t[t];};

        // Pointer to device data
        inline T* get_array_d() const {return array_d;};
        inline T** get_array_d_t() const {return array_d_t;};
        inline T* get_array_d(uint t) const {return array_d_t_host[t];};

    private:
        // Size of data array. Host data
        const uint tlevs;
        const uint Nx;
        const uint My;

        check_bounds bounds;

        // grid and block dimension
        dim3 block;
        dim3 grid;
        // Grid for accessing all tlevs
        dim3 grid_full;
        // Array data is on device
        // Pointer to device data
        T* array_d;
        // Pointer to each time stage. Pointer to array of pointers on device
        T** array_d_t;
        // Pointer to each time stage: Pointer to each time level on host
        T** array_d_t_host;

        // Storage copy of device data on host
        T* array_h;
        T** array_h_t;
};

#endif /* CUDA_ARRAY2_H_ */
