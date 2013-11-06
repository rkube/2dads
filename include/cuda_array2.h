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



// Error checking macro for cuda calls
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true){
    if (code != cudaSuccess) {
        std::cerr << "GPUassert: " << cudaGetErrorString(code) << "\t file: " << file << ", line: " << line << "\n";
        if (abort)
            exit(code);
    }
}



// Define complex type
typedef cufftDoubleComplex cmplx_t;
typedef double real_t;
// Constants

const int cuda_blockdim_nx = 1;
const int cuda_blockdim_my = 64;
const double PI = atan(1.0) * 4.0;


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
        __host__ __device__ ca_val(T a) : x(a) {};
        __host__ __device__ void set(real_t);
        __host__ __device__ T get() const {return x;};
    private:
        T x;
};


template<>
void ca_val<real_t> :: set(real_t a)
{
    x = a;
}


template<>
void ca_val<cmplx_t> :: set(real_t a)
{
    x = make_cuDoubleComplex(a, -a);
}

template class ca_val<real_t>;
template class ca_val<cmplx_t>;


template <class T>
class cuda_array{
    public:
        // Explicitly declare construction operators that allocate memory
        // on the device
        cuda_array(unsigned int, unsigned int, unsigned int);
        cuda_array(const cuda_array&);
        ~cuda_array();

        T* get_array_host(int) const;

        // Test function
        void enumerate_array(const int);
        void enumerate_array_t(const int);

        // Operators
        cuda_array& operator=(const cuda_array<T>&);
        cuda_array& operator=(const T&);
        cuda_array& set_all(const T&);
        // Access operator to host array
        T& operator()(unsigned int, unsigned int, unsigned int);
        T operator()(unsigned int, unsigned int, unsigned int) const;

        // Problem: cuDoubleComplex doesn't like cout. Implement a wrapper that
        // returns a string representation of a single element of the host data array
        //string dump_to_str();
        string cout_wrapper(const T &val) const;


        // Copy device memory to host and print to stdout
        friend std::ostream& operator<<(std::ostream& os, const cuda_array<T>& src)
        {
            const unsigned int tl = src.get_tlevs();
            const unsigned int nx = src.get_nx();
            const unsigned int my = src.get_my();
            //src.copy_device_to_host();
            os << std::setw(10);
            os << "Dumping array. size: " << tl << ", " << nx << ", " << my <<"\n";
            for(unsigned int t = 0; t < tl; t++)
            {
                os << "t: " << t << "\n";
                for(unsigned int n = 0; n < nx; n++)
                {
                    for(unsigned int m = 0; m < my; m++)
                    {
                        os << std::setw(8) << std::setprecision(6);
                        os << src.cout_wrapper(src(t,n,m)) << "\t";
                    }
                os << "\n";
                }
                os << "\n\n";
            }
            return (os);
        }

        void copy_device_to_host();

        // Transfer from host to device
        void copy_host_to_device();

        // Advance time levels
        void advance();
        // copy(dst, src);
        void copy(unsigned int, unsigned int);
        void copy(unsigned int, const cuda_array<T>&, unsigned int);
        void move(unsigned int, unsigned int);
        void normalize();

        // Initialize device data
        //void init_arr_d_sine();

        // Functions that initialize host data
        //void init_arr_h(T&);

        // Access to private members
        inline unsigned int get_nx() const {return Nx;};
        inline unsigned int get_my() const {return My;};
        inline unsigned int get_tlevs() const {return tlevs;};
        inline int address(unsigned int n, unsigned int m) const {return (n * My + m);};

        inline T* get_array_h() const {return array_h;};
        inline T* get_array_h(unsigned int t) const {return array_h_t[t];};


    private:
        // Size of data array. Host data
        const unsigned int tlevs;
        const unsigned int Nx;
        const unsigned int My;

        check_bounds bounds;

        // grid and block dimension
        dim3 block;
        dim3 grid;
        // Grid for accessing all tlevs
        dim3 grid_full;
        // Array data is on device
        // Pointer to device data
        T* array_d;
        // Pointer to each time stage
        T** array_d_t;

        // Storage copy of device data on host
        T* array_h;
        T** array_h_t;
};


#endif /* CUDA_ARRAY2_H_ */
