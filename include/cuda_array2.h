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
#include "check_bounds.h"
#include "error.h"



#ifndef CUDA_ARRAY_H
#define CUDA_ARRAY_H

// Error checking macro for cuda calls
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true){
    if (code != cudaSuccess) {
        std::cerr << "GPUassert: " << cudaGetErrorString(code) << "\t file: " << file << ", line: " << line << "\n";
        if (abort)
            exit(code);
    }
}

// Error checking macro for cufft calls
/*
#define cufftCall(ans) { cufftAssert((ans), __FILE__, __LINE__); }
inline void cufftAssert(cufftResult_t code, char *file, int line, bool abort=true){
    if (code != CUFFT_SUCCESS) {
        std::cerr << "cufftAssert: Error in cufft call in " << file << ", line: " << line << "\n";
        if (abort)
            exit(code);
    }
}
*/

// Define complex type
typedef cufftDoubleComplex cmplx_t;
//typedef std::complex<double> cmplx_t;

// Constants
//constexpr unsigned int block_nx = 16;
//constexpr unsigned int block_my = 16;
const int cuda_blockdim_nx = 4;
const int cuda_blockdim_my = 4;

const double PI = atan(1.0) * 4.0;

template <class T>
class cuda_array{
    public:
        // Explicitly declare construction operators that allocate memory
        // on the device
        cuda_array();
        cuda_array(unsigned int, unsigned int, unsigned int);
        cuda_array(const cuda_array&);

        ~cuda_array();

        //cmplx_t* get_array_host(int) const;
        T* get_array_host(int) const;

        // Test function
        void enumerate_array(const int);
        void dump_array() const;
        //void dump_array_d() const;
        //void dump_array_c() const;

        // Operators
        cuda_array& operator=(const cuda_array&);
        cuda_array& operator=(const T&);
        // Access operator to host array
        T& operator()(unsigned int, unsigned int, unsigned int);
        T operator()(unsigned int, unsigned int, unsigned int) const;

        // Copy device memory to host and print to stdout
        friend std::ostream& operator<<(std::ostream& os, const cuda_array& src)
        {
            const unsigned int tl = src.get_tlevs();
            const unsigned int nx = src.get_nx();
            const unsigned int my = src.get_my();
            src.copy_device_to_host();
            os << std::setw(10);
            for(unsigned int t = 0; t < tl; t++)
            {
                os << "t: " << t << "\n";
                for(unsigned int n = 0; n < nx; n++)
                {
                    for(unsigned int m = 0; m < my; m++)
                    {
                        os << std::setw(8) << std::setprecision(6);
                        os << src(t,n,m) << "\t";
                    }
                os << "\n";
                }
                os << "\n\n";
            }
            return (os);
        }


        //Transfer from device to host
        //void copy_device_to_host_d(const int tlev) const;
        //void copy_device_to_host_c(const int tlev) const;
        void copy_device_to_host() const;

        // Transfer from host to device
        //void copy_host_to_device_d(const int tlev) const;
        //void copy_host_to_device_c(const int tlev) const ;
        void copy_host_to_device() const;
        void advance();

        // Initialize device data
        //void init_arr_d_sine();
        void init_arr_d_sine();

        // Functions that initialize host data
        void init_arr_h(T&);
        //void init_arr_d_host();
        //void init_arr_d_host(double);
        //void init_arr_c_host(cmplx_t);

        // Access to private members
        inline unsigned int get_nx() const {return tlevs;};
        inline unsigned int get_my() const {return Nx;};
        inline unsigned int get_tlevs() const {return tlevs;};
        inline int address(unsigned int n, unsigned int m) const {return (n * My + m);};

        inline T* get_array_h() const {return array_h;};
        inline T* get_array_h(unsigned int t) const {return array_h_t[t];};
//      Compare to array_base.h
//        // Access to dimensions
//        inline  uint get_tlevs() const {return tlevs;}
//        inline  uint get_nx() const {return Nx;}
//        inline  uint get_my() const {return My;}
//        inline T* get_array() const {return array;}
//        inline T* get_array(uint t) const {return array_t[t];}


        // DFT of array data. These will be removed later on
        //void fft_forward();
        //void fft_backward();

    private:
        // Size of data array. Host data

        const unsigned int Nx;
        const unsigned int My;
        const unsigned int tlevs;
        check_bounds bounds;

        // grid and block dimension
        dim3 block;
        dim3 grid;
        // Array data is on device
        // Pointer to device data
        T* array_d;
        // Pointer to each time stage
        T** array_d_t;

        // Storage copy of device data on host
        T* array_h;
        T** array_h_t;
        //double* array_h_d;

        //cufftHandle plan_fw;
        //cufftHandle plan_bw;
};

#endif //CUDA_ARRAY_H



#endif /* CUDA_ARRAY2_H_ */
