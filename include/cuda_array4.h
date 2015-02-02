/*
 * cuda_array4.h
 *
 *  Created on: Feb 2, 2015
 *      Author: rku000
 *
 * Datatype to hold 2d CUDA arrays with three time levels
 *
 *  when PINNED_HOST_MEMORY is defined, memory for mirror copy of array in host
 *  memory is pinned, i.e. non-pageable. This increases memory transfer rates
 *  between host and device in exchange for more heavyweight memory allocation.
 *
 *
 * Version 4:
 * - Templated arithmetic operations
 *
 */

#ifndef CUDA_ARRAY4_H_
#define CUDA_ARRAY4_H_


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


// Verify last kernel launch
#define gpuStatus() { gpuVerifyLaunch(__FILE__, __LINE__); }
inline void gpuVerifyLaunch(const char* file, int line)
{
     cudaThreadSynchronize();
     cudaError_t error = cudaGetLastError();
     if(error != cudaSuccess)
     {
        stringstream err_str;
        err_str << "GPUassert: " << cudaGetErrorString(error) << "\t file: " << file << ", line: " << line << "\n";
        throw gpu_error(err_str.str());
     }
}

#ifdef __CUDACC__
/// Device function to compute column and row
__device__ inline int d_get_col() {
    return (blockIdx.y * blockDim.y + threadIdx.y);
}


__device__ inline int d_get_row() {
    return (blockIdx.x * blockDim.x + threadIdx.x);
}


/// Device functors for arithmetic operations
template <typename T>
class d_op_add{
public:
	__device__ T operator()(T a, T b) const {return(a + b);};
};


template <typename T>
class d_op_sub{
public:
	__device__ T operator()(T a, T b) const {return(a - b);};
};


template <typename T>
class d_op_mul{
public:
	__device__ T operator()(T a, T b) const {return(a * b);};
};


template <typename T>
class d_op_div{
public:
	__device__ T operator()(T a, T b) const {return(a / b);};
};


// Template kernel for d_enumerate_d and d_enumerate_c using the ca_val class
template <typename T>
__global__ void d_enumerate(T* array, uint Nx, uint My)
{
    const int col = d_get_col();
    const int row = d_get_row();
    const int index = row * (My) + col;

    if (index < Nx * My)
    {
        array[index] = T(index);
    }
}


// Template version of d_enumerate_t_x
template <typename T>
__global__ void d_enumerate_t(T** array_t, uint t, uint Nx, uint My)
{
    const int col = d_get_col();
    const int row = d_get_row();
    const int index = row * My + col;

	if (index < Nx * My)
    {
        array_t[t][index] = T(index);
    }
}

template <typename T>
__global__ void d_set_constant_t(T** array_t, T val, uint t, uint Nx, uint My)
{
    const int col = d_get_col();
    const int row = d_get_row();
	const int index = row * My + col;
	if (index < Nx * My)
		array_t[t][index] = val;
}


template <typename T>
__global__ void d_alloc_array_d_t(T** array_d_t, T* array,  uint tlevs,  uint Nx,  uint My)
{
	if (threadIdx.x < tlevs)
	{
		array_d_t[threadIdx.x] = &array[threadIdx.x * Nx * My];
	}
}


template <typename T>
__global__ void d_advance(T** array_t, int tlevs)
{
	T* tmp = array_t[tlevs - 1];
	int t = 0;

	for(t = tlevs - 1; t > 0; t--)
		array_t[t] = array_t[t - 1];
	array_t[0] = tmp;
}


template <typename T>
__global__ void d_swap(T**array_t, uint t1, uint t2)
{
    T* tmp = array_t[t1];
    array_t[t1] = array_t[t2];
    array_t[t2] = tmp;
}



template <typename T>
__global__ void test_alloc(T** array_t,  uint tlevs)
{
    for(int t = 0; t < tlevs; t++)
        printf("array_x_t[%d] at %p\n", t, array_t[t]);
}


/// Arithmetic operation,
/// op(lhs[0][idx], rhs[tlev][idx])
template <typename T, typename O>
__global__
void d_op_arr_t(T** lhs, T** rhs, uint tlev, O op, uint Nx, uint My)
{
    const int col = d_get_col();
    const int row = d_get_row();
    const int idx = row * My + col;
    if ((col < My) && (row < Nx))
    	lhs[0][idx] = op(lhs[0][idx], rhs[tlev][idx]);
}


// Add scalar
template <typename T, typename O>
__global__
void d_op_scalar(T** lhs, T rhs, O op, uint Nx, uint My)
{
    const int col = d_get_col();
    const int row = d_get_row();
    const int idx = row * My + col;
    if ((col < My) && (row < Nx))
    	lhs[0][idx] = op(lhs[0][idx], rhs);
        //lhs[0][idx] += rhs;
}



// Kill kx = 0 mode
template <typename T>
__global__
void d_kill_kx0(T* arr, uint Nx, uint My)
{
    const int row = 0;
    const int col = d_get_col();
    const int idx = row * My + col;
    if (col > My)
        return;

    T zero(0.0);
    arr[idx] = zero;
}


// Kill ky=0 mode
template <typename T>
__global__
void d_kill_ky0(T* arr, uint Nx, uint My)
{
    const int row = d_get_row();
    const int col = 0;
    const int idx = row * My + col;
    if (row > Nx)
        return;

    T zero(0.0);
    arr[idx] = zero;
}


// Kill k=0 mode
// This should be only invoked where U = cuda::cmplx_t*
template <typename T>
__global__
void d_kill_k0(T* arr)
{
    T zero(0.0);
    arr[0] = zero;
}


#endif // __CUDACC_

/// T is the base type of T, for example T=CuCmplx<double>, T = double, T = float...
template <typename T>
class cuda_array{
    public:
        // Explicitly declare construction operators that allocate memory
        // on the device
        cuda_array(uint, uint, uint);
        cuda_array(const cuda_array<T>&);
        cuda_array(const cuda_array<T>*);
        ~cuda_array();

        // Test function
        void enumerate_array(const uint);
        void enumerate_array_t(const uint);

        // Operators
        cuda_array<T>& operator=(const cuda_array<T>&);
        cuda_array<T>& operator=(const T&);

        cuda_array<T>& operator+=(const cuda_array<T>&);
        cuda_array<T>& operator+=(const T&);
        cuda_array<T> operator+(const cuda_array<T>&);
        cuda_array<T> operator+(const T&);

        cuda_array<T>& operator-=(const cuda_array<T>&);
        cuda_array<T>& operator-=(const T&);
        cuda_array<T> operator-(const cuda_array<T>&);
        cuda_array<T> operator-(const T&);

        cuda_array<T>& operator*=(const cuda_array<T>&);
        cuda_array<T>& operator*=(const T&);
        cuda_array<T> operator*(const cuda_array<T>&);
        cuda_array<T> operator*(const T&);

        cuda_array<T>& operator/=(const cuda_array<T>&);
        cuda_array<T>& operator/=(const T&);
        cuda_array<T> operator/(const cuda_array<T>&);
        cuda_array<T> operator/(const T&);

        // Similar to operator=, but operates on all time levels
        cuda_array<T>& set_all(const T&);
        // Set array to constant value for specified time level
        cuda_array<T>& set_t(const T&, uint);
        // Access operator to host array
        inline T& operator()(uint, uint, uint);
        inline T operator()(uint, uint, uint) const;

        // Copy device memory to host and print to stdout
        friend std::ostream& operator<<(std::ostream& os, cuda_array<T> src)
        {
            const uint tl = src.get_tlevs();
            const uint nx = src.get_nx();
            const uint my = src.get_my();
            src.copy_device_to_host();
            os << "\n";
            for(uint t = 0; t < tl; t++)
            {
                for(uint n = 0; n < nx; n++)
                {
                    for(uint m = 0; m < my; m++)
                    {
                        // Remember to also set precision routines in CuCmplx :: operator<<
                        os << std::setw(8) << std::setprecision(4) << src(t, n, m) << "\t";
                    }
                os << "\n";
                }
                os << "\n\n";
            }
            return (os);
        }

        inline void copy_device_to_host();
        inline void copy_device_to_host(uint);
        inline void copy_device_to_host(T*);

        // Transfer from host to device
        inline void copy_host_to_device();

        // Advance time levels
        inline void advance();

        inline void copy(uint, uint);
        inline void copy(uint, const cuda_array<T>&, uint);
        inline void move(uint, uint);
        inline void swap(uint, uint);
        inline void normalize();

        inline void kill_kx0();
        inline void kill_ky0();
        inline void kill_k0();

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


#ifdef __CUDACC__
// Default constructor
// Template parameters: U is either CuCmplx<T> or T
// T is usually double. But why not char or short int? :D
template <typename T>
cuda_array<T> :: cuda_array(uint t, uint nx, uint my) :
    tlevs(t), Nx(nx), My(my), bounds(tlevs, Nx, My),
    array_d(NULL), array_d_t(NULL), array_d_t_host(NULL),
    array_h(NULL), array_h_t(NULL)
{
    // Determine grid size for kernel launch
    block = dim3(cuda::blockdim_nx, cuda::blockdim_my);
    // Testing, use 64 threads in y direction. Thus blockDim.y = 1
    // blockDim.x is the x, one block for each row
    // Round integer division for grid.y, see: http://stackoverflow.com/questions/2422712/c-rounding-integer-division-instead-of-truncating
    // a la int a = (59 + (4 - 1)) / 4;
    grid = dim3(Nx, (My + (cuda::blockdim_my - 1)) / cuda::blockdim_my);

    // Allocate device memory
    size_t nelem = tlevs * Nx * My;
    gpuErrchk(cudaMalloc( (void**) &array_d, nelem * sizeof(T)));
    gpuErrchk(cudaMalloc( (void***) &array_d_t, tlevs * sizeof(T*)));

#ifdef PINNED_HOST_MEMORY
    // Allocate pinned host memory for faster memory transfers
    gpuErrchk(cudaMallocHost( (void**) &array_h, nelem * sizeof(T)));
    // Initialize host memory all zero
    for(int n = 0; n < Nx; n++)
        for(int m = 0; m < My; m++)
            array_h[n * My + m] = 0.0;
#endif
#ifndef PINNED_HOST_MEMORY
    array_h = new T[nelem];
    for(int n = 0; n < Nx * My; n++)
        array_h[n] = T(0);
#endif

    // array_[hd]_t is an array of pointers allocated on the host/device respectively
    //array_h_t = (T**) calloc(tlevs, sizeof(T*));
    //array_d_t_host = (T**) calloc(tlevs, sizeof(T*));
    array_h_t = new T*[tlevs];
    array_d_t_host = new T*[tlevs];
    // array_t[i] points to the i-th time level
    // Set pointers on device
    d_alloc_array_d_t<<<1, tlevs>>>(array_d_t, array_d, tlevs, Nx, My);
    // Update host copy
    gpuErrchk(cudaMemcpy(array_d_t_host, array_d_t, sizeof(T*) * tlevs, cudaMemcpyDeviceToHost));

    for(uint tl = 0; tl < tlevs; tl++)
        array_h_t[tl] = &array_h[tl * Nx * My];
    set_all(0.0);
//#ifdef DEBUG
//    cout << "Array size: Nx=" << Nx << ", My=" << My << ", tlevs=" << tlevs << "\n";
//    cout << "cuda::cuda_blockdim_x = " << cuda::cuda_blockdim_nx;
//    cout << ", cuda::cuda_blockdim_y = " << cuda::cuda_blockdim_my << "\n";
//    cout << "blockDim=(" << block.x << ", " << block.y << ")\n";
//    cout << "gridDim=(" << grid.x << ", " << grid.y << ")\n";
//    cout << "Device data at " << array_d << "\t";
//    cout << "Host data at " << array_h << "\t";
//    cout << nelem << " bytes of data\n";
//    for (uint tl = 0; tl < tlevs; tl++)
//    {
//        cout << "time level " << tl << " at ";
//        cout << array_h_t[tl] << "(host)\t";
//        cout << array_d_t_host[tl] << "(device)\n";
//    }
//    cout << "Testing allocation of array_d_t:\n";
//    test_alloc<<<1, 1>>>(array_d_t, tlevs);
//#endif // DEBUG
}


// copy constructor
// OBS: Cuda does not support delegating constructors yet. So this is just copy
// and paste of the standard constructor plus a memcpy of the array data
template <typename T>
cuda_array<T> :: cuda_array(const cuda_array<T>& rhs) :
     tlevs(rhs.tlevs), Nx(rhs.Nx), My(rhs.My), bounds(tlevs, Nx, My),
     block(rhs.block), grid(rhs.grid), grid_full(rhs.grid_full),
    array_d(NULL), array_d_t(NULL), array_d_t_host(NULL),
    array_h(NULL), array_h_t(NULL)
{
    uint tl = 0;
    // Allocate device memory
    size_t nelem = tlevs * Nx * My;
    gpuErrchk(cudaMalloc( (void**) &array_d, nelem * sizeof(T)));
    // Allocate pinned host memory
#ifdef PINNED_HOST_MEMORY
    gpuErrchk(cudaMallocHost( (void**) &array_h, nelem * sizeof(T)));
    // Initialize host memory all zero
    for(int n = 0; n < Nx; n++)
        for(int m = 0; m < My; m++)
            array_h[n * My + m] = 0.0;
#endif
#ifndef PINNED_HOST_MEMORY
    //array_h = (T*) calloc(nelem, sizeof(T));
    array_h = new T[nelem];
#endif

    gpuErrchk(cudaMalloc( (void***) &array_d_t, tlevs * sizeof(T*)));
    //array_h_t = (T**) calloc(tlevs, sizeof(T*));
    //array_d_t_host = (T**) calloc(tlevs, sizeof(T*));
    array_h_t = new T*[tlevs];
    array_d_t_host = new T*[tlevs];
    // Set pointers on device
    d_alloc_array_d_t<<<1, tlevs>>>(array_d_t, array_d, tlevs, Nx, My);
    // Update host copy
    gpuErrchk(cudaMemcpy(array_d_t_host, array_d_t, sizeof(T*) * tlevs, cudaMemcpyDeviceToHost));

    // Set time levels on host and copy time slices from rhs array to
    // local array
    for(tl = 0; tl < tlevs; tl++)
    {
        gpuErrchk( cudaMemcpy(array_d_t_host[tl], rhs.get_array_d(tl), sizeof(T) * Nx * My, cudaMemcpyDeviceToDevice));
        array_h_t[tl] = &array_h[tl * Nx * My];
    }
    //cudaDeviceSynchronize();
}


// copy constructor
// OBS: Cuda does not support delegating constructors yet. So this is just copy
// and paste of the standard constructor plus a memcpy of the array data
template <typename T>
cuda_array<T> :: cuda_array(const cuda_array<T>* rhs) :
     tlevs(rhs -> tlevs), Nx(rhs -> Nx), My(rhs -> My), bounds(tlevs, Nx, My),
     block(rhs -> block), grid(rhs -> grid), grid_full(rhs -> grid_full),
    array_d(NULL), array_d_t(NULL), array_d_t_host(NULL),
    array_h(NULL), array_h_t(NULL)
{
    uint tl = 0;
    // Allocate device memory
    size_t nelem = tlevs * Nx * My;
    gpuErrchk(cudaMalloc( (void**) &array_d, nelem * sizeof(T)));
#ifdef PINNED_HOST_MEMORY
    // Allocate host memory
    gpuErrchk(cudaMallocHost( (void**) &array_h, nelem * sizeof(T)));
    // Initialize host memory all zero
    for(int n = 0; n < Nx; n++)
        for(int m = 0; m < My; m++)
            array_h[n * My + m] = 0.0;
#endif // PINNED_HOST_MEMORY
#ifndef PINNED_HOST_MEMORY
    //array_h = (T*) calloc(nelem, sizeof(T));
    array_h = new T[nelem];
#endif //PINNED_HOST_MEMORY
    //cerr << "cuda_array :: cuda_array() : allocated array_d_t at " << array_d << "\n";
    gpuErrchk(cudaMalloc( (void***) &array_d_t, tlevs * sizeof(T*)));
    //array_h_t = (T**) calloc(tlevs, sizeof(T*));
    //array_d_t_host = (T**) calloc(tlevs, sizeof(T*));
    array_h_t = new T*[tlevs];
    array_d_t_host = new T*[tlevs];
    // Set pointers on device
    d_alloc_array_d_t<<<1, tlevs>>>(array_d_t, array_d, tlevs, Nx, My);
    // Update host copy
    gpuErrchk(cudaMemcpy(array_d_t_host, array_d_t, sizeof(T*) * tlevs, cudaMemcpyDeviceToHost));

    // Set time levels on host and copy time slices from rhs array to
    // local array
    for(tl = 0; tl < tlevs; tl++)
    {
        gpuErrchk( cudaMemcpy(array_d_t_host[tl], rhs -> get_array_d(tl), sizeof(T) * Nx * My, cudaMemcpyDeviceToDevice));
        array_h_t[tl] = &array_h[tl * Nx * My];
    }
    //cudaDeviceSynchronize();
}

template <typename T>
cuda_array<T> :: ~cuda_array(){
    free(array_d_t_host);
    free(array_h_t);
    cudaFree(array_d_t);
    //gpuErrchk(cudaFree(array_d_t));
#ifdef PINNED_HOST_MEMORY
    cudaFreeHost(array_h);
#endif
#ifndef PINNED_HOST_MEMORY
    free(array_h);
#endif
    //cerr << " freeing array_d at " << array_d << "\n";
    //gpuErrchk(cudaFree(&array_d));
    cudaFree(array_d);
    //cudaDeviceSynchronize();
}


// Access functions for private members

template <typename T>
void cuda_array<T> :: enumerate_array(const uint t)
{
	if (!bounds(t, Nx-1, My-1))
		throw out_of_bounds_err(string("cuda_array<T> :: enumerate_array(const int): out of bounds\n"));
	d_enumerate<<<grid, block>>>(array_d, Nx, My);
	//cudaDeviceSynchronize();
}


template <typename T>
void cuda_array<T> :: enumerate_array_t(const uint t)
{
	if (!bounds(t, Nx-1, My-1))
		throw out_of_bounds_err(string("cuda_array<T> :: enumerate_array_t(const int): out of bounds\n"));
	d_enumerate_t<<<grid, block >>>(array_d_t, t, Nx, My);
#ifdef DEBUG
    gpuStatus();
#endif
}

// Operators

// Copy data from array_d_t[0] from rhs to lhs
template <typename T>
cuda_array<T>& cuda_array<T> :: operator= (const cuda_array<T>& rhs)
{
    // check bounds
    if (!bounds(rhs.get_tlevs(), rhs.get_nx(), rhs.get_my()))
        throw out_of_bounds_err(string("cuda_array<T>& cuda_array<T> :: operator= (const cuda_array<T>& rhs): out of bounds!"));
    // Check if we assign to ourself
    if ((void*) this == (void*) &rhs)
        return *this;

    // Copy data from other array
    gpuErrchk(cudaMemcpy(array_d_t_host[0], rhs.get_array_d(0), sizeof(T) * Nx * My, cudaMemcpyDeviceToDevice));
    // Leave the pointer to the time leves untouched!!!
    return *this;
}


template <typename T>
cuda_array<T>& cuda_array<T> :: operator= (const T& rhs)
{
    d_set_constant_t<<<grid, block>>>(array_d_t, rhs, 0, Nx, My);
#ifdef DEBUG
    gpuStatus();
#endif
    return *this;
}


// Set entire array to rhs
template <typename T>
cuda_array<T>& cuda_array<T> :: set_all(const T& rhs)
{
	for(uint t = 0; t < tlevs; t++)
    {
		d_set_constant_t<<<grid, block>>>(array_d_t, rhs, t, Nx, My);
#ifdef DEBUG
        gpuStatus();
#endif
    }

	return *this;
}


// Set array to rhs for time level tlev
template <typename T>
cuda_array<T>& cuda_array<T> :: set_t(const T& rhs, uint t)
{
    d_set_constant_t<<<grid, block>>>(array_d_t, rhs, t, Nx, My);
    return *this;
}


template <typename T>
cuda_array<T>& cuda_array<T> :: operator+=(const cuda_array<T>& rhs)
{
    if(!bounds(rhs.get_nx(), rhs.get_my()))
        throw out_of_bounds_err(string("cuda_array<T>& cuda_array<T> :: operator+= (const cuda_array<T>& rhs): out of bounds!"));
    if ((void*) this == (void*) &rhs)
        throw operator_err(string("cuda_array<T>& cuda_array<T> :: operator+= (const cuda_array<T>&): RHS and LHS cannot be the same\n"));

    d_op_arr_t<<<grid, block>>>(array_d_t, rhs.get_array_d_t(), 0, d_op_add<T>(), Nx, My);
#ifdef DEBUG
    gpuStatus();
#endif
    return *this;
}


template <typename T>
cuda_array<T>& cuda_array<T> :: operator+=(const T& rhs)
{
    d_op_scalar<<<grid, block>>>(array_d_t, rhs, d_op_add<T>(), Nx, My);
#ifdef DEBUG
    gpuStatus();
#endif
    return *this;
}


template <typename T>
cuda_array<T> cuda_array<T> :: operator+(const cuda_array<T>& rhs)
{
    cuda_array<T> result(this);
    result += rhs;
    return result;
}


template <typename T>
cuda_array<T> cuda_array<T> :: operator+(const T& rhs)
{
    cuda_array<T> result(this);
    result += rhs;
    return result;
}


template <typename T>
cuda_array<T>& cuda_array<T> :: operator-=(const cuda_array<T>& rhs)
{
    if(!bounds(rhs.get_nx(), rhs.get_my()))
        throw out_of_bounds_err(string("cuda_array<T>& cuda_array<T> :: operator-= (const cuda_array<T>& rhs): out of bounds!"));
    if ((void*) this == (void*) &rhs)
        throw operator_err(string("cuda_array<T>& cuda_array<T> :: operator-= (const cuda_array<T>&): RHS and LHS cannot be the same\n"));

    d_op_arr_t<<<grid, block>>>(array_d_t, rhs.get_array_d_t(), 0, d_op_sub<T>(), Nx, My);
#ifdef DEBUG
    gpuStatus();
#endif
    return *this;
}


template <typename T>
cuda_array<T>& cuda_array<T> :: operator-=(const T& rhs)
{
    d_op_scalar<<<grid, block>>>(array_d_t, rhs, d_op_sub<T>(), Nx, My);
#ifdef DEBUG
    gpuStatus();
#endif
    return *this;
}


template <typename T>
cuda_array<T> cuda_array<T> :: operator-(const cuda_array<T>& rhs)
{
    cuda_array<T> result(this);
    result -= rhs;
    return result;
}


template <typename T>
cuda_array<T> cuda_array<T> :: operator-(const T& rhs)
{
    cuda_array<T> result(this);
    result -= rhs;
    return result;
}


template <typename T>
cuda_array<T>& cuda_array<T> :: operator*=(const cuda_array<T>& rhs)
{
    if(!bounds(rhs.get_nx(), rhs.get_my()))
        throw out_of_bounds_err(string("cuda_array<T>& cuda_array<T> :: operator*= (const cuda_array<T>& rhs): out of bounds!"));
    if ((void*) this == (void*) &rhs)
        throw operator_err(string("cuda_array<T>& cuda_array<T> :: operator*= (const cuda_array<T>&): RHS and LHS cannot be the same\n"));

    d_op_arr_t<<<grid, block>>>(array_d_t, rhs.get_array_d_t(), 0, d_op_mul<T>(), Nx, My);
#ifdef DEBUG
    gpuStatus();
#endif
    return *this;
}


template <typename T>
cuda_array<T>& cuda_array<T> :: operator*=(const T& rhs)
{
    d_op_scalar<<<grid, block>>>(array_d_t, rhs, d_op_mul<T>(), Nx, My);
#ifdef DEBUG
    gpuStatus();
#endif
    return (*this);
}


template <typename T>
cuda_array<T> cuda_array<T> :: operator*(const cuda_array<T>& rhs)
{
    cuda_array<T> result(this);
    result *= rhs;
    return result;
}


template <typename T>
cuda_array<T> cuda_array<T> :: operator*(const T& rhs)
{
    cuda_array<T> result(this);
    result *= rhs;
    return result;
}


template <typename T>
cuda_array<T>& cuda_array<T> :: operator/=(const cuda_array<T>& rhs)
{
    if(!bounds(rhs.get_nx(), rhs.get_my()))
        throw out_of_bounds_err(string("cuda_array<T>& cuda_array<T> :: operator*= (const cuda_array<T>& rhs): out of bounds!"));
    if ((void*) this == (void*) &rhs)
        throw operator_err(string("cuda_array<T>& cuda_array<T> :: operator*= (const cuda_array<T>&): RHS and LHS cannot be the same\n"));

    d_op_arr_t<<<grid, block>>>(array_d_t, rhs.get_array_d_t(), 0, d_op_div<T>(), Nx, My);
#ifdef DEBUG
    gpuStatus();
#endif
    return *this;
}


template <typename T>
cuda_array<T>& cuda_array<T> :: operator/=(const T& rhs)
{
    d_op_scalar<<<grid, block>>>(array_d_t, rhs, d_op_div<T>(), Nx, My);
#ifdef DEBUG
    gpuStatus();
#endif
    return (*this);
}


template <typename T>
cuda_array<T> cuda_array<T> :: operator/(const cuda_array<T>& rhs)
{
    cuda_array<T> result(this);
    result /= rhs;
    return result;
}


template <typename T>
cuda_array<T> cuda_array<T> :: operator/(const T& rhs)
{
    cuda_array<T> result(this);
    result /= rhs;
    return result;
}


template <typename T>
inline T& cuda_array<T> :: operator()(uint t, uint n, uint m)
{
	if (!bounds(t, n, m))
		throw out_of_bounds_err(string("T& cuda_array<T> :: operator()(uint, uint, uint): out of bounds\n"));
	return (*(array_h_t[t] + address(n, m)));
}


template <typename T>
inline T cuda_array<T> :: operator()(uint t, uint n, uint m) const
{
	if (!bounds(t, n, m))
		throw out_of_bounds_err(string("T cuda_array<T> :: operator()(uint, uint, uint): out of bounds\n"));
	return (*(array_h_t[t] + address(n, m)));
}


template <typename T>
void cuda_array<T> :: advance()
{
	//Advance array_d_t pointer on device
    //cout << "advance\n";
    //cout << "before advancing\n";
    //for(int t = tlevs - 1; t >= 0; t--)
    //    cout << "array_d_t_host["<<t<<"] = " << array_d_t_host[t] << "\n";

    // Cycle pointer array on device and zero out last time level
	d_advance<<<1, 1>>>(array_d_t, tlevs);
    d_set_constant_t<<<grid, block>>>(array_d_t, T(0.0), 0, Nx, My);
    // Cycle pointer array on host
    T* tmp = array_d_t_host[tlevs - 1];
    for(int t = tlevs - 1; t > 0; t--)
        array_d_t_host[t] = array_d_t_host[t - 1];
    array_d_t_host[0] = tmp;

    //cout << "after advancing\n";
    //for(int t = tlevs - 1; t >= 0; t--)
    //     cout << "array_d_t_host["<<t<<"] = " << array_d_t_host[t] << "\n";

    // Zero out last time level
}


// Copy all time levels from device to host buffer
template <typename T>
inline void cuda_array<T> :: copy_device_to_host()
{
    //const size_t line_size = Nx * My * tlevs * sizeof(T);
    //gpuErrchk(cudaMemcpy(array_h, array_d, line_size, cudaMemcpyDeviceToHost));
    const size_t line_size = Nx * My * sizeof(T);
    for(uint t = 0; t < tlevs; t++)
    {
        gpuErrchk(cudaMemcpy(&array_h[t * (Nx * My)], array_d_t_host[t], line_size, cudaMemcpyDeviceToHost));
    }
}


// Copy from deivce to host buffer for specified time level
template <typename T>
inline void cuda_array<T> :: copy_device_to_host(uint tlev)
{
    const size_t line_size = Nx * My * sizeof(T);
    gpuErrchk(cudaMemcpy(array_h_t[tlev], array_d_t_host[tlev], line_size, cudaMemcpyDeviceToHost));
}


// Copy from device to rhs host buffer for time level =0
template <typename T>
inline void cuda_array<T> :: copy_device_to_host(T* rhs) 
{
    const size_t line_size = Nx * My * sizeof(T);
    gpuErrchk(cudaMemcpy(rhs, array_d_t_host[0], line_size, cudaMemcpyDeviceToHost));
}


template <typename T>
inline void cuda_array<T> :: copy(uint t_dst, uint t_src)
{
    const size_t line_size = Nx * My * sizeof(T);
    gpuErrchk(cudaMemcpy(array_d_t_host[t_dst], array_d_t_host[t_src], line_size, cudaMemcpyDeviceToDevice));
}


template <typename T>
inline void cuda_array<T> :: copy(uint t_dst, const cuda_array<T>& src, uint t_src)
{
    const size_t line_size = Nx * My * sizeof(T);
    gpuErrchk(cudaMemcpy(array_d_t_host[t_dst], src.get_array_d(t_src), line_size, cudaMemcpyDeviceToDevice));
}


template <typename T>
inline void cuda_array<T> :: move(uint t_dst, uint t_src)
{
    // Copy data
    const size_t line_size = Nx * My * sizeof(T);
    gpuErrchk(cudaMemcpy(array_d_t_host[t_dst], array_d_t_host[t_src], line_size, cudaMemcpyDeviceToDevice));
    // Clear source
    d_set_constant_t<<<grid, block>>>(array_d_t, T(0.0), t_src, Nx, My);
#ifdef DEBUG
    gpuStatus();
#endif
}


template <typename T>
inline void cuda_array<T> :: swap(uint t1, uint t2)
{
    d_swap<<<1, 1>>>(array_d_t, t1, t2);
    gpuErrchk(cudaMemcpy(array_d_t_host, array_d_t, sizeof(T*) * tlevs, cudaMemcpyDeviceToHost));
	//cudaDeviceSynchronize();
}


template <typename T>
inline void cuda_array<T> :: kill_kx0()
{
    d_kill_kx0<<<grid.x, block.x>>>(array_d, Nx, My);
#ifdef DEBUG
    gpuStatus();
#endif
}


/// Do this for T= CuCmplx<%>
template <typename T>
inline void cuda_array<T> :: kill_ky0()
{
    d_kill_ky0<<<grid.y, block.y>>>(array_d, Nx, My);
#ifdef DEBUG
    gpuStatus();
#endif
}


/// Do this for T = CuCmplx<%>
template < typename T>
inline void cuda_array<T> :: kill_k0()
{
    d_kill_k0<<<1, 1>>>(array_d);
#ifdef DEBUG
    gpuStatus();
#endif
}

/// Do this for U = double, T = double
template <typename T>
inline void cuda_array<T> :: normalize()
{
    cuda::real_t norm = 1. / T(Nx * My);
    d_op_scalar<<<grid, block>>>(array_d_t, norm, d_op_mul<T>(), Nx, My);
#ifdef DEBUG
    gpuStatus();
#endif
}

#endif // __CUDACC__


#endif /* CUDA_ARRAY4_H_ */
