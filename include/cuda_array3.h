/*
 * cuda_array2.h
 *
 *  Created on: Oct 22, 2013
 *      Author: rku000
 */


#ifndef CUDA_ARRAY3_H
#define CUDA_ARRAY3_H

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



/// U is the data type for this array, f.ex. CuCmplx<double>
/// T is the base type of U, for example U=CuCmplx<double>, T = double
template <typename U, typename T>
class cuda_array{
    public:
        // Explicitly declare construction operators that allocate memory
        // on the device
        cuda_array(uint, uint, uint);
        cuda_array(const cuda_array<U, T>&);
        cuda_array(const cuda_array<U, T>*);
        ~cuda_array();

        U* get_array_host(int) const;

        // Test function
        void enumerate_array(const uint);
        void enumerate_array_t(const uint);

        // Operators
        cuda_array<U, T>& operator=(const cuda_array<U, T>&);
        cuda_array<U, T>& operator=(const U&);

        cuda_array<U, T>& operator+=(const cuda_array<U, T>&);
        cuda_array<U, T>& operator+=(const U&);
        cuda_array<U, T> operator+(const cuda_array<U, T>&);

        cuda_array<U, T>& operator-=(const cuda_array<U, T>&);
        cuda_array<U, T>& operator-=(const U&);
        cuda_array<U, T> operator-(const cuda_array<U, T>&);

        cuda_array<U, T>& operator*=(const cuda_array<U, T>&);
        cuda_array<U, T>& operator*=(const U&);
        cuda_array<U, T> operator*(const cuda_array<U, T>&);

        cuda_array<U, T>& operator/=(const cuda_array<U, T>&);
        cuda_array<U, T>& operator/=(const U&);
        cuda_array<U, T> operator/(const cuda_array<U, T>&);

        // Similar to operator=, but operates on all time levels
        cuda_array<U, T>& set_all(const U&);
        // Set array to constant value for specified time level
        cuda_array<U, T>& set_t(const U&, uint);
        // Access operator to host array
        inline U& operator()(uint, uint, uint);
        inline U operator()(uint, uint, uint) const;

        // Copy device memory to host and print to stdout
        friend std::ostream& operator<<(std::ostream& os, cuda_array<U, T> src)
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

        // Transfer from host to device
        inline void copy_host_to_device();

        // Advance time levels
        inline void advance(); 
        
        inline void copy(uint, uint);
        inline void copy(uint, const cuda_array<U, T>&, uint);
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
        inline U* get_array_h() const {return array_h;};
        inline U* get_array_h(uint t) const {return array_h_t[t];};

        // Pointer to device data
        inline U* get_array_d() const {return array_d;};
        inline U** get_array_d_t() const {return array_d_t;};
        inline U* get_array_d(uint t) const {return array_d_t_host[t];};

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
        U* array_d;
        // Pointer to each time stage. Pointer to array of pointers on device
        U** array_d_t;
        // Pointer to each time stage: Pointer to each time level on host
        U** array_d_t_host;

        // Storage copy of device data on host
        U* array_h;
        U** array_h_t;
};

#ifdef __CUDACC__

__device__ inline int d_get_col() {
    return (blockIdx.y * blockDim.y + threadIdx.y);    
}


__device__ inline int d_get_row() {
    return (blockIdx.x * blockDim.x + threadIdx.x);
}


// Template kernel for d_enumerate_d and d_enumerate_c using the ca_val class
template <typename U>
__global__ void d_enumerate(U* array, uint Nx, uint My)
{
    const int col = d_get_col();
    const int row = d_get_row();
    const int index = row * (My) + col;

    if (index < Nx * My)
    {
        array[index] = U(index);
        //ca_val<T> val;
        //val.set(double(index));
        //array[index] = val.get();
    }
}


// Template version of d_enumerate_t_x
template <typename U>
__global__ void d_enumerate_t(U** array_t, uint t, uint Nx, uint My)
{
    const int col = d_get_col();
    const int row = d_get_row();
    const int index = row * My + col;

	//if (blockIdx.x + threadIdx.x + threadIdx.y == 0)
	//	printf("blockIdx.x = %d: enumerating at t = %d, at %p(device)\n", blockIdx.x, t, array_t[t]);
	if (index < Nx * My)
    {
        //ca_val<T> val;
        //val.set(double(index));
		//array_t[t][index] = val.get();
        array_t[t][index] = U(index);
    }
}

template <typename U, typename T>
__global__ void d_set_constant_t(U** array_t, T val, uint t, uint Nx, uint My)
{
    const int col = d_get_col();
    const int row = d_get_row();
	const int index = row * My + col;
	if (index < Nx * My)
		array_t[t][index] = val;
}


template <typename U>
__global__ void d_alloc_array_d_t(U** array_d_t, U* array,  uint tlevs,  uint Nx,  uint My)
{
	if (threadIdx.x < tlevs)
	{
		array_d_t[threadIdx.x] = &array[threadIdx.x * Nx * My];
		//printf("Device: array_d_t[%d] at %p\n", threadIdx.x, array_d_t[threadIdx.x]);
	}
}


template <typename U>
__global__ void d_advance(U** array_t, int tlevs)
//__global__ void d_advance(U** array_t, int tlevs)
{
	U* tmp = array_t[tlevs - 1];
	int t = 0;

//#ifdef DEBUG 
//	printf("__global__ d_advance: Before:\n");
//	for(t = tlevs - 1; t >= 0; t--)
//		printf("array_t[%d] at %p\n", t, array_t[t]);
//#endif
	for(t = tlevs - 1; t > 0; t--)
		array_t[t] = array_t[t - 1];
	array_t[0] = tmp;
//#ifdef DEBUG
//	printf("__global__ d_advance: After:\n");
//	for(t = tlevs - 1; t >= 0; t--)
//		printf("array_t[%d] at %p\n", t, array_t[t]);
//#endif
}


template <typename U>
__global__ void d_swap(U**array_t, uint t1, uint t2)
{
    U* tmp = array_t[t1];
    array_t[t1] = array_t[t2];
    array_t[t2] = tmp;
}



template <typename U>
__global__ void test_alloc(U** array_t,  uint tlevs)
{
    for(int t = 0; t < tlevs; t++)
        printf("array_x_t[%d] at %p\n", t, array_t[t]);
}


// Add two arrays: complex data, specify time levels for RHS
template <typename U>
__global__ 
void d_add_arr_t(U** lhs, U** rhs, uint tlev, uint Nx, uint My)
{
    const int col = d_get_col();
    const int row = d_get_row();
    const int idx = row * My + col;
    if ((col < My) && (row < Nx))
        lhs[0][idx] += rhs[tlev][idx];
}


// Add scalar
template <typename U>
__global__ 
void d_add_scalar(U** lhs, U rhs, uint Nx, uint My)
{
    const int col = d_get_col();
    const int row = d_get_row();
    const int idx = row * My + col;
    if ((col < My) && (row < Nx))
        lhs[0][idx] += rhs;
}


// Subtract two arrays 
template <typename U>
__global__ 
void d_sub_arr_t(U** lhs, U** rhs, uint tlev, uint Nx, uint My)
{
    const int col = d_get_col();
    const int row = d_get_row();
    const int idx = row * My + col;
    if ((col < My) && (row < Nx))
        lhs[0][idx] -= rhs[tlev][idx];
}


//Subtract scalar
template <typename U>
__global__ 
void d_sub_scalar(U** lhs, U rhs, uint Nx, uint My)
{
    const int col = d_get_col();
    const int row = d_get_row();
    const int idx = row * My + col;
    if ((col < My) && (row < Nx))
        lhs[0][idx] -= rhs;
}


// Multiply by array: real data, specify time level for RHS
template <typename U>
__global__ 
void d_mul_arr_t(U** lhs, U** rhs, uint tlev, uint Nx, uint My)
{
    const int col = d_get_col();
    const int row = d_get_row();
    const int idx = row * My + col;
    if ((col < My) && (row < Nx))
        lhs[0][idx] *= rhs[tlev][idx];
}


// Multiply by scalar
template <typename U>
__global__ 
void d_mul_scalar(U** lhs, U rhs, uint Nx, uint My)
{
    const int col = d_get_col();
    const int row = d_get_row();
    const int idx = row * My + col;
    if ((col < My) && (row < Nx))
        lhs[0][idx] *= rhs;
}


// Divide by array: real data, specify time level for RHS
template <typename U>
__global__
void d_div_arr_t(U** lhs, U** rhs, uint tlev, uint Nx, uint My)
{
    const int col = d_get_col();
    const int row = d_get_row();
    const int idx = row * My + col;
    if ((col < My) && (row < Nx))
        lhs[0][idx] /= rhs[tlev][idx];
}


// Divide by scalar
template <typename U>
__global__
void d_div_scalar(U** lhs, U rhs, uint Nx, uint My)
{
    const int col = d_get_col();
    const int row = d_get_row();
    const int idx = row * My + col;
    if ((col < My) && (row < Nx))
        lhs[0][idx] /= rhs;
}


// Kill kx = 0 mode
template <typename U>
__global__
void d_kill_kx0(U* arr, uint Nx, uint My)
{
    const int row = 0;
    const int col = d_get_col();
    const int idx = row * My + col;
    if (col > My)
        return;
    
    U zero(0.0);
    arr[idx] = zero;
}


// Kill ky=0 mode
template <typename U>
__global__
void d_kill_ky0(U* arr, uint Nx, uint My)
{
    const int row = d_get_row();
    const int col = 0;
    const int idx = row * My + col;
    if (row > Nx)
        return;
    
    U zero(0.0);
    arr[idx] = zero;
}


// Kill k=0 mode
// This should be only invoked where U = cuda::cmplx_t*
template <typename U>
__global__
void d_kill_k0(U* arr)
{
    U zero(0.0);
    arr[0] = zero;
}

// Default constructor
// Template parameters: U is either CuCmplx<T> or T
// T is usually double. But why not char or short int? :D
template <typename U, typename T>
cuda_array<U, T> :: cuda_array(uint t, uint nx, uint my) :
    tlevs(t), Nx(nx), My(my), bounds(tlevs, Nx, My),
    array_d(NULL), array_d_t(NULL), array_d_t_host(NULL),
    array_h(NULL), array_h_t(NULL)
{
    // Determine grid size for kernel launch
    block = dim3(cuda::cuda_blockdim_nx, cuda::cuda_blockdim_my);
    // Testing, use 64 threads in y direction. Thus blockDim.y = 1
    // blockDim.x is the x, one block for each row
    // Round integer division for grid.y, see: http://stackoverflow.com/questions/2422712/c-rounding-integer-division-instead-of-truncating
    // a la int a = (59 + (4 - 1)) / 4;
    grid = dim3(Nx, (My + (cuda::cuda_blockdim_my - 1)) / cuda::cuda_blockdim_my);

    // Allocate device memory
    size_t nelem = tlevs * Nx * My;
    gpuErrchk(cudaMalloc( (void**) &array_d, nelem * sizeof(U)));
    gpuErrchk(cudaMalloc( (void***) &array_d_t, tlevs * sizeof(U*)));
    array_h = (U*) calloc(nelem, sizeof(U));
    //cerr << "cuda_array :: cuda_array() at " << this << " allocated array_d at " << array_d << "\t";
    //cerr << "array_d_t at " << array_d_t << "\n";

    // array_[hd]_t is an array of pointers allocated on the host/device respectively
    array_h_t = (U**) calloc(tlevs, sizeof(U*));
    array_d_t_host = (U**) calloc(tlevs, sizeof(U*));
    // array_t[i] points to the i-th time level
    // Set pointers on device
    d_alloc_array_d_t<<<1, tlevs>>>(array_d_t, array_d, tlevs, Nx, My);
    // Update host copy
    gpuErrchk(cudaMemcpy(array_d_t_host, array_d_t, sizeof(U*) * tlevs, cudaMemcpyDeviceToHost));

    for(uint tl = 0; tl < tlevs; tl++)
        array_h_t[tl] = &array_h[tl * Nx * My];
    set_all(0.0);
    cudaDeviceSynchronize();
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
template <typename U, typename T>
cuda_array<U, T> :: cuda_array(const cuda_array<U, T>& rhs) :
     tlevs(rhs.tlevs), Nx(rhs.Nx), My(rhs.My), bounds(tlevs, Nx, My),
     block(rhs.block), grid(rhs.grid), grid_full(rhs.grid_full),
    array_d(NULL), array_d_t(NULL), array_d_t_host(NULL),
    array_h(NULL), array_h_t(NULL)
{
    uint tl = 0;
    // Allocate device memory
    size_t nelem = tlevs * Nx * My;
    gpuErrchk(cudaMalloc( (void**) &array_d, nelem * sizeof(U)));
    cerr << "cuda_array :: cuda_array() : allocated array_d at " << array_d << "\n";
    array_h = (U*) calloc(nelem, sizeof(U));

    gpuErrchk(cudaMalloc( (void***) &array_d_t, tlevs * sizeof(U*)));
    array_h_t = (U**) calloc(tlevs, sizeof(U*));
    array_d_t_host = (U**) calloc(tlevs, sizeof(U*));
    // Set pointers on device
    d_alloc_array_d_t<<<1, tlevs>>>(array_d_t, array_d, tlevs, Nx, My);
    // Update host copy
    gpuErrchk(cudaMemcpy(array_d_t_host, array_d_t, sizeof(U*) * tlevs, cudaMemcpyDeviceToHost));

    // Set time levels on host and copy time slices from rhs array to
    // local array
    for(tl = 0; tl < tlevs; tl++)
    {   
        gpuErrchk( cudaMemcpy(array_d_t_host[tl], rhs.get_array_d(tl), sizeof(U) * Nx * My, cudaMemcpyDeviceToDevice));
        array_h_t[tl] = &array_h[tl * Nx * My];
    }
    cudaDeviceSynchronize();
}


// copy constructor
// OBS: Cuda does not support delegating constructors yet. So this is just copy
// and paste of the standard constructor plus a memcpy of the array data
template <typename U, typename T>
cuda_array<U, T> :: cuda_array(const cuda_array<U, T>* rhs) :
     tlevs(rhs -> tlevs), Nx(rhs -> Nx), My(rhs -> My), bounds(tlevs, Nx, My),
     block(rhs -> block), grid(rhs -> grid), grid_full(rhs -> grid_full),
    array_d(NULL), array_d_t(NULL), array_d_t_host(NULL),
    array_h(NULL), array_h_t(NULL)
{
    uint tl = 0;
    // Allocate device memory
    size_t nelem = tlevs * Nx * My;
    gpuErrchk(cudaMalloc( (void**) &array_d, nelem * sizeof(U)));
    //cerr << "cuda_array :: cuda_array() : allocated array_d at " << array_d << "\n";
    array_h = (U*) calloc(nelem, sizeof(U));

    //cerr << "cuda_array :: cuda_array() : allocated array_d_t at " << array_d << "\n";
    gpuErrchk(cudaMalloc( (void***) &array_d_t, tlevs * sizeof(U*)));
    array_h_t = (U**) calloc(tlevs, sizeof(U*));
    array_d_t_host = (U**) calloc(tlevs, sizeof(U*));
    // Set pointers on device
    d_alloc_array_d_t<<<1, tlevs>>>(array_d_t, array_d, tlevs, Nx, My);
    // Update host copy
    gpuErrchk(cudaMemcpy(array_d_t_host, array_d_t, sizeof(U*) * tlevs, cudaMemcpyDeviceToHost));

    // Set time levels on host and copy time slices from rhs array to
    // local array
    for(tl = 0; tl < tlevs; tl++)
    {   
        gpuErrchk( cudaMemcpy(array_d_t_host[tl], rhs -> get_array_d(tl), sizeof(U) * Nx * My, cudaMemcpyDeviceToDevice));
        array_h_t[tl] = &array_h[tl * Nx * My];
    }
    cudaDeviceSynchronize();
}

template <typename U, typename T>
cuda_array<U, T> :: ~cuda_array(){
    free(array_d_t_host);
    free(array_h_t);
    //cerr << "cuda_array :: ~cuda_array()  at " << this << " freeing array_d_t at " << array_d_t << "\t";
    cudaFree(array_d_t);
    //gpuErrchk(cudaFree(array_d_t));
    free(array_h);
    //cerr << " freeing array_d at " << array_d << "\n";
    //gpuErrchk(cudaFree(&array_d));
    cudaFree(array_d);
    cudaDeviceSynchronize();
}


// Access functions for private members

template <typename U, typename T>
void cuda_array<U, T> :: enumerate_array(const uint t)
{
	if (!bounds(t, Nx-1, My-1))
		throw out_of_bounds_err(string("cuda_array<U, T> :: enumerate_array(const int): out of bounds\n"));
	d_enumerate<<<grid, block>>>(array_d, Nx, My);
	cudaDeviceSynchronize();
}


template <typename U, typename T>
void cuda_array<U, T> :: enumerate_array_t(const uint t)
{
	if (!bounds(t, Nx-1, My-1))
		throw out_of_bounds_err(string("cuda_array<U, T> :: enumerate_array_t(const int): out of bounds\n"));
	d_enumerate_t<<<grid, block >>>(array_d_t, t, Nx, My);
#ifdef DEBUG
    gpuStatus();
#endif
}

// Operators

// Copy data fro array_d_t[0] from rhs to lhs
template <typename U, typename T>
cuda_array<U, T>& cuda_array<U, T> :: operator= (const cuda_array<U, T>& rhs)
{
    // check bounds
    if (!bounds(rhs.get_tlevs(), rhs.get_nx(), rhs.get_my()))
        throw out_of_bounds_err(string("cuda_array<U, T>& cuda_array<U, T> :: operator= (const cuda_array<U, T>& rhs): out of bounds!"));
    // Check if we assign to ourself
    if ((void*) this == (void*) &rhs)
        return *this;
    
    // Copy data from other array
    gpuErrchk(cudaMemcpy(array_d_t_host[0], rhs.get_array_d(0), sizeof(T) * Nx * My, cudaMemcpyDeviceToDevice));
    // Leave the pointer to the time leves untouched!!!
    return *this;
}


template <typename U, typename T>
cuda_array<U, T>& cuda_array<U, T> :: operator= (const U& rhs)
{
    d_set_constant_t<<<grid, block>>>(array_d_t, rhs, 0, Nx, My);
#ifdef DEBUG
    gpuStatus();
#endif
    return *this;
}


// Set entire array to rhs
template <typename U, typename T>
cuda_array<U, T>& cuda_array<U, T> :: set_all(const U& rhs)
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
template <typename U, typename T>
cuda_array<U, T>& cuda_array<U, T> :: set_t(const U& rhs, uint t)
{
    d_set_constant_t<<<grid, block>>>(array_d_t, rhs, t, Nx, My);
    return *this;
}


template <typename U, typename T>
cuda_array<U, T>& cuda_array<U, T> :: operator+=(const cuda_array<U, T>& rhs)
{
    if(!bounds(rhs.get_nx(), rhs.get_my()))
        throw out_of_bounds_err(string("cuda_array<T>& cuda_array<T> :: operator+= (const cuda_array<T>& rhs): out of bounds!"));
    if ((void*) this == (void*) &rhs)
        throw operator_err(string("cuda_array<T>& cuda_array<T> :: operator+= (const cuda_array<T>&): RHS and LHS cannot be the same\n"));

    d_add_arr_t<<<grid, block>>>(array_d_t, rhs.get_array_d_t(), 0, Nx, My);
#ifdef DEBUG
    gpuStatus();
#endif
    return *this;
}


template <typename U, typename T>
cuda_array<U, T>& cuda_array<U, T> :: operator+=(const U& rhs)
{
    d_add_scalar<<<grid, block>>>(array_d_t, rhs, Nx, My);
#ifdef DEBUG
    gpuStatus();
#endif
    return *this;
}


template <typename U, typename T>
cuda_array<U, T> cuda_array<U, T> :: operator+(const cuda_array<U, T>& rhs)
{
    cuda_array<U, T> result(this);
    result += rhs;
    return result;
}


template <typename U, typename T>
cuda_array<U, T>& cuda_array<U, T> :: operator-=(const cuda_array<U, T>& rhs)
{
    if(!bounds(rhs.get_nx(), rhs.get_my()))
        throw out_of_bounds_err(string("cuda_array<T>& cuda_array<T> :: operator-= (const cuda_array<T>& rhs): out of bounds!"));
    if ((void*) this == (void*) &rhs)
        throw operator_err(string("cuda_array<T>& cuda_array<T> :: operator-= (const cuda_array<T>&): RHS and LHS cannot be the same\n"));
    
    d_sub_arr_t<<<grid, block>>>(array_d_t, rhs.get_array_d_t(), 0, Nx, My);
#ifdef DEBUG
    gpuStatus();
#endif
    return *this;
}


template<typename U, typename T>
cuda_array<U, T>& cuda_array<U, T> :: operator-=(const U& rhs)
{
    d_sub_scalar<<<grid, block>>>(array_d_t, rhs, Nx, My);
#ifdef DEBUG
    gpuStatus();
#endif
    return *this;
}


template <typename U, typename T>
cuda_array<U, T> cuda_array<U, T> :: operator-(const cuda_array<U, T>& rhs)
{
    cuda_array<U, T> result(this);
    result -= rhs;
    return result;
}


template <typename U, typename T>
cuda_array<U, T>& cuda_array<U, T> :: operator*=(const cuda_array<U, T>& rhs)
{
    if(!bounds(rhs.get_nx(), rhs.get_my()))
        throw out_of_bounds_err(string("cuda_array<T>& cuda_array<T> :: operator*= (const cuda_array<T>& rhs): out of bounds!"));
    if ((void*) this == (void*) &rhs)
        throw operator_err(string("cuda_array<T>& cuda_array<T> :: operator*= (const cuda_array<T>&): RHS and LHS cannot be the same\n"));
    
    d_mul_arr_t<<<grid, block>>>(array_d_t, rhs.get_array_d_t(), 0, Nx, My);
#ifdef DEBUG
    gpuStatus();
#endif
    return *this;
}
      

template <typename U, typename T>
cuda_array<U, T>& cuda_array<U, T> :: operator*=(const U& rhs)
{
    d_mul_scalar<<<grid, block>>>(array_d_t, rhs, Nx, My);
#ifdef DEBUG
    gpuStatus();
#endif
    return (*this);
}


template <typename U, typename T>
cuda_array<U, T> cuda_array<U, T> :: operator*(const cuda_array<U, T>& rhs)
{
    cuda_array<U, T> result(this);
    result *= rhs;
    return result;
}


template <typename U, typename T>
cuda_array<U, T>& cuda_array<U, T> :: operator/=(const cuda_array<U, T>& rhs)
{
    if(!bounds(rhs.get_nx(), rhs.get_my()))
        throw out_of_bounds_err(string("cuda_array<T>& cuda_array<T> :: operator*= (const cuda_array<T>& rhs): out of bounds!"));
    if ((void*) this == (void*) &rhs)
        throw operator_err(string("cuda_array<T>& cuda_array<T> :: operator*= (const cuda_array<T>&): RHS and LHS cannot be the same\n"));
    
    d_div_arr_t<<<grid, block>>>(array_d_t, rhs.get_array_d_t(), 0, Nx, My);
#ifdef DEBUG
    gpuStatus();
#endif
    return *this;
}
      

template <typename U, typename T>
cuda_array<U, T>& cuda_array<U, T> :: operator/=(const U& rhs)
{
    d_div_scalar<<<grid, block>>>(array_d_t, rhs, Nx, My);
#ifdef DEBUG
    gpuStatus();
#endif
    return (*this);
}


template <typename U, typename T>
cuda_array<U, T> cuda_array<U, T> :: operator/(const cuda_array<U, T>& rhs)
{
    cuda_array<U, T> result(this);
    result /= rhs;
    return result;
}


template <typename U, typename T>
inline U& cuda_array<U, T> :: operator()(uint t, uint n, uint m)
{
	if (!bounds(t, n, m))
		throw out_of_bounds_err(string("T& cuda_array<T> :: operator()(uint, uint, uint): out of bounds\n"));
	return (*(array_h_t[t] + address(n, m)));
}


template <typename U, typename T>
inline U cuda_array<U, T> :: operator()(uint t, uint n, uint m) const
{
	if (!bounds(t, n, m))
		throw out_of_bounds_err(string("T cuda_array<T> :: operator()(uint, uint, uint): out of bounds\n"));
	return (*(array_h_t[t] + address(n, m)));
}

// Hardcodde for tlevs = 4.
/*
template <typename U, typename T>
void cuda_array<U, T> :: advance4()
{
    d_advance<<<1, 1>>>(array_d_t, 4);
    d_set_constant_t<<<grid, block>>>(array_d_t, 0.0, 0, Nx, My);

    cout << "advance\n";
    cout << "before advancing\n";
    for(int t = tlevs ; t >= 0; t--)
        cout << "array_d_t_host["<<t<<"] = " << array_d_t_host[t] << "\n";

    U* tmp = array_d_t_host[3];
    array_d_t_host[3] = array_d_t_host[2];
    array_d_t_host[2] = array_d_t_host[1];
    array_d_t_host[1] = array_d_t_host[0];
    array_d_t_host[0] = tmp;
    cout << "after advancing\n";
    for(int t = tlevs; t >= 0; t--)
        cout << "array_d_t_host["<<t<<"] = " << array_d_t_host[t] << "\n";


}
*/

template <typename U, typename T>
void cuda_array<U, T> :: advance()
{
	//Advance array_d_t pointer on device
    //cout << "advance\n";
    //cout << "before advancing\n";
    //for(int t = tlevs - 1; t >= 0; t--)
    //    cout << "array_d_t_host["<<t<<"] = " << array_d_t_host[t] << "\n";

    // Cycle pointer array on device and zero out last time level
	d_advance<<<1, 1>>>(array_d_t, tlevs);
    d_set_constant_t<<<grid, block>>>(array_d_t, 0.0, 0, Nx, My);
    // Cycle pointer array on host
    U* tmp = array_d_t_host[tlevs - 1];
    for(int t = tlevs - 1; t > 0; t--)
        array_d_t_host[t] = array_d_t_host[t - 1];
    array_d_t_host[0] = tmp;

    //cout << "after advancing\n";
    //for(int t = tlevs - 1; t >= 0; t--)
    //     cout << "array_d_t_host["<<t<<"] = " << array_d_t_host[t] << "\n";

    // Zero out last time level 
}


// The array is contiguous 
template <typename U, typename T>
inline void cuda_array<U, T> :: copy_device_to_host() 
{
    //const size_t line_size = Nx * My * tlevs * sizeof(T);
    //gpuErrchk(cudaMemcpy(array_h, array_d, line_size, cudaMemcpyDeviceToHost));
    const size_t line_size = Nx * My * sizeof(U);
    for(uint t = 0; t < tlevs; t++)
    {
        gpuErrchk(cudaMemcpy(&array_h[t * (Nx * My)], array_d_t_host[t], line_size, cudaMemcpyDeviceToHost));
    }
}


template <typename U, typename T>
inline void cuda_array<U, T> :: copy_device_to_host(uint tlev)
{
    const size_t line_size = Nx * My * sizeof(U);
    gpuErrchk(cudaMemcpy(array_h_t[tlev], array_d_t_host[tlev], line_size, cudaMemcpyDeviceToHost));
}


template <typename U, typename T>
inline void cuda_array<U, T> :: copy(uint t_dst, uint t_src)
{
    const size_t line_size = Nx * My * sizeof(U);
    gpuErrchk(cudaMemcpy(array_d_t_host[t_dst], array_d_t_host[t_src], line_size, cudaMemcpyDeviceToDevice));
}


template <typename U, typename T>
inline void cuda_array<U, T> :: copy(uint t_dst, const cuda_array<U, T>& src, uint t_src)
{
    const size_t line_size = Nx * My * sizeof(U);
    gpuErrchk(cudaMemcpy(array_d_t_host[t_dst], src.get_array_d(t_src), line_size, cudaMemcpyDeviceToDevice));
}


template <typename U, typename T>
inline void cuda_array<U, T> :: move(uint t_dst, uint t_src)
{
    // Copy data 
    const size_t line_size = Nx * My * sizeof(U);
    gpuErrchk(cudaMemcpy(array_d_t_host[t_dst], array_d_t_host[t_src], line_size, cudaMemcpyDeviceToDevice));
    // Clear source
    d_set_constant_t<<<grid, block>>>(array_d_t, 0.0, t_src, Nx, My);
#ifdef DEBUG
    gpuStatus();
#endif
}


template <typename U, typename T>
inline void cuda_array<U, T> :: swap(uint t1, uint t2)
{
    d_swap<<<1, 1>>>(array_d_t, t1, t2);
    gpuErrchk(cudaMemcpy(array_d_t_host, array_d_t, sizeof(U*) * tlevs, cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();
}


template <typename U, typename T>
inline void cuda_array<U, T> :: kill_kx0()
{
    d_kill_kx0<<<grid.x, block.x>>>(array_d, Nx, My);
#ifdef DEBUG
    gpuStatus();
#endif
}


/// Do this for U = CuCmplx<T>
template <typename U, typename T>
inline void cuda_array<U, T> :: kill_ky0()
{
    d_kill_ky0<<<grid.y, block.y>>>(array_d, Nx, My);
#ifdef DEBUG
    gpuStatus();
#endif
}


/// Do this for U = CuCmplx<T>
template <typename U, typename T>
inline void cuda_array<U, T> :: kill_k0()
{
    d_kill_k0<<<1, 1>>>(array_d);
#ifdef DEBUG
    gpuStatus();
#endif
}

/// Do this for U = double, T = double
template <typename U, typename T>
inline void cuda_array<U, T> :: normalize()
{
    cuda::real_t norm = 1. / U(Nx * My);
    d_mul_scalar<<<grid, block>>>(array_d_t, norm, Nx, My);
#ifdef DEBUG
    gpuStatus();
#endif
}

#endif // CUDA_CC

#endif // CUDA_ARRAY3_H 
