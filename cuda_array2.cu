/*
 *
 *  Created on: Oct 22, 2013
 *      Author: Ralph Kube
 *
 *  cuda_array2.cu
 *
 *  Implementation of cuda_array
 *
 */

#include <iostream>
#include "include/cuda_array2.h"

using namespace std;

__device__ int d_get_col(void)
{
    return (blockIdx.y * blockDim.y + threadIdx.y);
}


__device__ int d_get_row(void)
{
    return (blockIdx.x * blockDim.x + threadIdx.x);
}

// Template kernel for d_enumerate_d and d_enumerate_c using the ca_val class
template <typename T>
__global__ void d_enumerate(T* array, int Nx, int My)
{
    const int col = d_get_col();
    const int row = d_get_row();
    const int index = row * (My) + col;

    if (index < Nx * My)
    {
        ca_val<T> val;
        val.set(double(index));
        array[index] = val.get();
    }
}


// Template version of d_enumerate_t_x
template <typename T>
__global__ void d_enumerate_t(T** array_t, int t, int Nx, int My)
{
    //const int col = blockIdx.y * blockDim.y + threadIdx.y;
    //const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = d_get_col();
    const int row = d_get_row();
    const int index = row * My + col;

	if (blockIdx.x + threadIdx.x + threadIdx.y == 0)
		printf("blockIdx.x = %d: enumerating at t = %d, at %p(device)\n", blockIdx.x, t, array_t[t]);
    //printf("blockIdx.x = %d, blockDim.x = %d, threadIdx.x = %d,  blockIdy.y = %d, blockDim.y = %d, threadIdy.y = %d ,index = %d\n", blockIdx.x, blockDim.x, threadIdx.x,   blockIdx.y, blockDim.y, threadIdx.y, index);
	if (index < Nx * My)
    {
        ca_val<T> val;
        val.set(double(index));
		array_t[t][index] = val.get();
    }
}


template <typename T>
__global__ void d_set_constant_t(T** array_t, T val, int t, int Nx, int My)
{
	//const int col = blockIdx.y * blockDim.y + threadIdx.y;
	//const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = d_get_col();
    const int row = d_get_row();
	const int index = row * My + col;
	if (index < Nx * My)
		array_t[t][index] = val;
}


template <typename T>
__global__ void d_alloc_array_d_t(T** array_d_t, T* array, int tlevs, int Nx, int My)
{
	if (threadIdx.x < tlevs)
	{
		array_d_t[threadIdx.x] = &array[threadIdx.x * Nx * My];
		printf("Device: array_d_t[%d] at %p\n", threadIdx.x, array_d_t[threadIdx.x]);
	}
}


template <typename T>
__global__ void d_advance(T** array_t, int tlevs)
{
	T* tmp = array_t[tlevs - 1];
	unsigned int t = 0;

    
	printf("Before:\n");
	for(t = tlevs - 1; t > 0; t--)
		printf("array_t[%d] at %p\n", t, array_t[t]);
	printf("array_t[0] at %p\n", array_t[0]);
    
	for(t = tlevs - 1; t > 0; t--)
		array_t[t] = array_t[t - 1];
	array_t[0] = tmp;

    
	printf("After:\n");
	for(t = tlevs - 1; t > 0; t--)
		printf("array_t[%d] at %p\n", t, array_t[t]);
	printf("array_t[0] at %p\n", array_t[0]);
    
}


template <typename T>
__global__ void test_alloc(T** array_t, int tlevs)
{
    for(int t = 0; t < tlevs; t++)
        printf("array_x_t[%d] at %p\n", t, array_t[t]);
}


// Add two arrays, real data, one given time level
__global__ void d_add_t(cuda::real_t** lhs, cuda::real_t** rhs, int tlev, int Nx, int My)
{
    const int col = d_get_col();
    const int row = d_get_row();
    const int idx = row * My + col;
    if ((col < My) && (row < Nx))
        lhs[tlev][idx] = lhs[tlev][idx] + rhs[tlev][idx];
}


// Add two arrays: complex data, specified time levels
__global__ void d_add_t(cuda::cmplx_t** lhs, cuda::cmplx_t** rhs, int tlev, int Nx, int My)
{
    const int col = d_get_col();
    const int row = d_get_row();
    const int idx = row * My + col;
    if ((col < My) && (row < Nx))
        lhs[tlev][idx] = cuCadd(lhs[tlev][idx], rhs[tlev][idx]);
}


// Subtract two arrays: real data, specified time levels
__global__ void d_sub_t(cuda::real_t** lhs, cuda::real_t** rhs, int tlev, int Nx, int My)
{
    const int col = d_get_col();
    const int row = d_get_row();
    const int idx = row * My + col;
    if ((col < My) && (row < Nx))
        lhs[tlev][idx] = lhs[tlev][idx] - rhs[tlev][idx];
}


// Subtract two arrays: Complex data, specified time level
__global__ void d_sub_t(cuda::cmplx_t** lhs, cuda::cmplx_t** rhs, int tlev, int Nx, int My)
{
    const int col = d_get_col();
    const int row = d_get_row();
    const int idx = row * My + col;
    if ((col < My) && (row < Nx))
        lhs[tlev][idx] = cuCsub(lhs[tlev][idx], rhs[tlev][idx]);
}


// Multiply two arrays: real data, specified time level
__global__ void d_mul_t(cuda::real_t** lhs, cuda::real_t** rhs, int tlev, int Nx, int My)
{
    const int col = d_get_col();
    const int row = d_get_row();
    const int idx = row * My + col;
    if ((col < My) && (row < Nx))
        lhs[tlev][idx] = lhs[tlev][idx] * rhs[tlev][idx];
}


// Multiply two arrays: Complex data, specified time level
__global__ void d_mul_t(cuda::cmplx_t** lhs, cuda::cmplx_t** rhs, int tlev, int Nx, int My)
{
    const int col = d_get_col();
    const int row = d_get_row();
    const int idx = row * My + col;
    if ((col < My) && (row < Nx))
        lhs[tlev][idx] = cuCmul(lhs[tlev][idx], rhs[tlev][idx]);
}



// Default constructor
template <class T>
cuda_array<T> :: cuda_array(unsigned int t, unsigned int nx, unsigned int my) :
    tlevs(t), Nx(nx), My(my), bounds(tlevs, Nx, My),
    array_d(NULL), array_d_t(NULL),
    array_h(NULL), array_h_t(NULL)
{
    // Determine grid size for kernel launch
    block = dim3(cuda::cuda_blockdim_nx, cuda::cuda_blockdim_my);
    // Testing, use 64 threads in y direction. Thus blockDim.y = 1
    // blockDim.x is the x, one block for each row
    grid = dim3(Nx, 1);
    grid_full = dim3(tlevs * Nx, 1);
    cout << "blockDim=(" << block.x << ", " << block.y << ", " << block.z << ")\n";
    cout << "gridDim=(" << grid.x << ", " << grid.y << ", " << grid.z << ")\n";

    cout << "Array size: Nx=" << Nx << ", My=" << My << ", tlevs=" << tlevs << "\n";
    // Allocate device memory
    size_t nelem = tlevs * Nx * My;
    gpuErrchk(cudaMalloc( (void**) &array_d, nelem * sizeof(T)));
    array_h = (T*) calloc(nelem, sizeof(T));

    cout << "Device data at " << array_d << "\t";
    cout << "Host data at " << array_h << "\t";
    cout << nelem << " bytes of data\n";

    // array_[hd]_t is an array of pointers allocated on the host/device respectively
    gpuErrchk(cudaMalloc( (void***) &array_d_t, tlevs * sizeof(T*)));
    array_h_t = (T**) calloc(tlevs, sizeof(T*));
    array_d_t_host = (T**) calloc(tlevs, sizeof(T*));
    // array_t[i] points to the i-th time level
    // Set pointers on device
    d_alloc_array_d_t<<<1, tlevs>>>(array_d_t, array_d, tlevs, Nx, My);
    // Update host copy
    gpuErrchk(cudaMemcpy(array_d_t_host, array_d_t, sizeof(T*) * tlevs, cudaMemcpyDeviceToHost));

    for(unsigned int tl = 0; tl < tlevs; tl++)
    {
        array_h_t[tl] = &array_h[tl * Nx * My];
        cout << "time level " << tl << " at ";
        cout << array_h_t[tl] << "(host)\t";
        cout << array_d_t_host[tl] << "(device)\n";
    }

    cout << "Testing allocation of array_d_t:\n";
    test_alloc<<<1, 1>>>(array_d_t, tlevs);
}

template <class T>
cuda_array<T> :: ~cuda_array(){
    cudaFree(array_d);
    free(array_h);
    cudaFree(array_d_t);
    free(array_h_t);
    free(array_d_t_host);
}


// Access functions for private members

template <typename T>
void cuda_array<T> :: enumerate_array(const int t)
{
	if (!bounds(t, Nx-1, My-1))
		throw out_of_bounds_err(string("T& cuda_array<T> :: enumerate_array(const int): out of bounds\n"));
	d_enumerate<<<grid, block>>>(array_d, Nx, My);
	cudaDeviceSynchronize();
}


template <typename T>
void cuda_array<T> :: enumerate_array_t(const int t)
{
	if (!bounds(t, Nx-1, My-1))
		throw out_of_bounds_err(string("T& cuda_array<T> :: enumerate_array_t(const int): out of bounds\n"));
	d_enumerate_t<<<grid, block >>>(array_d_t, t, Nx, My);
	cudaDeviceSynchronize();
}

// Operators

// Copy data fro array_d_t[0] from rhs to lhs
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
    //gpuErrchk(cudaMemcpy(array_d, rhs.get_array_d(), sizeof(T*) * tlevs * Nx * My, cudaMemcpyDeviceToDevice));
    gpuErrchk(cudaMemcpy(array_d_t_host[0], rhs.get_array_d(0), sizeof(T) * Nx * My, cudaMemcpyDeviceToDevice));
    // Leave the pointer to the time leves untouched!!!
    return *this;
}

// Set entire array to rhs
template <typename T>
cuda_array<T>& cuda_array<T> :: operator= (const T& rhs)
{
    //ca_val<T> val;
    //val.set(rhs);
    d_set_constant_t<<<grid, block>>>(array_d_t, rhs, 0, Nx, My);
    cudaDeviceSynchronize();
    return *this;
}


template <class T>
cuda_array<T>& cuda_array<T> :: set_all(const T& rhs)
{
	for(unsigned int t = 0; t < tlevs; t++)
	{
		d_set_constant_t<<<grid, block>>>(array_d_t, rhs, t, Nx, My);
		cudaDeviceSynchronize();
	}
	return *this;
}

template <typename T>
cuda_array<T>& cuda_array<T> :: operator+=(const cuda_array<T>& rhs)
{
    if(!bounds(rhs.get_nx(), rhs.get_my()))
        throw out_of_bounds_err(string("cuda_array<T>& cuda_array<T> :: operator+= (const cuda_array<T>& rhs): out of bounds!"));
    if ((void*) this == (void*) &rhs)
        throw operator_err(string("cuda_array<T>& cuda_array<T> :: operator+= (const cuda_array<T>&): RHS and LHS cannot be the same\n"));

    d_add_t<<<grid, block>>>(array_d_t, rhs.get_array_d_t(), 0, Nx, My);
    cudaDeviceSynchronize();
    return *this;
}


template <typename T>
cuda_array<T>& cuda_array<T> :: operator-=(const cuda_array<T>& rhs)
{
    if(!bounds(rhs.get_nx(), rhs.get_my()))
        throw out_of_bounds_err(string("cuda_array<T>& cuda_array<T> :: operator-= (const cuda_array<T>& rhs): out of bounds!"));
    if ((void*) this == (void*) &rhs)
        throw operator_err(string("cuda_array<T>& cuda_array<T> :: operator-= (const cuda_array<T>&): RHS and LHS cannot be the same\n"));
    
    d_sub_t<<<grid, block>>>(array_d_t, rhs.get_array_d_t(), 0, Nx, My);
    cudaDeviceSynchronize();
    return *this;
}


template <typename T>
cuda_array<T>& cuda_array<T> :: operator*=(const cuda_array<T>& rhs)
{
    if(!bounds(rhs.get_nx(), rhs.get_my()))
        throw out_of_bounds_err(string("cuda_array<T>& cuda_array<T> :: operator*= (const cuda_array<T>& rhs): out of bounds!"));
    if ((void*) this == (void*) &rhs)
        throw operator_err(string("cuda_array<T>& cuda_array<T> :: operator*= (const cuda_array<T>&): RHS and LHS cannot be the same\n"));
    
    d_mul_t<<<grid, block>>>(array_d_t, rhs.get_array_d_t(), 0, Nx, My);
    cudaDeviceSynchronize();
    return *this;
}
       

template <class T>
T& cuda_array<T> :: operator()(unsigned int t, unsigned int n, unsigned int m)
{
	if (!bounds(t, n, m))
		throw out_of_bounds_err(string("T& cuda_array<T> :: operator()(uint, uint, uint): out of bounds\n"));
	return (*(array_h_t[t] + address(n, m)));
}


template <class T>
T cuda_array<T> :: operator()(unsigned int t, unsigned int n, unsigned int m) const
{
	if (!bounds(t, n, m))
		throw out_of_bounds_err(string("T cuda_array<T> :: operator()(uint, uint, uint): out of bounds\n"));
	return (*(array_h_t[t] + address(n, m)));
}


template <class T>
void cuda_array<T> :: advance()
{
	//Advance array_d_t pointer on device
	d_advance<<<1, 1>>>(array_d_t, tlevs);
	cudaDeviceSynchronize();
    // Zero out last time level for visualization purpose
    ca_val<T> val;
    val.set(double(0.0));
    d_set_constant_t<<<grid, block>>>(array_d_t, val.get(), tlevs - 1, Nx, My);
    // Update array_d_t_host
    gpuErrchk(cudaMemcpy(array_d_t_host, array_d_t, sizeof(T*) * tlevs, cudaMemcpyDeviceToHost));
}



template <class T>
string cuda_array<T> :: cout_wrapper(const T& val) const
{
	stringstream ss;
	ss << val;
	return ss.str();
}


template <>
string cuda_array<cuda::cmplx_t> :: cout_wrapper(const cuda::cmplx_t& val) const
{
	stringstream ss;
	ss << "(" << cuCreal(val) << ", " << cuCimag(val) << "i)";
	return ss.str();
}


// The array is contiguous, so use only one memcpy
template <class T>
void cuda_array<T> :: copy_device_to_host() 
{
    const size_t line_size = Nx * My * tlevs * sizeof(T);
    cout << "Copying " << line_size << " bytes to " << array_h << " (host) from ";
    cout << array_d << " (device)\n";
    gpuErrchk(cudaMemcpy(array_h, array_d, line_size, cudaMemcpyDeviceToHost));
}


template <class T>
void cuda_array<T> :: copy_device_to_host(unsigned int tlev)
{
    const size_t line_size = Nx * My * sizeof(T);
    gpuErrchk(cudaMemcpy(array_h_t[tlev], array_d_t_host[tlev], line_size, cudaMemcpyDeviceToHost));
}

template class cuda_array<cuda::cmplx_t>;
template class cuda_array<cuda::real_t>;

// End of file cuda_array.cpp
