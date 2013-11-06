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



// Template kernels
__global__ void d_enumerate_d(double* array, int Nx, int My)
{
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int index = row * (My) + col;

    if (index < Nx * My)
        array[index] = double(index);
}


__global__ void d_enumerate_c(cmplx_t* array, int Nx, int My)
{
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int index = row * (My) + col;

    if (index < Nx * My)
        array[index] = make_cuDoubleComplex(double(index), -double(index));
}



__global__ void d_enumerate_t_d(double** array_t, int t, int Nx, int My)
{
	const int col = blockIdx.y * blockDim.y + threadIdx.y;
	const int row = blockIdx.x * blockDim.x + threadIdx.x;
	const int index = row * My + col;

	if (blockIdx.x + threadIdx.x + threadIdx.y == 0)
		printf("blockIdx.x = %d: enumerating at t = %d, at %p(device)\n", blockIdx.x, t, array_t[t]);
	if (index < Nx * My)
		array_t[t][index] = double(index);
}


__global__ void d_enumerate_t_c(cmplx_t** array_t, int t, int Nx, int My)
{
    const int col = blockIdx.y + blockDim.y + threadIdx.y;
    const int row = blockIdx.x + blockDim.x + threadIdx.x;
    const int index = row * My + col;

	if (blockIdx.x + threadIdx.x + threadIdx.y == 0)
		printf("blockIdx.x = %d: enumerating at t = %d, at %p(device)\n", blockIdx.x, t, array_t[t]);
	if (index < Nx * My)
		array_t[t][index] = make_cuDoubleComplex(double(index), -double(index));
}


template <typename T>
__global__ void d_set_constant_t(T** array_t, T val, int t, int Nx, int My)
{
	const int col = blockIdx.y * blockDim.y + threadIdx.y;
	const int row = blockIdx.x * blockDim.x + threadIdx.x;
	const int index = row * My + col;
	if( index < Nx * My)
		array_t[t][index] = val;
}


template <typename T>
__global__ void d_alloc_array_d_t(T** array_d_t, T* array, int tlevs, int Nx, int My)
//__global__ void d_alloc_array_d_t(double** array_d_t, double* array, int Nx, int My, int tlevs)
{
	if (threadIdx.x < tlevs)
	{
		array_d_t[threadIdx.x] = &array[threadIdx.x * Nx * My];
		printf("array_d_t[%d] at %p\n", threadIdx.x, array_d_t[threadIdx.x]);
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
	{
		array_t[t] = array_t[t - 1];
	}
	array_t[0] = tmp;

	printf("After:\n");
	for(t = tlevs - 1; t > 0; t--)
		printf("array_t[%d] at %p\n", t, array_t[t]);
	printf("array_t[0] at %p\n", array_t[0]);
}



__global__ void d_test_alloc(double** array_t, int tlevs)
{
	for(int t = 0; t < tlevs; t++)
		printf("array_d_t[%d] at %p\n", t, array_t[t]);

}


template <typename T>
__global__ void test_alloc(T** array_t, int tlevs)
{
    for(int t = 0; t < tlevs; t++)
        printf("array_x_t[%d] at %p\n", t, array_t[t]);
}



// Default constructor
template <class T>
cuda_array<T> :: cuda_array(unsigned int t, unsigned int nx, unsigned int my) :
    tlevs(t), Nx(nx), My(my), bounds(tlevs, Nx, My),
    array_d(NULL), array_d_t(NULL),
    array_h(NULL), array_h_t(NULL)
{
    //cudaError_t err;
    // Determine grid size for kernel launch
    block = dim3(cuda_blockdim_nx, cuda_blockdim_my);
    // Testing, use 64 threads in y direction. Thus blockDim.y = 1
    // blockDim.x is the x, one block for each row
    grid = dim3(Nx, 1);
    grid_full = dim3(tlevs * Nx, 1);
    cout << "blockDim=(" << block.x << ", " << block.y << ", " << block.z << ")\n";
    cout << "gridDim=(" << grid.x << ", " << grid.y << ", " << grid.z << ")\n";

    cout << "Array size: " << Nx << "," << My << "," << tlevs << "\n";
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

    // array_t[i] points to the i-th time level
    d_alloc_array_d_t<<<1, tlevs>>>(array_d_t, array_d, tlevs, Nx, My);

    for(unsigned int tl = 0; tl < tlevs; tl++)
    {
        array_h_t[tl] = &array_h[tl * Nx * My];
        cout << "time level " << tl << " at ";
        cout << array_h_t[tl] << "(host)\n";
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
}


// Access functions for private members

template <>
void cuda_array<real_t> :: enumerate_array(const int t)
{
    // Call enumeration kernel
	if (!bounds(t, Nx-1, My-1))
		throw out_of_bounds_err(string("T& cuda_array<T> :: enumerate_array(const int): out of bounds\n"));
	d_enumerate_d<<<grid, block>>>(array_d, Nx, My);
	cudaDeviceSynchronize();
}

template <> 
void cuda_array<cmplx_t> :: enumerate_array(const int t)
{
	if (!bounds(t, Nx-1, My-1))
		throw out_of_bounds_err(string("T& cuda_array<T> :: enumerate_array(const int): out of bounds\n"));
	d_enumerate_c<<<grid, block>>>(array_d, Nx, My);
	cudaDeviceSynchronize();
}


template <>
void cuda_array<real_t> :: enumerate_array_t(const int t)
{
	if (!bounds(t, Nx-1, My-1))
		throw out_of_bounds_err(string("T& cuda_array<T> :: enumerate_array_t(const int): out of bounds\n"));
	d_enumerate_t_d<<<grid, block >>>(array_d_t, t, Nx, My);
	cudaDeviceSynchronize();
}

template <>
void cuda_array<cmplx_t> :: enumerate_array_t(const int t)
{
	if (!bounds(t, Nx-1, My-1))
		throw out_of_bounds_err(string("T& cuda_array<T> :: enumerate_array_t(const int): out of bounds\n"));
	d_enumerate_t_c<<<grid, block >>>(array_d_t, t, Nx, My);
	cudaDeviceSynchronize();
}


// Operators
template <class T>
cuda_array<T>& cuda_array<T> :: operator= (const cuda_array<T>& rhs)
{
    // Copy array data from one array to another array
    // Ensure dimensions are equal
	cout << "If i had a kernel I would copy the argument cuda_array\n";
    return *this;
}

// Set whole array to rhs
template <class T>
cuda_array<T>& cuda_array<T> :: operator= (const T& rhs)
{
    //d_set_constant_t<<<grid_full, block>>>(array_d, rhs, Nx, My, tlevs);
	d_set_constant_t<<<grid, block>>>(array_d_t, rhs, 0, Nx, My);
	cout << "If i have a kernel and set the array to " << cout_wrapper(rhs) << "\n";
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
	d_set_constant_t<<<grid, block>>>(array_d_t, double(0.0), tlevs - 1, Nx, My);
}



template <class T>
string cuda_array<T> :: cout_wrapper(const T& val) const
{
	stringstream ss;
	ss << val;
	return ss.str();
}


template <>
string cuda_array<cmplx_t> :: cout_wrapper(const cmplx_t& val) const
{
	stringstream ss;
	ss << "(" << cuCreal(val) << ", " << cuCimag(val) << "i)";
	return ss.str();
}


// The array is contiguous, so use only one memcpy
template <class T>
void cuda_array<T> :: copy_device_to_host() {
    const size_t size_line = Nx * My * tlevs * sizeof(T);
    cout << "Copying " << size_line << " bytes to " << array_h << " (host) from ";
    cout << array_d << " (device)\n";
    gpuErrchk(cudaMemcpy(array_h, array_d, size_line, cudaMemcpyDeviceToHost));
}


template class cuda_array<cmplx_t>;
template class cuda_array<double>;

// End of file cuda_array.cpp
