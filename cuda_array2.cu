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
//#include "include/common_kernel.h"
#include "include/cuda_array2.h"

using namespace std;


__device__ int d_get_col()
{
    return (blockIdx.y * blockDim.y + threadIdx.y);    
}


__device__ int d_get_row()
{
    return (blockIdx.x * blockDim.x + threadIdx.x);
}


// Template kernel for d_enumerate_d and d_enumerate_c using the ca_val class
template <typename T>
__global__ void d_enumerate(T* array, uint Nx, uint My)
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
__global__ void d_enumerate_t(T** array_t, uint t, uint Nx, uint My)
{
    const int col = d_get_col();
    const int row = d_get_row();
    const int index = row * My + col;

	//if (blockIdx.x + threadIdx.x + threadIdx.y == 0)
	//	printf("blockIdx.x = %d: enumerating at t = %d, at %p(device)\n", blockIdx.x, t, array_t[t]);
	if (index < Nx * My)
    {
        ca_val<T> val;
        val.set(double(index));
		array_t[t][index] = val.get();
    }
}


template <typename T>
__global__ void d_set_constant_t(T** array_t, ca_val<T> val, uint t, uint Nx, uint My)
{
    const int col = d_get_col();
    const int row = d_get_row();
	const int index = row * My + col;
	if (index < Nx * My)
		array_t[t][index] = val.get();
}


template <typename T>
__global__ void d_alloc_array_d_t(T** array_d_t, T* array,  uint tlevs,  uint Nx,  uint My)
{
	if (threadIdx.x < tlevs)
	{
		array_d_t[threadIdx.x] = &array[threadIdx.x * Nx * My];
		//printf("Device: array_d_t[%d] at %p\n", threadIdx.x, array_d_t[threadIdx.x]);
	}
}


template <typename T>
__global__ void d_advance(T** array_t, int tlevs)
{
	T* tmp = array_t[tlevs - 1];
	uint t = 0;
    
	//printf("Before:\n");
	//for(t = tlevs - 1; t > 0; t--)
	//	printf("array_t[%d] at %p\n", t, array_t[t]);
	//printf("array_t[0] at %p\n", array_t[0]);
    
	for(t = tlevs - 1; t > 0; t--)
		array_t[t] = array_t[t - 1];
	array_t[0] = tmp;

	//printf("After:\n");
	//for(t = tlevs - 1; t > 0; t--)
	//	printf("array_t[%d] at %p\n", t, array_t[t]);
	//printf("array_t[0] at %p\n", array_t[0]);
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


// Add two arrays: real data, specify time level for RHS
__global__ 
void d_add_arr_t(cuda::real_t** lhs, cuda::real_t** rhs,  uint tlev,  uint Nx,  uint My)
{
    const int col = d_get_col();
    const int row = d_get_row();
    const int idx = row * My + col;
    if ((col < My) && (row < Nx))
        lhs[0][idx] = lhs[tlev][idx] + rhs[tlev][idx];
}


// Add two arrays: complex data, specify time levels for RHS
__global__ 
void d_add_arr_t(cuda::cmplx_t** lhs, cuda::cmplx_t** rhs,  uint tlev,  uint Nx,  uint My)
{
    const int col = d_get_col();
    const int row = d_get_row();
    const int idx = row * My + col;
    if ((col < My) && (row < Nx))
        lhs[0][idx] = cuCadd(lhs[tlev][idx], rhs[tlev][idx]);
}


// Add scalar, real
__global__ 
void d_add_scalar(cuda::real_t** lhs, cuda::real_t rhs,  uint Nx,  uint My)
{
    const int col = d_get_col();
    const int row = d_get_row();
    const int idx = row * My + col;
    if ((col < My) && (row < Nx))
        lhs[0][idx] += rhs;
}

// Add scalar, cmplx
__global__ 
void d_add_scalar(cuda::cmplx_t** lhs, cuda::cmplx_t rhs,  uint Nx,  uint My)
{
    const int col = d_get_col();
    const int row = d_get_row();
    const int idx = row * My + col;
    if ((col < My) && (row < Nx))
        lhs[0][idx] = cuCadd(lhs[0][idx], rhs);
}


// Subtract array: real data, specify time level for RHS
__global__ 
void d_sub_arr_t(cuda::real_t** lhs, cuda::real_t** rhs,  uint tlev,  uint Nx,  uint My)
{
    const int col = d_get_col();
    const int row = d_get_row();
    const int idx = row * My + col;
    if ((col < My) && (row < Nx))
        lhs[0][idx] = lhs[0][idx] - rhs[tlev][idx];
}


// Subtract array: Complex data, specify time level for RHS
__global__ 
void d_sub_arr_t(cuda::cmplx_t** lhs, cuda::cmplx_t** rhs,  uint tlev,  uint Nx,  uint My)
{
    const int col = d_get_col();
    const int row = d_get_row();
    const int idx = row * My + col;
    if ((col < My) && (row < Nx))
        lhs[0][idx] = cuCsub(lhs[0][idx], rhs[tlev][idx]);
}


// Multiply by array: real data, specify time level for RHS
__global__ 
void d_mul_arr_t(cuda::real_t** lhs, cuda::real_t** rhs,  uint tlev,  uint Nx,  uint My)
{
    const int col = d_get_col();
    const int row = d_get_row();
    const int idx = row * My + col;
    if ((col < My) && (row < Nx))
        lhs[0][idx] = lhs[0][idx] * rhs[tlev][idx];
}


// Multiply by array: Complex data, specify time level for RHS
__global__ 
void d_mul_arr_t(cuda::cmplx_t** lhs, cuda::cmplx_t** rhs,  uint tlev,  uint Nx,  uint My)
{
    const int col = d_get_col();
    const int row = d_get_row();
    const int idx = row * My + col;
    if ((col < My) && (row < Nx))
        lhs[0][idx] = cuCmul(lhs[0][idx], rhs[tlev][idx]);
}


// Multiply by scalar: real data 
__global__ 
void d_mul_scalar(cuda::real_t** lhs, cuda::real_t rhs,  uint Nx,  uint My)
{
    const int col = d_get_col();
    const int row = d_get_row();
    const int idx = row * My + col;
    if ((col < My) && (row < Nx))
        lhs[0][idx] = lhs[0][idx] * rhs;
}


// Multiply by scalar: Complex data, specified time level
__global__ 
void d_mul_scalar(cuda::cmplx_t** lhs, cuda::cmplx_t rhs,  uint Nx,  uint My)
{
    const int col = d_get_col();
    const int row = d_get_row();
    const int idx = row * My + col;
    if ((col < My) && (row < Nx))
        lhs[0][idx] = cuCmul(lhs[0][idx], rhs);
}


// Divide by array: real data, specify time level for RHS
__global__
void d_div_arr_t(cuda::real_t** lhs, cuda::real_t** rhs,  uint tlev,  uint Nx,  uint My)
{
    const int col = d_get_col();
    const int row = d_get_row();
    const int idx = row * My + col;
    if ((col < My) && (row < Nx))
        lhs[0][idx] = lhs[0][idx] / rhs[tlev][idx];
}


// Divide by array: complex data, specify time level for RHS
__global__
void d_div_arr_t(cuda::cmplx_t** lhs, cuda::cmplx_t** rhs,  uint tlev,  uint Nx,  uint My)
{
    const int col = d_get_col();
    const int row = d_get_row();
    const int idx = row * My + col;
    if ((col < My) && (row < Nx))
        lhs[0][idx] = cuCdiv(lhs[0][idx], rhs[tlev][idx]);
}


// Kill kx = 0 mode
__global__
void d_kill_kx0(cuda::cmplx_t* arr, uint Nx, uint My)
{
    const int col = d_get_col();
    if (col > My)
        return;
    arr[col].x = 0.; 
    arr[col].y = 0.;
}


// Kill ky=0 mode
__global__
void d_kill_ky0(cuda::cmplx_t* arr, uint Nx, uint My)
{
    const int row = d_get_row();
    if (row > Nx)
        return;
    arr[row * My].x = 0.0;
    arr[row * My].y = 0.0;
}


// Kill k=0 mode
__global__
void d_kill_k0(cuda::cmplx_t* arr) 
{
    arr[0].x = 0.0;
    arr[0].y = 0.0;
}

// Default constructor
template <class T>
cuda_array<T> :: cuda_array(uint t, uint nx, uint my) :
    tlevs(t), Nx(nx), My(my), bounds(tlevs, Nx, My),
    array_d(NULL), array_d_t(NULL),
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
    gpuErrchk(cudaMalloc( (void**) &array_d, nelem * sizeof(T)));
    array_h = (T*) calloc(nelem, sizeof(T));

    // array_[hd]_t is an array of pointers allocated on the host/device respectively
    gpuErrchk(cudaMalloc( (void***) &array_d_t, tlevs * sizeof(T*)));
    array_h_t = (T**) calloc(tlevs, sizeof(T*));
    array_d_t_host = (T**) calloc(tlevs, sizeof(T*));
    // array_t[i] points to the i-th time level
    // Set pointers on device
    d_alloc_array_d_t<<<1, tlevs>>>(array_d_t, array_d, tlevs, Nx, My);
    // Update host copy
    gpuErrchk(cudaMemcpy(array_d_t_host, array_d_t, sizeof(T*) * tlevs, cudaMemcpyDeviceToHost));

    for(uint tl = 0; tl < tlevs; tl++)
        array_h_t[tl] = &array_h[tl * Nx * My];
    set_all(0.0);
    cudaDeviceSynchronize();
#ifdef DEBUG
    cout << "Array size: Nx=" << Nx << ", My=" << My << ", tlevs=" << tlevs << "\n";
    cout << "cuda::cuda_blockdim_x = " << cuda::cuda_blockdim_nx;
    cout << ", cuda::cuda_blockdim_y = " << cuda::cuda_blockdim_my << "\n";
    cout << "blockDim=(" << block.x << ", " << block.y << ")\n";
    cout << "gridDim=(" << grid.x << ", " << grid.y << ")\n";
    cout << "Device data at " << array_d << "\t";
    cout << "Host data at " << array_h << "\t";
    cout << nelem << " bytes of data\n";
    for (uint tl = 0; tl < tlevs; tl++)
    {
        cout << "time level " << tl << " at ";
        cout << array_h_t[tl] << "(host)\t";
        cout << array_d_t_host[tl] << "(device)\n";
    }
    cout << "Testing allocation of array_d_t:\n";
    test_alloc<<<1, 1>>>(array_d_t, tlevs);
#endif // DEBUG
}


// copy constructor
template <class T>
cuda_array<T> :: cuda_array(const cuda_array<T>& rhs) :
    tlevs(rhs.tlevs), Nx(rhs.Nx), My(rhs.My), bounds(tlevs, Nx, My),
    block(rhs.block), grid(rhs.grid), grid_full(rhs.grid_full),
    array_d(NULL), array_d_t(NULL),
    array_h(NULL), array_h_t(NULL)
{
    cout << "cuda_array::cuda_array(const cuda_array& rhs)\n";
    cout << "Not implemented yet\n";
    // Memcpy's should do it, right? :)
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
void cuda_array<T> :: enumerate_array(const  uint t)
{
	if (!bounds(t, Nx-1, My-1))
		throw out_of_bounds_err(string("T& cuda_array<T> :: enumerate_array(const int): out of bounds\n"));
	d_enumerate<<<grid, block>>>(array_d, Nx, My);
	cudaDeviceSynchronize();
}


template <typename T>
void cuda_array<T> :: enumerate_array_t(const  uint t)
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


template <typename T>
cuda_array<T>& cuda_array<T> :: operator= (const T& rhs)
{
    ca_val<T> val;
    val.set(rhs);
    d_set_constant_t<<<grid, block>>>(array_d_t, val, 0, Nx, My);
    cudaDeviceSynchronize();
    return *this;
}


// Set entire array to rhs
template <class T>
cuda_array<T>& cuda_array<T> :: set_all(const double& rhs)
{
    ca_val<T> val;
    val.set(rhs);
	for(uint t = 0; t < tlevs; t++)
	{
		d_set_constant_t<<<grid, block>>>(array_d_t, val, t, Nx, My);
		cudaDeviceSynchronize();
	}
	return *this;
}


// Set array to rhs for time level tlev
template <class T>
cuda_array<T>& cuda_array<T> :: set_t(const T& rhs, uint t)
{
    ca_val<T> val;
    val.set(rhs);
    d_set_constant_t<<<grid, block>>>(array_d_t, val, t, Nx, My);
    cudaDeviceSynchronize();
    return *this;
}


template <typename T>
cuda_array<T>& cuda_array<T> :: operator+=(const cuda_array<T>& rhs)
{
    if(!bounds(rhs.get_nx(), rhs.get_my()))
        throw out_of_bounds_err(string("cuda_array<T>& cuda_array<T> :: operator+= (const cuda_array<T>& rhs): out of bounds!"));
    if ((void*) this == (void*) &rhs)
        throw operator_err(string("cuda_array<T>& cuda_array<T> :: operator+= (const cuda_array<T>&): RHS and LHS cannot be the same\n"));

    d_add_arr_t<<<grid, block>>>(array_d_t, rhs.get_array_d_t(), 0, Nx, My);
    cudaDeviceSynchronize();
    return *this;
}


// Template specialization for scalar operations since CUDA insists on not
// overloading operators for arithmetic for cuDoubleComplex... :/
template <>
cuda_array<cuda::real_t>& cuda_array<cuda::real_t> :: operator+=(const cuda::real_t& rhs)
{
    d_add_scalar<<<grid, block>>>(array_d_t, rhs, Nx, My);
    cudaDeviceSynchronize();
    return *this;
}


template <>
cuda_array<cuda::cmplx_t>& cuda_array<cuda::cmplx_t> :: operator+=(const cuda::cmplx_t& rhs)
{
    d_add_scalar<<<grid, block>>>(array_d_t, rhs, Nx, My);
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
    
    d_sub_arr_t<<<grid, block>>>(array_d_t, rhs.get_array_d_t(), 0, Nx, My);
    cudaDeviceSynchronize();
    return *this;
}


// Template specialization
template <>
cuda_array<cuda::real_t>& cuda_array<cuda::real_t> :: operator-=(const cuda::real_t& rhs)
{
    d_add_scalar<<<grid, block>>>(array_d_t, -1.0 * rhs, Nx, My);
    cudaDeviceSynchronize();
    return *this;
}


template<>
cuda_array<cuda::cmplx_t>& cuda_array<cuda::cmplx_t> :: operator-=(const cuda::cmplx_t& rhs)
{
    cuda::cmplx_t minus_rhs = cuCmul(make_cuDoubleComplex(-1.0, 0.0), rhs);
    d_add_scalar<<<grid, block>>>(array_d_t, minus_rhs, Nx, My);
    return *this;
}


template <typename T>
cuda_array<T>& cuda_array<T> :: operator*=(const cuda_array<T>& rhs)
{
    if(!bounds(rhs.get_nx(), rhs.get_my()))
        throw out_of_bounds_err(string("cuda_array<T>& cuda_array<T> :: operator*= (const cuda_array<T>& rhs): out of bounds!"));
    if ((void*) this == (void*) &rhs)
        throw operator_err(string("cuda_array<T>& cuda_array<T> :: operator*= (const cuda_array<T>&): RHS and LHS cannot be the same\n"));
    
    d_mul_arr_t<<<grid, block>>>(array_d_t, rhs.get_array_d_t(), 0, Nx, My);
    cudaDeviceSynchronize();
    return *this;
}
      

template<>
cuda_array<cuda::real_t>& cuda_array<cuda::real_t> :: operator*=(const cuda::real_t& rhs)
{
    d_mul_scalar<<<grid, block>>>(array_d_t, rhs, Nx, My);
    cudaDeviceSynchronize();
    return *this;
}


template<>
cuda_array<cuda::cmplx_t>& cuda_array<cuda::cmplx_t> :: operator*=(const cuda::cmplx_t& rhs)
{
    d_mul_scalar<<<grid, block>>>(array_d_t, rhs, Nx, My);
    cudaDeviceSynchronize();
    return *this;
}



template <class T>
T& cuda_array<T> :: operator()(uint t, uint n, uint m)
{
	if (!bounds(t, n, m))
		throw out_of_bounds_err(string("T& cuda_array<T> :: operator()(uint, uint, uint): out of bounds\n"));
	return (*(array_h_t[t] + address(n, m)));
}


template <class T>
T cuda_array<T> :: operator()(uint t, uint n, uint m) const
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
    // Update array_d_t_host
    gpuErrchk(cudaMemcpy(array_d_t_host, array_d_t, sizeof(T*) * tlevs, cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();

    // Zero out last time level 
    ca_val<T> val;
    val.set(double(0.0));
    //d_set_constant_t<<<grid, block>>>(array_d_t, val, tlevs - 1, Nx, My);
    d_set_constant_t<<<grid, block>>>(array_d_t, val, 0, Nx, My);
}



template <typename T>
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


// The array is contiguous 
template <typename T>
void cuda_array<T> :: copy_device_to_host() 
{
    //const size_t line_size = Nx * My * tlevs * sizeof(T);
    //gpuErrchk(cudaMemcpy(array_h, array_d, line_size, cudaMemcpyDeviceToHost));
    const size_t line_size = Nx * My * sizeof(T);
    for(uint t = 0; t < tlevs; t++)
    {
        gpuErrchk(cudaMemcpy(&array_h[t * (Nx * My)], array_d_t_host[t], line_size, cudaMemcpyDeviceToHost));
    }
}


template <typename T>
void cuda_array<T> :: copy_device_to_host(uint tlev)
{
    const size_t line_size = Nx * My * sizeof(T);
    gpuErrchk(cudaMemcpy(array_h_t[tlev], array_d_t_host[tlev], line_size, cudaMemcpyDeviceToHost));
}


template <typename T>
void cuda_array<T> :: copy(uint t_dst, uint t_src)
{
    const size_t line_size = Nx * My * sizeof(T);
    gpuErrchk(cudaMemcpy(array_d_t_host[t_dst], array_d_t_host[t_src], line_size, cudaMemcpyDeviceToDevice));
}


template <typename T>
void cuda_array<T> :: copy(uint t_dst, const cuda_array<T>& src, uint t_src)
{
    const size_t line_size = Nx * My * sizeof(T);
    gpuErrchk(cudaMemcpy(array_d_t_host[t_dst], src.get_array_d(t_src), line_size, cudaMemcpyDeviceToDevice));
}


template <typename T>
void cuda_array<T> :: move(uint t_dst, uint t_src)
{
    // Copy data 
    const size_t line_size = Nx * My * sizeof(T);
    gpuErrchk(cudaMemcpy(array_d_t_host[t_dst], array_d_t_host[t_src], line_size, cudaMemcpyDeviceToDevice));
    // Clear source
    ca_val<T> zero;
    zero.set(0.0);
    d_set_constant_t<<<grid, block>>>(array_d_t, zero, t_src, Nx, My);
}


template <typename T>
void cuda_array<T> :: swap(uint t1, uint t2)
{
    //
    d_swap<<<1, 1>>>(array_d_t, t1, t2);
    gpuErrchk(cudaMemcpy(array_d_t_host, array_d_t, sizeof(T*) * tlevs, cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();
}


template <>
void cuda_array<cuda::cmplx_t> :: kill_kx0()
{
    d_kill_kx0<<<grid.x, block.x>>>(array_d, Nx, My);
}

template <>
void cuda_array<cuda::cmplx_t> :: kill_ky0()
{
    d_kill_ky0<<<grid.y, block.y>>>(array_d, Nx, My);
}

template <>
void cuda_array<cuda::cmplx_t> :: kill_k0()
{
    d_kill_k0<<<1, 1>>>(array_d);
}

// Normalize works only for real arrays
template <>
void cuda_array<cuda::real_t> :: normalize()
{
    cuda::real_t norm = 1. / cuda::real_t(Nx * My);
    d_mul_scalar<<<grid, block>>>(array_d_t, norm, Nx, My);
}


template <>
void cuda_array<cuda::cmplx_t> :: normalize()
{
    // Do nothing
}

template class cuda_array<cuda::cmplx_t>;
template class cuda_array<cuda::real_t>;

// End of file cuda_array.cpp
