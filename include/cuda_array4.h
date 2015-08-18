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

#ifndef CUDA_ARRAY4_H
#define CUDA_ARRAY4_H


#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cufft.h>
//#include <string>
#include <sstream>
#include "bounds.h"
#include "error.h"
#include "cuda_types.h"
#include "cuda_operators.h"



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


//void mycudaMalloc(void** devPtr, size_t size)
//{
//    std::ofstream of;
//    static int counter{0};
//
//    gpuErrchk(cudaMalloc(devPtr, size));
//
//    of.open("cuda_malloc.log", ios::app);
//    of << counter << ": " << *devPtr << endl;
//    of.close();
//
//    //if (counter == 12)
//    //    throw;
//
//    counter++;
//
//    return;
//}


/// Device function to compute column and row
__device__ inline int d_get_col() {
    return (blockIdx.x * blockDim.x + threadIdx.x);
}


__device__ inline int d_get_row() {
    return (blockIdx.y * blockDim.y + threadIdx.y);
}


// Enumerate array at a given time level
template <typename T>
__global__ void d_enumerate(T* array, const uint My, const uint Nx)
{
    const int col = d_get_col();
    const int row = d_get_row();
    const int index = row * Nx + col;

    if (index < Nx * My)
        array[index] = T(index);
}


// Template version of d_enumerate_t_x
template <typename T>
__global__ void d_enumerate_t(T** array_t, const uint t, const uint My, const uint Nx)
{
    const int col = d_get_col();
    const int row = d_get_row();
    const int index = row * Nx + col;

	if (index < Nx * My)
        array_t[t][index] = 10000 * T(t) + T(index);
}


// T** array_d_t : array of pointers to device data. array_d_t[tlev] points to data at time level tlve
// T* array      : array data, in consecutive memory.
template <typename T>
__global__ void d_alloc_array_d_t(T** array_d_t, T* array, const uint tlevs, const uint My, const uint Nx)
{
#ifdef DEBUG
    printf("T **array_d_t at %p\n", array_d_t);
#endif
    for(uint t = 0; t < tlevs; t++)
    {
        array_d_t[t] = &array[t * Nx * My];
#ifdef DEBUG
        printf("array_d_t[%d] (at %p) -> %p\n", t, &array_d_t[t], &array[t * Nx * My]);
#endif
    }
}


// Advances pointer in array_d_t
template <typename T>
__global__ void d_advance(T** array_d_t, const uint tlevs)
{
	int t = 0;
    //printf("\tadvance\n\tbefore: \n");
	//for(t = tlevs - 1; t >= 0; t--)
    //    printf("\t\tarray_d_t[%d] = %p\n", t, array_d_t[t]);

	T* tmp = array_d_t[tlevs - 1];
	for(t = tlevs - 1; t > 0; t--)
		array_d_t[t] = array_d_t[t - 1];
	array_d_t[0] = tmp;

    //printf("\tafter:\n");
	//for(t = tlevs - 1; t >= 0; t--)
    //    printf("\t\tarray_d_t[%d] = %p\n", t, array_d_t[t]);
}


// Swap time level t1 and t2
template <typename T>
__global__ void d_swap(T**array_d_t, const uint t1, const uint t2)
{
    T* tmp = array_d_t[t1];
    array_d_t[t1] = array_d_t[t2];
    array_d_t[t2] = tmp;
}


// Print address of time level in device memory
template <typename T>
__global__ void test_alloc(T** array_d_t,  const uint tlevs)
{
    for(int t = 0; t < tlevs; t++)
        printf("array_d_t[%d] at %p\n", t, array_d_t[t]);
}


/// Arithmetic operation,
/// op(lhs[idx], rhs[idx])
/// where op is of type op1, see cuda_operators.h
template <typename T, typename O>
__global__
void d_op1_arr(T* lhs, T* rhs, uint My, uint Nx)
{
    const uint col = d_get_col();
    const uint row = d_get_row();
    const uint idx = row * Nx + col;
    O op;

    if ((col < Nx) && (row < My))
    	op(lhs[idx], rhs[idx]);
}

// Arithmetic operation with scalar RHS
// performs op(lhs[idx], rhs) on array
// where op is of type op1, see coda_operators.h
template <typename T, typename O>
__global__
void d_op1_scalar(T* lhs, const T rhs, const uint My, const uint Nx)
{
    const uint col = d_get_col();
    const uint row = d_get_row();
    const uint idx = row * Nx + col;
    O op;

    if ((col < Nx) && (row < My))
    	op(lhs[idx], rhs);
}


// Apply an expression on each element
// lhs[idx] = op(lhs[idx])
template <typename T, typename O>
__global__
void d_op0_apply(T* lhs, const uint My, const uint Nx)
{
	const uint col = d_get_col();
	const uint row = d_get_row();
	const uint idx = row * Nx + col;
	O op;

	if((col < Nx) && (row < My))
		op(lhs[idx]);
}

// Apply function on each element
template <typename T, typename O>
__global__
void d_op0_apply_fun(T* lhs, cuda::slab_layout_t sl, const uint My, const uint Nx)
{
    const uint col = d_get_col();
    const uint row = d_get_row();
    const uint idx = row * Nx + col;

    const double x = sl.x_left + (T) col * sl.delta_x;
    const double y = sl.y_lo + (T) row * sl.delta_y;
    O fun;

    if((col < Nx) && (row < My))
        lhs[idx] = fun(x, y);
}

// Kill kx = 0 mode
template <typename T>
__global__
void d_kill_kx0(T* arr, const uint My, const uint Nx)
{
    //const int row = 0;
    const int col = d_get_col();
    //const int idx = row * Nx + col;
    if(col < Nx)
        arr[col] = T(0.0);
}


// Kill ky=0 mode
template <typename T>
__global__
void d_kill_ky0(T* arr, const uint Nx, const uint My)
{
    const int row = d_get_row();
    //const int col = 0;
    //const int idx = row * My + col;
    if(row < My)
        arr[row * Nx] = T(0.0);
}


// Kill k=0 mode
// This should be only invoked where U = cuda::cmplx_t*
template <typename T>
__global__
void d_kill_k0(T* arr)
{
    arr[0] = T(0.0);
}





#endif // __CUDACC_

/// T is the base type of T, for example T=CuCmplx<double>, T = double, T = float...
template <typename T>
class cuda_array{
    public:
        // Explicitly declare construction operators that allocate memory
        // on the device
        cuda_array(uint, uint, uint); // time levels, my, nx
        cuda_array(uint my_, uint nx_) : cuda_array(1, my_, nx_) {};  // my, nx
        cuda_array(const cuda_array<T>&); // copy constructor
        cuda_array(cuda_array<T>&&); // move constructor
        ~cuda_array();

        // Test function
        void enumerate_array(const uint);
        void enumerate_array_t(const uint);


        /// @brief Apply an expression element-wise to array data and assign element to result
        /// @detailed O(array_d[tlev][idx]) , f.ex. array_d[tlev][idx] = exp(array_d[tlev][idx])
        template <typename O>
        inline void op_apply_t(const uint);

        /// @brief Perform element-wise arithmetic operation on array data with a scalar RHS
        /// @detailed O is of type d_op_[???]: d_op_assign, d_op_{add,sub,mul,div}assign
        /// @detailed see cuda_operators.h
        /// @detailed performs O(array_d[tlev][idx], T), idx=0...My * Nx on array data
        template<typename O>
        inline void op_scalar_t(const T&, const uint);

        /// @brief Perform element-wise arithmetic operation on array data with a array RHS
        /// @detailed O is of type d_op_[???]: d_op_assign, d_op_{add,sub,mul,div}assign
        /// @detailed see cuda_operators.h
        /// @detailed performs O(array_d[tlev][idx], rhs[tlev][idx]), idx=0...My * /Nx on array data
        template <typename O>
        inline void op_array_t(const cuda_array<T>&, const uint);


        /// @brief Perform element-wise arithmetic operation on array data with array RHS
        /// @detailed See op_array_t(const cuda_array<T>&, const uint). This method allows specification
        /// @detailed of used time levels:
        /// @detailed perform O(array_d[t_dst][idx], rhs[t_src][idx]), idx = 0 ... My * Nx
        /// @param const uint t_dst: Time level of this
        /// @param const cuda_array<T>& rhs: right hand side in operator expression
        /// @param
        //template <typename O>
        //void op_array_t(const uint t_dst, const cuda_array<T>& rhs, const uint t_src);


        /// @brief Evaluate the functor O on the array
        template <typename O>
        inline void op_scalar_fun(const cuda::slab_layout_t, const uint);

        // Operators
        // operator= returns a cuda_array, so that it can be used in chaining assignments:
        // cuda_darray<T> a1, a2, a3, ...
        // a1 = a2 = a3 = ...
        // copies data on all time levels from rhs to new cuda_array
        cuda_array<T>& operator=(const cuda_array<T>&);
        ///@ moves data on all time levels from rhs to new cuda_array
        cuda_array<T>& operator=(cuda_array<T>&&);
        /// @brief copies data from all time levels from this to new cuda_array, set data on tlev=0 to rhs on new cuda_array
        cuda_array<T>& operator=(const T&);

        // call d_op_array_t<d_op1_addassign<T>>
        cuda_array<T>& operator+=(const cuda_array<T>&);
        // call d_op_scalar_t<d_op1_assign<T>>
        cuda_array<T>& operator+=(const T&);
        cuda_array<T> operator+(const cuda_array<T>&) const;
        cuda_array<T> operator+(const T&) const;

        // call d_op_array_t<d_op1_subassign<T>>
        cuda_array<T>& operator-=(const cuda_array<T>&);
        // call d_op_scalar_t<d_op1_subassign<T>>
        cuda_array<T>& operator-=(const T&);
        cuda_array<T> operator-(const cuda_array<T>&) const;
        cuda_array<T> operator-(const T&) const;

        // call d_op_array_t<d_op1_mulassign<T>>
        cuda_array<T>& operator*=(const cuda_array<T>&);
        // call d_op_scalar_t<d_op1_mulassign<T>>
        cuda_array<T>& operator*=(const T&);
        cuda_array<T> operator*(const cuda_array<T>&) const;
        cuda_array<T> operator*(const T&) const;

        // call d_op_array_t<d_op1_divassign<T>>
        cuda_array<T>& operator/=(const cuda_array<T>&);
        // call d_op_scalar_t<d_op1_divassign<T>>
        cuda_array<T>& operator/=(const T&);
        cuda_array<T> operator/(const cuda_array<T>&) const;
        cuda_array<T> operator/(const T&) const;

        // Access operator to host array
        T& operator()(uint, uint, uint);
        T operator()(uint, uint, uint) const;

        // Copy device memory to host and print to stdout
        friend std::ostream& operator<<(std::ostream& os, cuda_array<T>& src)
        {
            const uint tl = src.get_tlevs();
            const uint my = src.get_my();
            const uint nx = src.get_nx();
            src.copy_device_to_host();
            os << "\n";
            for(uint t = 0; t < tl; t++)
            {
                for(uint m = 0; m < my; m++)
                {
                    for(uint n = 0; n < nx; n++)
                    {
                        // Remember to also set precision routines in CuCmplx :: operator<<
                        os << std::setw(cuda::io_w) << std::setprecision(cuda::io_p) << src(t, m, n) << "\t";
                    }
                os << endl;
                }
                os << endl;
            }
            return (os);
        }

        // Copy entire device data to host
        void copy_device_to_host();
        // Copy entire device data to external data pointer in host memory
        void copy_device_to_host(T*);
        // Copy device to host at specified time level
        void copy_device_to_host(uint);

        // Copy deice data at specified time level to external pointer in device memory
        void copy_device_to_device(const uint, T*);

        // Transfer from host to device
        void copy_host_to_device();
        void copy_host_to_device(uint);
        void copy_host_to_device(T*);

        // Advance time levels
        void advance();

        ///@brief Copy data from t_src to t_dst
        void copy(uint t_dst, uint t_src);
        ///@brief Copy data from src, t_src to t_dst
        void copy(uint t_dst, const cuda_array<T>& src, uint t_src);
        ///@brief Move data from t_src to t_dst, zero out t_src
        void move(uint t_dst, uint t_src);
        ///@brief swap data in t1, t2
        void swap(uint t1, uint t2);
        void normalize();

        void kill_kx0();
        void kill_ky0();
        void kill_k0();

        // Access to private members
        inline uint get_nx() const {return(Nx);};
        inline uint get_my() const {return(My);};
        inline uint get_tlevs() const {return(tlevs);};
        inline int address(uint m, uint n) const {return(m * Nx + n);};
        inline dim3 get_grid() const {return grid;};
        inline dim3 get_block() const {return block;};

        //bounds get_bounds() const {return check_bounds;};
        // Pointer to host copy of device data
        inline T* get_array_h() const {return array_h;};
        inline T* get_array_h(uint t) const {return array_h_t[t];};

        // Pointer to device data, entire array
        inline T* get_array_d() const {return array_d;};
        // Pointer to array of pointers, corresponding to time levels
        inline T** get_array_d_t() const {return array_d_t;};
        // Pointer to device data at time level t
        inline T* get_array_d(uint t) const {return array_d_t_host[t];};

        // Check bounds
        inline void check_bounds(uint t, uint m, uint n) const {array_bounds(t, m, n);};
        inline void check_bounds(uint m, uint n) const {array_bounds(m, n);};

    private:
        // Size of data array. Host data
        const uint tlevs;
        const uint My;
        const uint Nx;

        bounds array_bounds;

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
// Template parameters: T is the type parameter
// T is usually doublem or CuCmplx<double>. Test for float not done
template <typename T>
cuda_array<T> :: cuda_array(uint t, uint my, uint nx) :
	tlevs(t), 
    My(my),
    Nx(nx), 
    array_bounds(tlevs, My, Nx),
	block(dim3(cuda::blockdim_nx, cuda::blockdim_my)),
	// Round integer division for grid.y, see: http://stackoverflow.com/questions/2422712/c-rounding-integer-division-instead-of-truncating
	// a la int a = (59 + (4 - 1)) / 4;
	grid(dim3((Nx + (cuda::blockdim_nx - 1)) / cuda::blockdim_nx, My)),
	array_d(nullptr), 
    array_d_t(nullptr), 
    array_d_t_host(new T*[tlevs]),
	array_h(nullptr), 
    array_h_t(new T*[tlevs])
{
    // Allocate device memory
    size_t nelem = tlevs * My * Nx;
//#ifdef DEBUG
//    mycudaMalloc( (void**) &array_d, nelem * sizeof(T));
//#endif // DEBUG

    gpuErrchk(cudaMalloc( (void***) &array_d_t, tlevs * sizeof(T*)));

//#ifndef DEBUG
    gpuErrchk(cudaMalloc( (void**) &array_d, nelem * sizeof(T)));
//#endif

#ifdef PINNED_HOST_MEMORY
    // Allocate pinned host memory for faster memory transfers
    gpuErrchk(cudaMallocHost( (void**) &array_h, nelem * sizeof(T)));
#endif
#ifndef PINNED_HOST_MEMORY
    array_h = new T[nelem];
#endif
	// Initialize array to zero
    for(uint n = 0; n < Nx * My; n++)
        array_h[n] = T(0.0);

    // array_[hd]_t is an array of pointers allocated on the host/device respectively
    // array_t[i] points to the i-th time level
    // Set pointers on device
    d_alloc_array_d_t<<<1, 1>>>(array_d_t, array_d, tlevs, Nx, My);
    // Update host copy
    gpuErrchk(cudaMemcpy(array_d_t_host, array_d_t, sizeof(T*) * tlevs, cudaMemcpyDeviceToHost));

    for(uint t = 0; t < tlevs; t++)
    {
        array_h_t[t] = &array_h[t * Nx * My];
        op_scalar_t<d_op1_assign<T> >(T(0.0), t);
    }

#ifdef DEBUG
    cout << "!!!!! FULL CONSTRUCTOR! YOU JUST HAVE ALLOCATED MEMORY !!!!!" << endl;
//    cout << "cuda_array<T> :: cuda_array<T>(uint, uint, uint)" << endl;
//    cout << "Array size: Nx=" << Nx << ", My=" << My << ", tlevs=" << tlevs << endl;
//    cout << "cuda::blockdim_x = " << cuda::blockdim_nx;
//    cout << ", cuda::blockdim_y = " << cuda::blockdim_my <<endl;
//    cout << "blockDim=(" << block.x << ", " << block.y << ")" << endl;
//    cout << "gridDim=(" << grid.x << ", " << grid.y << ")" << endl;
    cout << "Device data: array_d at " << array_d << "\t";
    cout << "Host data: array_h at " << array_h << "\t";
    cout << nelem << " bytes of data\n";
    cout << "array_d_t_host at " << array_d_t_host << endl;
    cout << "array_h_t at " << array_h_t << endl;
    for (uint tl = 0; tl < tlevs; tl++)
    {
        cout << "time level " << tl << " at ";
        cout << "array_h_t[" << tl << "] at " << array_h_t[tl] << "\t";
        cout << "array_d_t_host[" << tl << "] at " << array_d_t_host[tl] << "\n";
    }
//    cout << "Testing allocation of array_d_t:\n";
//    test_alloc<<<1, 1>>>(array_d_t, tlevs);
#endif // DEBUG
}


template <typename T>
cuda_array<T> :: cuda_array(const cuda_array<T>& rhs) :
	cuda_array(rhs.get_tlevs(), rhs.get_my(), rhs.get_nx())
{
	const size_t line_size = get_tlevs() * get_my() * get_nx() * sizeof(T);
	gpuErrchk(cudaMemcpy(get_array_d(), rhs.get_array_d(), line_size, cudaMemcpyDeviceToDevice));
}



template <typename T>
cuda_array<T> :: cuda_array(cuda_array<T>&& rhs) :
	tlevs(rhs.get_tlevs()), My(rhs.get_my()), Nx(rhs.get_nx()),
	array_bounds(tlevs, My, Nx),
    block(rhs.block),
	//grid(dim3((Nx + (cuda::blockdim_nx - 1)) / cuda::blockdim_nx, My)),
    grid(rhs.grid),
	array_d(nullptr),
	array_d_t(nullptr),
	array_d_t_host(nullptr),
	array_h(nullptr),
	array_h_t(nullptr)
{
//#ifdef DEBUG
//	cout << "cuda_array<T> :: cuda_array(cuda_array<T>&& rhs)";
//	cout << "initially:" << endl;
//	cout << "array_d:        this" << array_d       << "\trhs:" << rhs.array_d << endl;
//	cout << "array_d_t:      this" << array_d_t     << "\trhs:" << rhs.array_d_t << endl;
//	cout << "array_d_t_host: this" << array_d_t_host << "\trhs:" << rhs.array_d_t_host << endl;
//	cout << "array_h:        this" << array_h       << "\trhs:" << rhs.array_h << endl;
//	cout << "array_h_t:      this" << array_h_t     << "\trhs:" << rhs.array_h_t << endl;
//#endif //DEBUG

    //cout << "\tblocksize = (" << get_block().x << ", " << get_block().y << ")" << endl;
    //cout << "\tgridsize = (" << get_grid().x << ", " << get_grid().y << ")" << endl;
	array_d = rhs.array_d;
	rhs.array_d = nullptr;

	array_d_t = rhs.array_d_t;
	rhs.array_d_t = nullptr;

	array_d_t_host = rhs.array_d_t_host;
	rhs.array_d_t_host = nullptr;

	array_h = rhs.array_h;
	rhs.array_h = nullptr;

	array_h_t = rhs.array_h_t;
	rhs.array_h_t = nullptr;
//#ifdef DEBUG
//	cout << "finally:" << endl;
//	cout << "array_d:        this" << array_d       << "\trhs:" << rhs.array_d << endl;
//	cout << "array_d_t:      this" << array_d_t     << "\trhs:" << rhs.array_d_t << endl;
//	cout << "array_d_t_host: this" << array_d_t_host << "\trhs:" << rhs.array_d_t_host << endl;
//	cout << "array_h:        this" << array_h       << "\trhs:" << rhs.array_h << endl;
//	cout << "array_h_t:      this" << array_h_t     << "\trhs:" << rhs.array_h_t << endl;
//#endif //DEBUG
}

template <typename T>
cuda_array<T> :: ~cuda_array()
{
    if(array_d_t_host != nullptr)
		delete [] array_d_t_host;
	if(array_h_t != nullptr)
		delete [] array_h_t;
	if(array_d_t != nullptr)
		gpuErrchk(cudaFree(array_d_t));
    if(array_h != nullptr)
#ifdef PINNED_HOST_MEMORY
    cudaFreeHost(array_h);
#endif
#ifndef PINNED_HOST_MEMORY
    free(array_h);
#endif

    if(array_d != nullptr)
        gpuErrchk(cudaFree(array_d));
    //cudaDeviceSynchronize();
}


// Enumerate array elements to check correct kernel access
template <typename T>
void cuda_array<T> :: enumerate_array(const uint t)
{
	d_enumerate<<<grid, block>>>(array_d, My, Nx);
}


template <typename T>
void cuda_array<T> :: enumerate_array_t(const uint t)
{
	d_enumerate_t<<<grid, block >>>(array_d_t, t, My, Nx);
#ifdef DEBUG
    gpuStatus();
#endif
}


// Apply expression O at time level t
template <typename T>
template <typename O>
void cuda_array<T> :: op_apply_t(const uint tlev)
{
	d_op0_apply<T, O> <<<grid, block>>>(get_array_d(tlev), My, Nx);
#ifdef DEBUG
	gpuStatus();
#endif
}


// Perform scalar operation on time level t
template <typename T>
template <typename O>
inline void cuda_array<T> :: op_scalar_t(const T& rhs, const uint tlev)
{
	d_op1_scalar<T, O> <<<grid, block>>>(get_array_d(tlev), rhs, My, Nx);
#ifdef DEBUG
	gpuStatus();
#endif
}

template <typename T>
template <typename O>
inline void cuda_array<T> :: op_scalar_fun(const cuda::slab_layout_t sl, const uint tlev)
{
    d_op0_apply_fun<T, O> <<<grid, block>>>(get_array_d(tlev), sl, My, Nx);
#ifdef DEBUG
    gpuStatus();
#endif
}


// Perform operation element-wise on time level t
template <typename T>
template <typename O>
inline void cuda_array<T> :: op_array_t(const cuda_array<T>& rhs, const uint tlev)
{
	if ((void*) this == (void*) &rhs)
		throw operator_err(string("cuda_array<T>& cuda_array<T> :: operator+= (const cuda_array<T>&): RHS and LHS cannot be the same\n"));
	check_bounds(rhs.get_my(), rhs.get_nx());

	d_op1_arr<T, O><<<grid, block>>>(get_array_d(tlev), rhs.get_array_d(tlev), My, Nx);
#ifdef DEBUG
	gpuStatus();
#endif
}


// Operators

// Copy data from array_d_t[0] from rhs to lhs
template <typename T>
cuda_array<T>& cuda_array<T> :: operator= (const cuda_array<T>& rhs)
{
    // Check if we assign to ourself
    if (this == &rhs)
        return *this;
    // check bounds
    check_bounds(rhs.get_tlevs(), rhs.get_my(), rhs.get_nx());
    gpuErrchk(cudaMemcpy(get_array_d(), rhs.get_array_d(), tlevs * My * Nx * sizeof(T), cudaMemcpyDeviceToDevice));
    return (*this);
}



template <typename T>
cuda_array<T>& cuda_array<T> :: operator= (cuda_array<T>&& rhs)
{
	if(this == &rhs)
		return *this;
	check_bounds(rhs.get_tlevs(), rhs.get_my(), rhs.get_nx());

    if(array_d != nullptr)
        gpuErrchk(cudaFree(array_d));
    array_d = rhs.array_d;
	rhs.array_d = nullptr;

    if(array_d_t != nullptr)
        gpuErrchk(cudaFree(array_d_t));
	array_d_t = rhs.array_d_t;
	rhs.array_d_t = nullptr;

    if(array_d_t_host != nullptr)
        delete [] array_d_t_host;
	array_d_t_host = rhs.array_d_t_host;
	rhs.array_d_t_host = nullptr;

    if(array_h != nullptr)
#ifdef PINNED_HOST_MEMORY
        cudaFreeHost(array_h);
#endif
#ifndef PINNED_HOST_MEMORY
        delete [] array_h;
#endif
	array_h = rhs.array_h;
	rhs.array_h = nullptr;

    if(array_h_t != nullptr)
        delete [] array_h_t;
	array_h_t = rhs.array_h_t;
	rhs.array_h_t = nullptr;

	return(*this);
}


// set data on time level 0 to rhs
template <typename T>
cuda_array<T>& cuda_array<T> :: operator= (const T& rhs)
{
	op_scalar_t<d_op1_assign<T> >(rhs, (uint) 0);
    return(*this);
}




// Perform arithmetic operation on time level 0
template <typename T>
cuda_array<T>& cuda_array<T> :: operator+=(const cuda_array<T>& rhs)
{
	op_array_t<d_op1_addassign<T> >(rhs, (uint) 0);
    return *this;
}


template <typename T>
cuda_array<T>& cuda_array<T> :: operator+=(const T& rhs)
{
	op_scalar_t<d_op1_addassign<T> >(rhs, (uint) 0);
    return *this;
}


template <typename T>
cuda_array<T> cuda_array<T> :: operator+(const cuda_array<T>& rhs) const
{
    cuda_array<T> result(*this);
    result += rhs;
    return result;
}


template <typename T>
cuda_array<T> cuda_array<T> :: operator+(const T& rhs) const
{
    cuda_array<T> result(*this);
    result += rhs;
    return result;
}


template <typename T>
cuda_array<T>& cuda_array<T> :: operator-=(const cuda_array<T>& rhs)
{
	op_array_t<d_op1_subassign<T> >(rhs, (uint) 0);
    return *this;
}


template <typename T>
cuda_array<T>& cuda_array<T> :: operator-=(const T& rhs)
{
	op_scalar_t<d_op1_subassign<T> >(rhs, (uint) 0);
    return *this;
}


template <typename T>
cuda_array<T> cuda_array<T> :: operator-(const cuda_array<T>& rhs) const
{
    cuda_array<T> result(*this);
    result -= rhs;
    return result;
}


template <typename T>
cuda_array<T> cuda_array<T> :: operator-(const T& rhs) const
{
    cuda_array<T> result(*this);
    result -= rhs;
    return result;
}


template <typename T>
cuda_array<T>& cuda_array<T> :: operator*=(const cuda_array<T>& rhs)
{
	op_array_t<d_op1_mulassign<T> >(rhs, (uint) 0);
    return *this;
}


template <typename T>
cuda_array<T>& cuda_array<T> :: operator*=(const T& rhs)
{
	op_scalar_t<d_op1_mulassign<T> >(rhs, (uint) 0);
    return (*this);
}


template <typename T>
cuda_array<T> cuda_array<T> :: operator*(const cuda_array<T>& rhs) const
{
    cuda_array<T> result(*this);
    result *= rhs;
    return result;
}


template <typename T>
cuda_array<T> cuda_array<T> :: operator*(const T& rhs) const
{
    cuda_array<T> result(*this);
    result *= rhs;
    return result;
}


template <typename T>
cuda_array<T>& cuda_array<T> :: operator/=(const cuda_array<T>& rhs)
{
	op_array_t<d_op1_divassign<T> >(rhs, (uint) 0);
    return *this;
}


template <typename T>
cuda_array<T>& cuda_array<T> :: operator/=(const T& rhs)
{
	op_scalar_t<d_op1_divassign<T> >(rhs, (uint) 0);
    return (*this);
}


template <typename T>
cuda_array<T> cuda_array<T> :: operator/(const cuda_array<T>& rhs) const
{
    cuda_array<T> result(*this);
    result /= rhs;
    return result;
}


template <typename T>
cuda_array<T> cuda_array<T> :: operator/(const T& rhs) const
{
    cuda_array<T> result(*this);
    result /= rhs;
    return result;
}


template <typename T>
inline T& cuda_array<T> :: operator()(uint t, uint m, uint n)
{
	check_bounds(t, m, n);
	return (*(array_h_t[t] + address(m, n)));
}


template <typename T>
inline T cuda_array<T> :: operator()(uint t, uint m, uint n) const
{
	check_bounds(t, m, n);
	return (*(array_h_t[t] + address(m, n)));
}


template <typename T>
void cuda_array<T> :: advance()
{
	//Advance array_d_t pointer on device
    //cout << "=============================================================================";
    //cout << "advance\n";
    //cout << "before advancing\n";
    //for(int t = tlevs - 1; t >= 0; t--)
    //    cout << "array_d_t_host["<<t<<"] = " << array_d_t_host[t] << "\n";

    // Cycle pointer array on device and zero out last time level
	d_advance<<<1, 1>>>(array_d_t, tlevs);
    // Cycle pointer array on host
    T* tmp = array_d_t_host[tlevs - 1];
    for(int t = tlevs - 1; t > 0; t--)
        array_d_t_host[t] = array_d_t_host[t - 1];
    array_d_t_host[0] = tmp;

    // get_array_d[t] returns array_d_t_host[t]. Zero out t=0 after we have cycled
    // the pointer to device data at different time levels in the host pointer too.
	d_op1_scalar<T, d_op1_assign<T> > <<<grid, block>>>(get_array_d(0), T(0.0), My, Nx);
    //cout << "after advancing\n";
    //for(int t = tlevs - 1; t >= 0; t--)
    //     cout << "array_d_t_host["<<t<<"] = " << array_d_t_host[t] << "\n";
    //cout << "=============================================================================";

    // Zero out last time level
}


// Copy all time levels from device to host buffer
template <typename T>
inline void cuda_array<T> :: copy_device_to_host()
{
    const size_t line_size = My * Nx * sizeof(T);
    for(uint t = 0; t < tlevs; t++)
    {
        gpuErrchk(cudaMemcpy(&array_h[t * My * Nx], get_array_d(t), line_size, cudaMemcpyDeviceToHost));
    }
}


// Copy from deivce to host buffer for specified time level
template <typename T>
inline void cuda_array<T> :: copy_device_to_host(uint tlev)
{
    const size_t line_size = My * Nx * sizeof(T);
    gpuErrchk(cudaMemcpy(array_h_t[tlev], get_array_d(tlev), line_size, cudaMemcpyDeviceToHost));
}


// Copy from device to rhs host buffer for time level =0
template <typename T>
inline void cuda_array<T> :: copy_device_to_host(T* buffer)
{
    const size_t line_size = My * Nx * sizeof(T);
    gpuErrchk(cudaMemcpy(buffer, get_array_d(0), line_size, cudaMemcpyDeviceToHost));
}


// Copy data at time level tlev to device data pointer
template <typename T>
inline void cuda_array<T> :: copy_device_to_device(const uint tlev, T* buffer)
{
	const size_t line_size = My * Nx * sizeof(T);
	gpuErrchk(cudaMemcpy(buffer, get_array_d(tlev), line_size, cudaMemcpyDeviceToDevice));
}


// Copy all time levels from device to host buffer
template <typename T>
inline void cuda_array<T> :: copy_host_to_device()
{
    const size_t line_size = My * Nx * sizeof(T);
    for(uint t = 0; t < tlevs; t++)
    {
        gpuErrchk(cudaMemcpy(get_array_d(t), &array_h[t * My * Nx], line_size, cudaMemcpyHostToDevice));
    }
}


// Copy from deivce to host buffer for specified time level
template <typename T>
inline void cuda_array<T> :: copy_host_to_device(uint tlev)
{
    const size_t line_size = My * Nx * sizeof(T);
    gpuErrchk(cudaMemcpy(get_array_d(tlev), array_h_t[tlev], line_size, cudaMemcpyHostToDevice));
}


// Copy from device to rhs host buffer for time level =0
template <typename T>
inline void cuda_array<T> :: copy_host_to_device(T* rhs)
{
    const size_t line_size = My * Nx * sizeof(T);
    gpuErrchk(cudaMemcpy(get_array_d(0), rhs, line_size, cudaMemcpyHostToDevice));
}

template <typename T>
inline void cuda_array<T> :: copy(uint t_dst, uint t_src)
{
    const size_t line_size = My * Nx * sizeof(T);
    gpuErrchk(cudaMemcpy(get_array_d(t_dst), get_array_d(t_src), line_size, cudaMemcpyDeviceToDevice));
}


template <typename T>
inline void cuda_array<T> :: copy(uint t_dst, const cuda_array<T>& src, uint t_src)
{
    const size_t line_size = My * Nx * sizeof(T);
    gpuErrchk(cudaMemcpy(get_array_d(t_dst), src.get_array_d(t_src), line_size, cudaMemcpyDeviceToDevice));
}


template <typename T>
inline void cuda_array<T> :: move(uint t_dst, uint t_src)
{
    // Copy data
    const size_t line_size = My * Nx * sizeof(T);
    gpuErrchk(cudaMemcpy(get_array_d(t_dst), get_array_d(t_src), line_size, cudaMemcpyDeviceToDevice));
    // Clear source
	d_op1_scalar<T, d_op1_assign<T> > <<<grid, block>>>(get_array_d(0), T(0.0), My, Nx);
#ifdef DEBUG
    gpuStatus();
#endif
}


template <typename T>
inline void cuda_array<T> :: swap(uint t1, uint t2)
{
	// swap pointers to timelevel t1 and t2 on device
    d_swap<<<1, 1>>>(array_d_t, t1, t2);
    // Update pointers to time level on device in host data structure
    gpuErrchk(cudaMemcpy(get_array_d(0), array_d_t, sizeof(T*) * tlevs, cudaMemcpyDeviceToHost));
    //cudaDeviceSynchronize();
}


template <typename T>
inline void cuda_array<T> :: kill_kx0()
{
    d_kill_kx0<<<grid.x, block.x>>>(array_d, My, Nx);
#ifdef DEBUG
    gpuStatus();
#endif
}


/// Do this for T= CuCmplx<%>
template <typename T>
inline void cuda_array<T> :: kill_ky0()
{
    d_kill_ky0<<<grid.y, block.y>>>(array_d, My, Nx);
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

/// Do this for T = CuCmplx<%>
template <typename T>
inline void cuda_array<T> :: normalize()
{
    cuda::real_t norm = 1. / T(My * Nx);
	d_op1_scalar<T, d_op1_mulassign<T> > <<<grid, block>>>(get_array_d(0), T(norm), My, Nx);
#ifdef DEBUG
    gpuStatus();
#endif
}

#endif // __CUDACC__


#endif /* CUDA_ARRAY4_H */
