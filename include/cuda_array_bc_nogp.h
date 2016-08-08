/*
 * cuda_array_bc.h
 *
 *  Created on: May 13, 2016
 *      Author: ralph
 *
 * Array type with variable padding in x- and y-direction.
 * Knows about boundary conditions, routines operating on this datatype are to compute
 * them on the fly
 *
 * Memory Layout
 *
 * rows: 0...My-1 ... My-1 + pad_y
 * cols: 0...Nx-1 ... Nx-1 + pad_x
 *
 *     0                        My-1 ... My - 1 + pad_y
 * Nx - 1 |--------- ... ------|    |
 *        |--------- ... ------|    |
 * ...
 *  0     |--------- ... ------|    |
 *
 * idx = n * (My + pad_y) + m
 *
 * columns (m, y-direction) are consecutive in memory
 *
 *
 * Mapping of CUDA threads on the array:
 *
 * Columns: 0..My - 1 + pad_y -> col = blockIdx.x * blockDim.x + threadIdx.x
 * Rows:    0..Nx - 1 + pad_x -> row = blockIdx.y * blockDim.y + threadIdx.y
 *
 * dimBlock = (blocksize_row, blocksize_col)
 * dimGrid = (My + pad_y) / blocksize_row, (My + pad_y) / blocksize_col
 *
 * Ghost points are to be computed on the fly, not stored in memory
 * They can be access by the address object
 *
 *
 */

#ifndef cuda_array_bc_H_
#define cuda_array_bc_H_

#include <iostream>
#include <iomanip>
#include <map>
#include <functional>
#include <sstream>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "bounds.h"
#include "address.h"
#include "error.h"
#include "cuda_types.h"
#include "allocators.h"


typedef unsigned int uint;
typedef double real_t;

// Error checking macro for cuda calls
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess)
    {
        std::stringstream err_str;
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
         std::stringstream err_str;
        err_str << "GPUassert: " << cudaGetErrorString(error) << "\t file: " << file << ", line: " << line << "\n";
        throw gpu_error(err_str.str());
     }
}

#ifdef __CUDACC__

/// Device function to compute column and row
__device__ inline size_t d_get_col() {
    return (blockIdx.x * blockDim.x + threadIdx.x);
}


__device__ inline size_t d_get_row() {
    return (blockIdx.y * blockDim.y + threadIdx.y);
}


/// Return true if row and column are within geom(excluding padded rows/cols)
/// Return false if row or column is outside the geometry
__device__ inline bool good_idx(size_t row, size_t col, const cuda::slab_layout_t geom)
{
    return((row < geom.get_nx()) && (col < geom.get_my()));
}


template <typename T>
__global__
void kernel_alloc_array_d_t(T** array_d_t, T* array, const cuda::slab_layout_t geom, const size_t tlevs)
{
    for(size_t t = 0; t < tlevs; t++)
    {
        array_d_t[t] = &array[t * (geom.get_nx() + geom.get_pad_x()) * (geom.get_my() + geom.get_pad_y())];
    }
}


/// Evaluate the function lambda d_op_fun (with type given by template parameter O)
/// on the cell centers
template <typename T, typename O>
__global__
void kernel_evaluate(T* array_d_t, O d_op_fun, const cuda::slab_layout_t geom) 
{
    const size_t col{d_get_col()};
    const size_t row{d_get_row()};
    const size_t index{row * (geom.get_my() + geom.get_pad_y()) + col};

	if (good_idx(row, col, geom))
		array_d_t[index] = d_op_fun(row, col, geom);
}


/// Evaluate the function lambda d_op_fun (with type given by template parameter O)
/// on the cell centers. Same as kernel_evaluate, but pass the value at the cell center
/// to the lambda function 
template <typename T, typename O>
__global__
void kernel_evaluate_2(T* array_d_t, O op_func, const cuda::slab_layout_t geom) 
{
	const size_t col{d_get_col()};
    const size_t row{d_get_row()};
    const size_t index{row * (geom.get_my() + geom.get_pad_y()) + col};

	if (good_idx(row, col, geom))
		array_d_t[index] = op_func(array_d_t[index], row, col, geom);
}


/// Perform arithmetic operation lhs[idx] = op(lhs[idx], rhs[idx])
template<typename T, typename O>
__global__
void kernel_op1_arr(T* lhs, T* rhs, O op_func, const cuda::slab_layout_t geom)
{
    const size_t col{d_get_col()};
    const size_t row{d_get_row()};
    const size_t index{row * (geom.get_my() + geom.get_pad_y()) + col};

    if(good_idx(row, col, geom))
        lhs[index] = op_func(lhs[index], rhs[index]);
}


/// For accessing elements in GPU kernels and interpolating ghost points
template <typename T>
__global__
void kernel_init_address(address_t<T>** my_address, 
        T* data, 
        const cuda::slab_layout_t geom, 
        const cuda::bvals_t<T> bvals)
{
    *my_address = new address_t<T>(geom, bvals);
    //printf("kernel_init_address: address_t at %p. Value: %f\n", *my_address, (*(*my_address))(data, -1, 0));
}


template <typename T>
__global__
void kernel_free_address(address_t<T>** my_address)
{
    //printf("kernel_free_address: address_t at %p\n", *my_address);
    delete *my_address;
}


/// Reduction kernel, taken from cuda_darray.h
// Perform reduction of in_data, stored in column-major order
// Use stride_size = 1, offset_size = Nx for row-wise reduction (threads in one block reduce one row, i.e. consecutive elements of in_data)
// row-wise reduction:
// stride_size = 1
// offset_size = Nx
// blocksize = (Nx, 1)
// gridsize = (1, My)
//
// column-wise reduction:
// stride_size = My
// offset_size = 1
// blocksize = (My, 1)
// gridsize = (1, Nx)
template <typename T, typename O>
__global__ void kernel_reduce(const T* __restrict__ in_data, 
                              T* __restrict__ out_data, 
                              O op_func, 
                              const size_t stride_size, const size_t offset_size, const size_t Nx, const size_t My)
{
	extern __shared__ T sdata[];

	const size_t tid = threadIdx.x;
	const size_t idx_data = tid * stride_size + blockIdx.y * offset_size;
	const size_t idx_out = blockIdx.y;
	if(idx_data < Nx * My)
	{
        sdata[tid] = in_data[idx_data];
	    // reduction in shared memory
        __syncthreads();
	    for(size_t s = 1; s < blockDim.x; s *= 2)
	    {
	        if(tid % (2*s) == 0)
	        {
	            sdata[tid] = op_func(sdata[tid], sdata[tid + s]);
	        }
	        __syncthreads();
	    }
	    // write result for this block to global mem
	    if (tid == 0)
	    {
	    	//printf("threadIdx = %d: out_data[%d] = %f\n", threadIdx.x, row, sdata[0]);
	    	out_data[idx_out] = sdata[0];
	    }
	}
}


#endif // __CUDACC_


template <typename allocator>
class cuda_array_bc_nogp{
public:

    using allocator_t = typename my_allocator_traits<allocator> :: allocator_type;
    using value_t = typename my_allocator_traits<allocator> :: value_type;
    using deleter_t = typename my_allocator_traits<allocator> :: deleter_type;
    using ptr_t = std::unique_ptr<value_t, deleter_t>;

	cuda_array_bc_nogp(const cuda::slab_layout_t, const cuda::bvals_t<value_t>, const size_t);
    cuda_array_bc_nogp(const cuda_array_bc_nogp<allocator>* rhs);
    cuda_array_bc_nogp(const cuda_array_bc_nogp<allocator>& rhs);

	~cuda_array_bc_nogp();

    /// Apply a function to the data
    template <typename F> void evaluate(F, const size_t);
    /// Enumerate data elements (debug function)
	void enumerate();
    void enumerate(const size_t);
    /// Initialize all elements to zero. Making this private results in compile error:
    /// /home/rku000/source/2dads/include/cuda_array_bc_nogp.h(414): error: An explicit __device__ lambda cannot be defined in a member function that has private or protected access within its class ("cuda_array_bc_nogp")
    void initialize();
    void initialize(const size_t);

	inline value_t& operator() (const size_t n, const size_t m) {
        return(*(array_h + n * (get_geom().get_my() + get_geom().get_pad_y()) + m));
    };
	inline value_t operator() (const size_t n, const size_t m) const {
        return(*(array_h + n * (get_geom().get_my() + get_geom().get_pad_y()) + m));
    }

    inline value_t& operator() (const size_t t, const size_t n, const size_t m) {
        return(*(array_h_t[t] + n * (get_geom().get_my() + get_geom().get_pad_y()) + m));
    };
    inline value_t operator() (const size_t t, const size_t n, const size_t m) const {
        return(*(array_h_t[t] + n * (get_geom().get_my() + get_geom().get_pad_y()) + m));
    };

    cuda_array_bc_nogp<allocator>& operator=(const cuda_array_bc_nogp<allocator>&);

    cuda_array_bc_nogp<allocator>& operator+=(const cuda_array_bc_nogp<allocator>&);
    cuda_array_bc_nogp<allocator>& operator-=(const cuda_array_bc_nogp<allocator>&);
    cuda_array_bc_nogp<allocator>& operator*=(const cuda_array_bc_nogp<allocator>&);
    cuda_array_bc_nogp<allocator>& operator/=(const cuda_array_bc_nogp<allocator>&);

    cuda_array_bc_nogp<allocator> operator+(const cuda_array_bc_nogp<allocator>&) const;  
    cuda_array_bc_nogp<allocator> operator-(const cuda_array_bc_nogp<allocator>&) const;  
    cuda_array_bc_nogp<allocator> operator*(const cuda_array_bc_nogp<allocator>&) const;  
    cuda_array_bc_nogp<allocator> operator/(const cuda_array_bc_nogp<allocator>&) const;  

    // Copy device memory to host and print to stdout
    friend std::ostream& operator<<(std::ostream& os, cuda_array_bc_nogp& src)
    {
        const size_t tl{src.get_tlevs()};
        const size_t my{src.get_my()};
        const size_t nx{src.get_nx()};
        const size_t pad_x{src.get_geom().pad_x};
        const size_t pad_y{src.get_geom().pad_y};

        src.copy_device_to_host();

        os << "\n";
        for(size_t t = 0; t < tl; t++)
        {
            for(size_t n = 0; n < nx + pad_x; n++)
            {
                for(size_t m = 0; m < my + pad_y; m++)
                {
                    // Remember to also set precision routines in CuCmplx :: operator<<
                	os << std::setw(cuda::io_w) << std::setprecision(cuda::io_p) << std::fixed << src(n, m) << "\t";
                }
                os << endl;
            }
            os << endl;
        }
        return (os);
    }
	
    void print() const;
	
	void normalize(const size_t);
    value_t L2(const size_t);

	// Copy entire device data to host
	void copy_device_to_host();
	// Copy entire device data to external data pointer in host memory
	//void copy_device_to_host(value_T*);
	// Copy device to host at specified time level
	//void copy_device_to_host(const size_t);

	// Copy deice data at specified time level to external pointer in device memory
	void copy_device_to_device(const size_t, value_t*);

	// Transfer from host to device
	void copy_host_to_device();
	//void copy_host_to_device(const size_t);
	//void copy_host_to_device(value_t*);

	// Advance time levels
	//void advance();

	///@brief Copy data from t_src to t_dst
	//void copy(const size_t t_dst, const size_t t_src);
	///@brief Copy data from src, t_src to t_dst
	//void copy(const size_t t_dst, const cuda_array<allocator>& src, const size_t t_src);
	///@brief Move data from t_src to t_dst, zero out t_src
	//void move(const size_t t_dst, const size_t t_src);
	///@brief swap data in t1, t2
	//void swap(const size_t t1, const size_t t2);
	//void normalize();

	//void kill_kx0();
	//void kill_ky0();
	//void kill_k0();

	// Access to private members
	inline size_t get_nx() const {return(Nx);};
	inline size_t get_my() const {return(My);};
	inline size_t get_tlevs() const {return(tlevs);};
    inline cuda::slab_layout_t get_geom() const {return(geom);};
    inline cuda::bvals_t<value_t> get_bvals() const {return(boundaries);};
    inline address_t<value_t>** get_address() const {return(d_address);};

	inline dim3 get_grid() const {return grid;};
	inline dim3 get_block() const {return block;};

	//bounds get_bounds() const {return check_bounds;};
	// Pointer to host copy of device data
	inline value_t* get_array_h() const {return array_h;};
	inline value_t* get_array_h(size_t t) const {return array_h_t[t];};

	// smart pointer to device data, entire array
	inline ptr_t get_array_d() const {return array_d.get();};
	// Pointer to array of pointers, corresponding to time levels
	inline value_t** get_array_d_t() const {return array_d_t;};
	// Pointer to device data at time level t
	inline value_t* get_array_d(size_t t) const {return array_d_t_host[t];};

	// Check bounds
	inline void check_bounds(size_t t, size_t n, size_t m) const {array_bounds(t, n, m);};
	inline void check_bounds(size_t n, size_t m) const {array_bounds(n, m);};

	// Number of elements
	inline size_t get_nelem_per_t() const {return ((geom.Nx + geom.pad_x) * (geom.My + geom.pad_y));};

    // Set true if transformed
    inline bool is_transformed() const {return(transformed);};
    inline bool set_transformed(bool val) 
    {
        transformed = val; 
        return(transformed);
    };

private:
	const size_t tlevs;
	const size_t Nx;
	const size_t My;
    const bounds array_bounds;
	const cuda::bvals_t<value_t> boundaries;
    const cuda::slab_layout_t geom;
    bool transformed;

    allocator my_alloc;
    address_t<value_t>** d_address;

	// block and grid for access without ghost points, use these normally
	const dim3 block;
	const dim3 grid;

    // Size of shared memory bank
    const size_t shmem_size_col;   
	// Array data is on device
	// Pointer to device data
	ptr_t array_d;
	// Pointer to each time stage. Pointer to array of pointers on device
	value_t** array_d_t;
	// Pointer to each time stage: Pointer to each time level on host
	value_t** array_d_t_host;
	value_t* array_h;
	value_t** array_h_t;	
};


template < typename allocator>
cuda_array_bc_nogp<allocator> :: cuda_array_bc_nogp 
        (const cuda::slab_layout_t _geom, const cuda::bvals_t<value_t> bvals, size_t _tlevs) : 
		tlevs(_tlevs), 
        Nx(_geom.get_nx()), 
        My(_geom.get_my()), 
        array_bounds(get_tlevs(), get_nx(), get_my()),
        boundaries(bvals), 
        geom(_geom), 
        transformed{false},
        d_address{nullptr},
        block(dim3(cuda::blockdim_row, cuda::blockdim_col)),
		grid(dim3(((get_my() + geom.get_pad_y()) + cuda::blockdim_row - 1) / cuda::blockdim_row, 
                  ((get_nx() + geom.get_pad_x()) + cuda::blockdim_col - 1) / cuda::blockdim_col)),
        shmem_size_col(get_nx() * sizeof(value_t)),
        array_d(my_alloc.alloc(get_tlevs() * get_nelem_per_t() * sizeof(value_t))),
		array_d_t(nullptr),
		array_d_t_host(new value_t*[get_tlevs()]),
		array_h(new value_t[get_tlevs() * get_nelem_per_t()]),
		array_h_t(new value_t*[get_tlevs()])
{
	gpuErrchk(cudaMalloc( (void***) &array_d_t, get_tlevs() * sizeof(value_t*)));

    //cout << "cuda_array_bc<allocator> ::cuda_array_bc<allocator>\t";
    //cout << "Nx = " << Nx << ", pad_x = " << geom.pad_x << ", My = " << My << ", pad_y = " << geom.pad_y << endl;
    //cout << "block = ( " << block.x << ", " << block.y << ")" << endl;
    //cout << "grid = ( " << grid.x << ", " << grid.y << ")" << endl;
    //cout << geom << endl;

	kernel_alloc_array_d_t<<<1, 1>>>(array_d_t, array_d.get(), get_geom(), get_tlevs());
	gpuErrchk(cudaMemcpy(array_d_t_host, array_d_t, sizeof(value_t*) * get_tlevs(), cudaMemcpyDeviceToHost));

	for(size_t t = 0; t < tlevs; t++)
    {
        array_h_t[t] = &array_h[t * get_nelem_per_t()];
    }
    initialize();

    gpuErrchk(cudaMalloc(&d_address, sizeof(address_t<value_t>**)));

    kernel_init_address<<<1, 1>>>(get_address(), get_array_d(0), get_geom(), get_bvals());
}


template <typename allocator>
cuda_array_bc_nogp<allocator> :: cuda_array_bc_nogp(const cuda_array_bc_nogp<allocator>* rhs) :
    cuda_array_bc_nogp(rhs -> get_geom(), rhs -> get_bvals(), rhs -> get_tlevs()) 
{
    check_bounds(rhs -> get_tlevs(), rhs -> get_nx(), rhs -> get_my());
    gpuErrchk(cudaMemcpy(get_array_d(0),
                         rhs -> get_array_d(0),
                         sizeof(value_t) * get_tlevs() * get_nelem_per_t(), 
                         cudaMemcpyDeviceToDevice));
};


template <typename allocator>
cuda_array_bc_nogp<allocator> :: cuda_array_bc_nogp(const cuda_array_bc_nogp<allocator>& rhs) :
    cuda_array_bc_nogp(rhs.get_geom(), rhs.get_bvals(), rhs.get_tlevs()) 
{
    check_bounds(rhs -> get_tlevs(), rhs -> get_nx(), rhs -> get_my());
    gpuErrchk(cudaMemcpy(get_array_d(0),
                         rhs.get_array_d(0),
                         sizeof(value_t) * get_tlevs() * get_nelem_per_t(), 
                         cudaMemcpyDeviceToDevice));
};


template <typename allocator>
cuda_array_bc_nogp<allocator> :: ~cuda_array_bc_nogp()
{
	if(array_h != nullptr)
		delete [] array_h;
	if(array_h_t != nullptr)
		delete [] array_h_t;
	if(array_d_t != nullptr)
		cudaFree(array_d_t);

    kernel_free_address<<<1, 1>>>(get_address());
}
    

template <typename allocator>
void cuda_array_bc_nogp<allocator> :: enumerate(const size_t tlev)
{
        evaluate([=] __device__ (const size_t n, const size_t m, cuda::slab_layout_t geom) -> value_t
                                 {
                                     return(n * (geom.get_my() + geom.get_pad_y()) + m);
                                 }, tlev);
}

template <typename allocator>
void cuda_array_bc_nogp<allocator> :: enumerate()
{
	for(size_t t = 0; t < tlevs; t++)
	{
        enumerate(t);
    }
}


template <typename allocator>
void cuda_array_bc_nogp<allocator> :: initialize(const size_t tlev)
{
        evaluate([=] __device__ (size_t n, size_t m, cuda::slab_layout_t geom) -> value_t
                                 {
                                    return(value_t(4.2));
                                 }, tlev);
}

template <typename allocator>
void cuda_array_bc_nogp<allocator> :: initialize()
{
    for(size_t t = 0; t < tlevs; t++)
    {
        initialize(t);
    }
}


template <typename allocator>
cuda_array_bc_nogp<allocator>& cuda_array_bc_nogp<allocator> :: operator= (const cuda_array_bc_nogp<allocator>& rhs)
{
    // Check for self-assignment
    if(this == &rhs)
        return(*this);

    check_bounds(rhs.get_tlevs(), rhs.get_nx(), rhs.get_my());
    gpuErrchk(cudaMemcpy(get_array_d(0),
                         rhs.get_array_d(0),
                         get_nelem_per_t() * sizeof(value_t), cudaMemcpyDeviceToDevice));
    return (*this);
}

template <typename allocator>
cuda_array_bc_nogp<allocator>& cuda_array_bc_nogp<allocator> :: operator+=(const cuda_array_bc_nogp<allocator>& rhs) 
{
    check_bounds(rhs.get_tlevs(), rhs.get_nx(), rhs.get_my());
    kernel_op1_arr<<<get_grid(), get_block()>>>(get_array_d(0), rhs.get_array_d(0),
                                                [=] __device__ (value_t lhs, value_t rhs) -> value_t
                                                {
                                                    return(lhs + rhs);
                                                }, get_geom());
    return *this;
}


template <typename allocator>
cuda_array_bc_nogp<allocator>& cuda_array_bc_nogp<allocator> :: operator-=(const cuda_array_bc_nogp<allocator>& rhs) 
{
    check_bounds(rhs.get_tlevs(), rhs.get_nx(), rhs.get_my());
    kernel_op1_arr<<<get_grid(), get_block()>>>(get_array_d(0), rhs.get_array_d(0),
                                                [=] __device__ (value_t lhs, value_t rhs) -> value_t
                                                {
                                                    return(lhs - rhs);
                                                }, get_geom());
    return *this;
}


template <typename allocator>
cuda_array_bc_nogp<allocator>& cuda_array_bc_nogp<allocator> :: operator*=(const cuda_array_bc_nogp<allocator>& rhs) 
{
    check_bounds(rhs.get_tlevs(), rhs.get_nx(), rhs.get_my());
    kernel_op1_arr<<<get_grid(), get_block()>>>(get_array_d(0), rhs.get_array_d(0),
                                                [=] __device__ (value_t lhs, value_t rhs) -> value_t
                                                {
                                                    return(lhs * rhs);
                                                }, get_geom());
    return *this;
}


template <typename allocator>
cuda_array_bc_nogp<allocator>& cuda_array_bc_nogp<allocator> :: operator/=(const cuda_array_bc_nogp<allocator>& rhs) 
{
    check_bounds(rhs.get_tlevs(), rhs.get_nx(), rhs.get_my());
    kernel_op1_arr<<<get_grid(), get_block()>>>(get_array_d(0), rhs.get_array_d(0),
                                                [=] __device__ (value_t lhs, value_t rhs) -> value_t
                                                {
                                                    return(lhs / rhs);
                                                }, get_geom());
    return *this;
}


template <typename allocator>
cuda_array_bc_nogp<allocator> cuda_array_bc_nogp<allocator> :: operator+(const cuda_array_bc_nogp<allocator>& rhs) const
{
    cuda_array_bc_nogp<allocator> result(this);
    result += rhs;
    return(result);
}


template <typename allocator>
cuda_array_bc_nogp<allocator> cuda_array_bc_nogp<allocator> :: operator-(const cuda_array_bc_nogp<allocator>& rhs) const
{
    cuda_array_bc_nogp<allocator> result(this);
    result -= rhs;
    return(result);
}


template <typename allocator>
cuda_array_bc_nogp<allocator> cuda_array_bc_nogp<allocator> :: operator*(const cuda_array_bc_nogp<allocator>& rhs) const
{
    cuda_array_bc_nogp<allocator> result(this);
    result *= rhs;
    return(result);
}


template <typename allocator>
cuda_array_bc_nogp<allocator> cuda_array_bc_nogp<allocator> :: operator/(const cuda_array_bc_nogp<allocator>& rhs) const
{
    cuda_array_bc_nogp<allocator> result(this);
    result /= rhs;
    return(result);
}


template <typename allocator>
template <typename F>
void cuda_array_bc_nogp<allocator> :: evaluate(F myfunc, const size_t tlev)
{
    kernel_evaluate<<<grid, block>>>(get_array_d(tlev), myfunc, geom);
}


template <typename allocator>
void cuda_array_bc_nogp<allocator> :: print() const
{
	for(size_t t = 0; t < tlevs; t++)
	{
        cout << "print = " << t << endl;
		for (size_t n = 0; n < get_geom().get_nx() + get_geom().get_pad_x(); n++)
		{
			for(size_t m = 0; m < get_geom().get_my() + get_geom().get_pad_y(); m++)
			{
				cout << std::setw(8) << std::setprecision(5) << (*this)(t, n, m) << "\t";
                //cout << std::setw(7) << std::setprecision(5) << *(array_h_t[t] + n * (geom.My + geom.pad_y) + m) << "\t";
			}
			cout << endl;
		}
        cout << endl << endl;
	}
}


template <typename allocator>
void cuda_array_bc_nogp<allocator> :: normalize(const size_t tlev)
{
    // If we made a 1d DFT normalize by My. Otherwise nomalize by Nx * My
    switch (boundaries.get_bc_left())
    {
        case cuda::bc_t::bc_dirichlet:
            // fall through
        case cuda::bc_t::bc_neumann:
            cout << " Normalizing for 1d" << endl;
            kernel_evaluate_2<<<grid, block>>>(get_array_d(tlev), [=] __device__ (value_t in, size_t n, size_t m, cuda::slab_layout_t geom) -> value_t
            {
                return(in / value_t(geom.get_my()));
            }, get_geom());
            break;

        case cuda::bc_t::bc_periodic:
            cout << " Normalizing for 2d" << endl;
            kernel_evaluate_2<<<grid, block>>>(get_array_d(tlev), [=] __device__ (value_t in, size_t n, size_t m, cuda::slab_layout_t geom) -> value_t
            {
                return(in / value_t(geom.get_nx() * geom.get_my()));
            }, get_geom());
            break;
    }
}


// Computes the L2 norm
// Problem: when value_t is CuCmplx<T>, we return a CuCmplx<T>, when value_t is double, we return double.
// Anyway, the Lambda calls abs(in), which does not give the correct result when value_t is CuCmplx<T>.
//
// For now, just use this routine if we have doubles!
//
template <typename allocator>
cuda_array_bc_nogp<allocator>::value_t cuda_array_bc_nogp<allocator> :: L2(const size_t tlev)
{
    // Configuration for reduction kernel
    const size_t shmem_size_row = get_nx() * sizeof(value_t);
    const dim3 blocksize_row(static_cast<int>(get_nx()), 1, 1);
    const dim3 gridsize_row(1, static_cast<int>(get_my()), 1);

    // temporary value profile
    //value_t* h_tmp_profile(new value_t[Nx]);
    // return value
    ptr_t d_rval_ptr(my_alloc.alloc(sizeof(value_t)));
    value_t rval{0.0};

    if(tlev < get_tlevs())
    {
        // Store result of 2d->1d reduction
        ptr_t d_tmp_profile(my_alloc.alloc(get_nx() * sizeof(value_t)));
        // Copy data to non-strided memory layout
        ptr_t device_copy(my_alloc.alloc(get_nx() * get_my() * sizeof(value_t)));
        // Geometry of the temporary array, no padding
        cuda::slab_layout_t tmp_geom{get_geom().get_xleft(), get_geom().get_deltax(), get_geom().get_ylo(), get_geom().get_deltay(),
                                     get_geom().get_nx(), 0, get_geom().get_my(), 0};
        for(size_t n = 0; n < get_nx(); n++)
        {
            gpuErrchk(cudaMemcpy((void*) (device_copy.get() + n * get_my()),
                                 (void*) (get_array_d(tlev) + n * (get_my() + geom.get_pad_y())), 
                                 get_my() * sizeof(value_t), 
                                 cudaMemcpyDeviceToDevice));
        }
        // Take the square of the absolute value
        kernel_evaluate_2<<<get_grid(), get_block()>>>(device_copy.get(),
                                                       [=] __device__ (value_t in, size_t n, size_t m, cuda::slab_layout_t geom ) -> value_t
                                                       {
                                                           return(abs(in) * abs(in));
                                                           //return(in);
                                                       },
                                                       tmp_geom);
        //value_t* tmp_arr(new value_t[Nx * My]);
        //gpuErrchk(cudaMemcpy(tmp_arr, device_copy.get(), get_nx() * get_my() * sizeof(value_t), cudaMemcpyDeviceToHost));
        //for(size_t n = 0; n < get_nx(); n++)
        //{
        //    for(size_t m = 0; m < get_my(); m++)
        //    {
        //        cout << tmp_arr[n * get_my() + m] << "\t";
        //    }
        //    cout << endl;
        //}
        //delete [] tmp_arr;
        // Perform 2d -> 1d reduction
        kernel_reduce<<<gridsize_row, blocksize_row, shmem_size_row>>>(device_copy.get(), d_tmp_profile.get(), 
                                                                       [=] __device__ (value_t op1, value_t op2) -> value_t {return(op1 + op2);},
                                                                       1, get_nx(), get_nx(), get_my());
        //gpuErrchk(cudaMemcpy(h_tmp_profile, d_tmp_profile.get(), get_nx() * sizeof(value_t), cudaMemcpyDeviceToHost));
        //for(size_t n = 0; n < Nx; n++)
        //{
        //    cout << n << ": " << h_tmp_profile[n] << endl;
        //}
        // Perform 1d -> 0d reduction
        kernel_reduce<<<1, get_nx(), shmem_size_row>>>(d_tmp_profile.get(), d_rval_ptr.get(), 
                                                       [=] __device__ (value_t op1, value_t op2) -> value_t {return(op1 + op2);},
                                                       1, get_nx(), get_nx(), 1);
        gpuErrchk(cudaMemcpy(&rval, (void*) d_rval_ptr.get(), sizeof(value_t), cudaMemcpyDeviceToHost));
    }
    else
    {
        stringstream err_msg;
        err_msg << "cuda_array_bc_nogp<allocator> :: value_t cuda_array_bc_nogp<allocator> :: L2(const size_t tlev):    ";
        err_msg << "tlev = " << tlev << " is out of bounds: get_tlevs() = " << get_tlevs() << "\n";
        throw out_of_bounds_err(err_msg.str());
    }

    //delete [] h_tmp_profile;
    return(sqrt(rval / static_cast<value_t>(get_nx() * get_my())));
}

template <typename allocator>
void cuda_array_bc_nogp<allocator>::copy_device_to_host()
{
    for(size_t t = 0; t < tlevs; t++)
        gpuErrchk(cudaMemcpy(&array_h[t * get_nelem_per_t()], get_array_d(t), sizeof(value_t) * get_nelem_per_t(), cudaMemcpyDeviceToHost));	
}


template <typename allocator>
void cuda_array_bc_nogp<allocator> :: copy_host_to_device()
{ 
    for(size_t t = 0; t < tlevs; t++)
        gpuErrchk(cudaMemcpy(&array_h[t * get_nelem_per_t()], get_array_d(t), sizeof(value_t) * get_nelem_per_t(), cudaMemcpyDeviceToHost));
}



#endif /* cuda_array_bc_H_ */
