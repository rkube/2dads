/// @detailed Array used in diagnostics functions
///
/// @detailed From cuda_array. Includes function to compute poloidal profile
/// @detailed fluctuation around profile.
/// @detailed cuda_darray inherits from cuda_array<double, double> and has thus only
/// @detailed one template parameter (T = double, float, ...)


#ifndef CUDA_DARRAY_H
#define CUDA_DARRAY_H

#include <cstring>
#include <iostream>
#include <cassert>
#include "error.h"
#include "cuda_array4.h"

#ifdef __CUDACC__


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
__global__ void d_reduce(const T* __restrict__ in_data, T* __restrict__ out_data, uint stride_size, uint offset_size, uint Nx, uint My)
{
	extern __shared__ T sdata[];

	uint tid = threadIdx.x;
	uint idx_data = tid * stride_size + blockIdx.y * offset_size;
	uint idx_out = blockIdx.y;
	if(idx_data < Nx * My)
	{
		O op;
		sdata[tid] = in_data[idx_data];
	    // reduction in shared memory
	    for(uint s = 1; s < blockDim.x; s *= 2)
	    {
	        if(tid % (2*s) == 0)
	        {
	            //sdata[tid] = op(sdata[tid], sdata[tid + s]);
	        	op(sdata[tid], sdata[tid + s]);
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


template <typename T>
__global__ void d_enumerate_profile(T* in_data, T* profile, uint Nx)
{
	extern __shared__ T sdata[];

	uint tid = threadIdx.x;
	uint row = threadIdx.x * Nx;

	sdata[tid] = in_data[row];

	profile[tid] = sdata[tid];
}


// Construct tilde or bar arrays - operate column-wise.
// T* in_data: 2d cuda_array<T> data, Nx columns, My rows
// T* profile:data: 1d cuda_array<T> data, Nx columns
// O op: op is either assignment (=), for bar or sub (for tilde)
template <typename T, typename O>
__global__ void d_op_1d2d_col(T* __restrict__ in_data , const T* __restrict__ profile_data, uint Nx, uint My)
{
	uint col = blockIdx.x * blockDim.x + threadIdx.x;
	uint row = blockIdx.y * blockDim.y + threadIdx.y;
	uint idx = row * Nx + col;

	if((col < Nx) && (row < My))
	{
		O op;
		//T orig = in_data[idx];
		op(in_data[idx], profile_data[col]);
		//printf("threadIdx = (%3d, %3d)\t blockIdx = (%3d, %3d)\tidx=%d col=%d\t %f - %f = %f\n",
		//		threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, idx, col,
		//		orig, profile_data[col], in_data[idx]);

	}
	return;
}

// Row-wise operations
template <typename T, typename O>
__global__ void d_op_1d2d_row(T* __restrict__ in_data, const T* __restrict__ profile_data, uint Nx, uint My)
{
	uint col = blockIdx.x * blockDim.x + threadIdx.x;
	uint row = blockIdx.y * blockDim.y + threadIdx.y;
	uint idx = row * Nx + col;

	if((col < Nx) && (row < My))
	{
		O op;
		op(in_data[idx], profile_data[row]);
	}
	return;
}


#endif


template <typename T>
class cuda_darray : public cuda_array<T>
{
    // typedef the parent: http://stackoverflow.com/questions/6592512/templates-parent-class-member-variables-not-visible-in-inherited-class
    typedef cuda_array<T> parent;
    // non-dependent member lookups http://www.parashift.com/c++-faq/nondependent-name-lookup-members.html
    using parent :: get_array_h;
    using parent :: get_array_d;
    using parent :: address;
    using parent :: copy_device_to_host;
    using parent :: copy_host_to_device;
    using parent :: get_grid;
    using parent :: get_block;
    using parent :: get_nx;
    using parent :: get_my;
    using parent :: check_bounds;
    
public:
    cuda_darray(uint, uint);
    cuda_darray(const cuda_darray<T>&);
    cuda_darray(cuda_darray<T>&&);
    cuda_darray(const parent&);
    cuda_darray(parent&&);
    ~cuda_darray();

    T& operator()(uint m, uint n) {return (parent::operator()(0, m, n));};
    T operator()(uint m, uint n) const {return parent::operator()(0, m, n);};

    cuda_darray operator=(const parent&);
    cuda_darray operator=(const cuda_darray<T>&);
    //cuda_darray operator=(cuda_darray<T>&&);
    cuda_darray operator=(const T& rhs)
    {
    	parent::operator=(rhs);
    	return(*this);
    }


    cuda_darray operator+= (const parent& rhs)
    {
		parent::operator+=(rhs);
		is_reduced_sum = false;
		is_reduced_max = false;
		is_reduced_min = false;
		is_reduced_profile = false;
    	return(*this);
    }
    cuda_darray operator+= (const T& rhs)
    {
        parent::operator+=(rhs);
		is_reduced_sum = false;
		is_reduced_max = false;
		is_reduced_min = false;
		is_reduced_profile = false;
    	return(*this);
    }
        

    cuda_darray operator+ (const parent& rhs) const
    {
    	cuda_darray<T> result(*this);
    	result += rhs;
    	return(result);

    }
    cuda_darray operator+ (const T& rhs) const
    {
        cuda_darray<T> result(*this);
        result += rhs;
        return(result);
    }


    cuda_darray operator-= (const cuda_array<T>& rhs)
    {
    	parent::operator-=(rhs);
    	is_reduced_sum = false;
    	is_reduced_max = false;
    	is_reduced_min = false;
    	is_reduced_profile = false;
    	return(*this);
    }
    cuda_darray operator-= (const T&);

    cuda_darray operator- (const parent& rhs) const
    {
        cuda_darray<T> result(*this);
        result -= rhs;
        return(result);
    }


    cuda_darray operator*= (const parent& rhs)
	{
    	parent::operator*=(rhs);
    	is_reduced_sum = false;
    	is_reduced_max = false;
    	is_reduced_min = false;
    	is_reduced_profile = false;
    	return(*this);
	}
    cuda_darray operator*= (const T&);

    cuda_darray operator* (const parent&) const;
    cuda_darray operator* (const cuda_darray<T>& rhs) const
	{
    	cuda_darray<T> result(*this);
    	result *= rhs;
    	return(result);
	}

    cuda_darray operator/= (const parent&);
    cuda_darray operator/= (const cuda_darray<T>& rhs)
	{
    	parent::operator/=(rhs);
    	is_reduced_sum = false;
    	is_reduced_max = false;
    	is_reduced_min = false;
    	is_reduced_profile = false;
    	return(*this);
	}
    cuda_darray operator/= (const T&);

    /// @brief Returns cuda_darray with mean value in y-direction subtracted
    cuda_darray<T> tilde() const;
    /// @brief Returns cuda_darray with fluctuations around mean profile subtracted
    cuda_darray<T> bar() const;
    /// @brief Returns mean value of an array
    cuda_darray<T> get_mean() const;

    // Compute profile, max, min, mean,... by reduction kernel
    /// @brief Returns the profile of the array
    void get_profile(T*);

    /// @brief Upcast the vector pointed to by T* along the columns
    void upcast_col(T*, const uint);

    /// @brief Upcast the vector pointed to by T along the rows
    void upcast_row(T*, const uint);


    /// @brief: take exp of entire field and subtract bg_level
    void remove_bg(const T& bg_level);
    /// @brief: add background level and take log of array
    void add_bg(const T& bg_level);

    /// @brief Returns maximum value of the array
    T get_max();
    /// @brief Returns maximum absolute value of the array
    T get_absmax();
    /// @brief Returns minimum value of the array
    T get_min();
    /// @brief Returns minimum absolute value of the array
    T get_absmin();
    /// @brief Returns sum of the array
    T get_sum();
    /// @brief Compute mean value of the array
    T get_mean();

    /// @brief copy profile to host
    void copy_profile_to_host(); 
    void copy_tmp_profile_to_host();
    /// @brief copy profile to device
    void copy_profile_to_device();
    /// @brief Compute array profile
    void print_profile() const;
    void print_tmp_profile() const;

    // Copy memory contents from parent, only
    void update(const parent&);

    /// @brief Return pointer to device data
    /// @detailed Clear all flags
    inline T* get_array_d() {
    	T* my_array_d_ptr = parent::get_array_d();
    	is_reduced_sum = false;
    	is_reduced_max = false;
    	is_reduced_min = false;
    	is_reduced_profile = false;
    	return (my_array_d_ptr);
    };
    /// @brief Return pointer to profile in device memory
    inline T* get_profile_d() const {return(d_profile);};
    inline T* get_tmp_profile_d() const {return(d_tmp_profile);};
    /// @brief Return pointer to profile in host memory
    inline T* get_profile_h() const {return(h_profile);};
    inline T get_profile_h(unsigned int n) const {return(h_profile[n]);};
    /// @brief Pointer to values from 0d reduction in device memory
    inline T* get_d_rval_ptr() const {return(d_rval_ptr);};

    inline dim3 get_blocksize_col() const {return(blocksize_col);};
    inline dim3 get_gridsize_col() const {return(gridsize_col);};
    inline dim3 get_blocksize_row() const {return(blocksize_row);};
    inline dim3 get_gridsize_row() const {return(gridsize_row);};

    inline size_t get_shmem_size_col() const {return(shmem_size_col);};
    inline size_t get_shmem_size_row() const {return(shmem_size_row);};

private:
    // Pointer to profile array (profile along x-direction)
    T* d_profile;
    // Pointer to temprary result from reduce function
    T* d_tmp_profile;
    T* h_profile;
    T* h_tmp_profile;

    // Result of 2d->0d reduction (max, min, ...), host
    T h_max;
    T h_min;
    T h_mean;
    T h_sum;
    // Result of 2d->0d reduction, device
    T* d_rval_ptr;


    // Block- and grid-size for column-wise reduction (to compute profiles)
    dim3 blocksize_col;
    dim3 gridsize_col;
    size_t shmem_size_col;

    // Block- and grid-size for row-wise reduction (max, min, sum, etc) -> Faster than column-wise reduction
    dim3 blocksize_row;
    dim3 gridsize_row;
    // shared memory size for row-based reduction
    size_t shmem_size_row;

    //Flag whether profile/max/min/sum are already computed
    bool is_reduced_profile;
    bool is_reduced_max;
    bool is_reduced_min;
    bool is_reduced_sum;

    void reduce_profile();
    void reduce_min();
    void reduce_max();
    void reduce_sum();

    // Reduce 2d array to 0d value, row-wise element access
    template <typename O> void reduce_row_2d0d();
    // 1d column-wise reduction of 2d data
    template <typename O> void reduce_col_2d1d();
    // 1d row-wise reduction of 2d array
    template <typename O> void reduce_row_2d1d();

    //Set array data to profile (implicit upcast of profile on 2nd array dimension)
    void upcast_profile();

    //Subtract profile from array data (implicit upcast of profile on 2nd array dimension)
    void subtract_profile();
};


#ifdef __CUDACC__

template <typename T>
cuda_darray<T> :: cuda_darray(uint my_, uint nx_) :
	cuda_array< T>(1, my_, nx_),
	d_profile(nullptr),
	d_tmp_profile(nullptr),
	h_profile(new T[nx_]),
	h_tmp_profile(new T[nx_]),
	h_max(0.0), h_min(0.0), h_mean(0.0), h_sum(0.0),
	d_rval_ptr(nullptr),
	blocksize_col(my_, 1, 1),
	gridsize_col(1, nx_, 1),
	shmem_size_col(my_ * sizeof(T)),
	blocksize_row(nx_, 1, 1),
	gridsize_row(1, my_, 1),
	shmem_size_row(nx_ * sizeof(T)),
	is_reduced_profile(false),
	is_reduced_max(false),
	is_reduced_min(false),
	is_reduced_sum(false)
{
    gpuErrchk(cudaMalloc((void**) &d_profile, get_nx() * sizeof(T)));
    gpuErrchk(cudaMalloc((void**) &d_tmp_profile, get_nx() * sizeof(T)));

    for(uint n = 0; n < get_nx(); n++)
        h_profile[n] = (T) 0.0;

    gpuErrchk(cudaMemcpy(d_profile, h_profile, get_nx() * sizeof(T), cudaMemcpyHostToDevice));

    // Allocate memory for max, sum, min on device and zero out
    double zero = 0.0;
    gpuErrchk(cudaMalloc(&d_rval_ptr, sizeof(T)));
    gpuErrchk(cudaMemcpy(d_rval_ptr, &zero, sizeof(T), cudaMemcpyHostToDevice));
}


template <typename T>
cuda_darray<T> :: cuda_darray(const parent& rhs) :
	cuda_darray<T>(rhs.get_nx(), rhs.get_my())
{
    cerr << "\tcuda_darray :: cuda_darray(const parent& rhs) " << endl;
	const size_t line_size = get_nx() * get_my() * sizeof(T);
	gpuErrchk(cudaMemcpy(get_array_d(), rhs.get_array_d(), line_size, cudaMemcpyDeviceToDevice));
}


// delegating constructor
// creates a new cuda_darray and copy data from RHS
// profile etc needs to be calculated again
template <typename T>
cuda_darray<T> :: cuda_darray(const cuda_darray<T>& rhs) :
	cuda_darray<T>(rhs.get_nx(), rhs.get_my())
{
	const size_t line_size = get_nx() * get_my() * sizeof(T);
	gpuErrchk(cudaMemcpy(get_array_d(), rhs.get_array_d(), line_size, cudaMemcpyDeviceToDevice));
}


template <typename T>
cuda_darray<T> :: cuda_darray(parent&& rhs) :
	// See http://stackoverflow.com/questions/15351341/move-constructors-and-inheritance
	cuda_array<T>(std::move(rhs)),
	d_profile(NULL),
	d_tmp_profile(NULL),
	h_profile(NULL),
	h_tmp_profile(NULL),
	h_max(0.0), h_min(0.0), h_mean(0.0), h_sum(0.0),
	d_rval_ptr(NULL),
	blocksize_col(get_my(), 1, 1),
	gridsize_col(1, get_nx(), 1),
	shmem_size_col(get_my() * sizeof(T)),
	blocksize_row(get_nx(), 1, 1),
	gridsize_row(1, get_my(), 1),
	shmem_size_row(get_nx() * sizeof(T)),
	is_reduced_profile(false),
	is_reduced_max(false),
	is_reduced_min(false),
	is_reduced_sum(false)
{
    //cerr << "\tcuda_darray :: cuda_darray(const parent&& rhs) " << endl;
    //cout << "\tblocksize = (" << get_block().x << ", " << get_block().y << ")" << endl;
    //cout << "\tgridsize = (" << get_grid().x << ", " << get_grid().y << ")" << endl;
	gpuErrchk(cudaMalloc((void**) &d_profile, get_nx() * sizeof(T)));
	gpuErrchk(cudaMalloc((void**) &d_tmp_profile, get_nx() * sizeof(T)));
	h_profile = new T[get_nx()];
	h_tmp_profile = new T[get_nx()];

	for(uint n = 0; n < get_nx(); n++)
		h_profile[n] = (T) n;

	gpuErrchk(cudaMemcpy(d_profile, h_profile, get_nx() * sizeof(T), cudaMemcpyHostToDevice));
	for(uint n = 0; n < get_nx(); n++)
		h_profile[n] = (T) 0.0;
	//print_profile();

	gpuErrchk(cudaMemcpy(h_profile, d_profile, get_nx() * sizeof(T), cudaMemcpyDeviceToHost));
	//print_profile();

	// Allocate memory for max, sum, min on device and zero out
	double zero = 0.0;
	gpuErrchk(cudaMalloc(&d_rval_ptr, sizeof(T)));
	gpuErrchk(cudaMemcpy(d_rval_ptr, &zero, sizeof(T), cudaMemcpyHostToDevice));
}


template <typename T>
cuda_darray<T> :: cuda_darray(cuda_darray<T>&& rhs) :
// Move constructor, do no allocate any memory at all
// See http://stackoverflow.com/questions/15351341/move-constructors-and-inheritance
	cuda_array<T>(std::move(rhs))
{
    // Pointer to profile array (profile along x-direction)
    d_profile = rhs.d_profile;
    rhs.d_profile = nullptr;

    // Pointer to temprary result from reduce function
    d_tmp_profile = rhs.d_tmp_profile;
    rhs.d_tmp_profile = nullptr;

    h_profile = rhs.h_profile;
    rhs.h_profile = nullptr;

    h_tmp_profile = rhs.h_tmp_profile;
    rhs.h_tmp_profile = nullptr;

    h_max = rhs.h_max;
    rhs.h_max = -1.0;
    h_min = rhs.h_min;
    rhs.h_min = -1.0;
    h_mean = rhs.h_mean;
    rhs.h_mean = -1.0;
    h_sum = rhs.h_sum;
    rhs.h_sum = -1.0;

    d_rval_ptr = rhs.d_rval_ptr;
    rhs.d_rval_ptr = nullptr;

    blocksize_col = rhs.blocksize_col;
    gridsize_col = rhs.gridsize_col;
    shmem_size_col = rhs.shmem_size_col;

    blocksize_row = rhs.blocksize_row;
    gridsize_row = rhs.gridsize_row;
    shmem_size_row = rhs.shmem_size_row;

    is_reduced_profile = rhs.is_reduced_profile;
    is_reduced_max = rhs.is_reduced_max;
    is_reduced_min = rhs.is_reduced_min;
    is_reduced_sum = rhs.is_reduced_sum;
}



template <typename T>
cuda_darray<T> :: ~cuda_darray()
{
    gpuErrchk(cudaFree(d_rval_ptr));
    gpuErrchk(cudaFree(d_tmp_profile));
    gpuErrchk(cudaFree(d_profile));
    delete [] h_profile;
}


template <typename T>
cuda_darray<T> cuda_darray<T> :: operator=(const parent& rhs)
{
    if ((void*) this == (void*) &rhs)
        return*this;
    check_bounds(rhs.get_my(), rhs.get_nx());
    gpuErrchk(cudaMemcpy(get_array_d(), rhs.get_array_d(), sizeof(T) * get_my() * get_nx(), cudaMemcpyDeviceToHost));
    is_reduced_profile = false;
    is_reduced_max = false;
    is_reduced_min = false;
    is_reduced_sum = false;
    return(*this);
}


template <typename T>
cuda_darray<T> cuda_darray<T> :: operator=(const cuda_darray<T>& rhs)
{
   if ((void*) this == (void*) &rhs)
        return*this;

    check_bounds(rhs.get_my(), rhs.get_nx());
    gpuErrchk(cudaMemcpy(get_array_d(), rhs.get_array_d(), sizeof(T) * get_my() * get_nx(), cudaMemcpyDeviceToHost));
    is_reduced_profile = false;
    is_reduced_max = false;
    is_reduced_min = false;
    is_reduced_sum = false;
    return(*this);
}

template <typename T>
template <typename O>
void cuda_darray<T> :: reduce_col_2d1d()
{
	//column-wise reduction (radial profile)
	d_reduce<T, O> <<<get_gridsize_col(), get_blocksize_col(), get_shmem_size_col()>>>(get_array_d(), get_profile_d(), get_nx(), 1, get_nx(), get_my());
}


template <typename T>
template <typename O>
void cuda_darray<T> :: reduce_row_2d1d()
{
	// row-wise reduction (poloidal profile)
	d_reduce<T, O> <<<get_gridsize_row(), get_blocksize_row(), get_shmem_size_row()>>>(get_array_d(), get_profile_d(), 1, get_nx(), get_nx(), get_my());
}



// Perform a 2d->1d->0d reduction, store result in h_rval
template <typename T>
template <typename O>
void cuda_darray<T> :: reduce_row_2d0d()
{
	d_reduce<T, O> <<<get_gridsize_row(), get_blocksize_row(), get_shmem_size_row()>>>(get_array_d(), get_tmp_profile_d(), 1, get_nx(), get_nx(), get_my());
	d_reduce<T, O> <<<1, get_nx(), get_shmem_size_row()>>>(get_tmp_profile_d(), get_d_rval_ptr(), 1, get_nx(), get_nx(), 1);
}


template <typename T>
void cuda_darray<T> :: copy_profile_to_host()
{
    const size_t line_size = get_nx() * sizeof(T);
    gpuErrchk(cudaMemcpy(h_profile, d_profile, line_size, cudaMemcpyDeviceToHost));
}


template <typename T>
void cuda_darray<T> :: copy_tmp_profile_to_host()
{
	const size_t line_size = get_nx() * sizeof(T);
	gpuErrchk(cudaMemcpy(h_tmp_profile, d_tmp_profile, line_size, cudaMemcpyDeviceToHost));
}


template <typename T>
void cuda_darray<T> :: copy_profile_to_device()
{
	const size_t line_size = get_nx() * sizeof(T);
	gpuErrchk(cudaMemcpy(d_profile, h_profile, line_size, cudaMemcpyHostToDevice));
}

template <typename T>
void cuda_darray<T> :: print_profile() const
{
    for(uint n = 0; n < get_nx(); n++)
        cout << h_profile[n] << ", ";
    cout << "\n";
}


template <typename T>
void cuda_darray<T> :: print_tmp_profile() const
{
    for(uint n = 0; n < get_nx(); n++)
        cout << h_tmp_profile[n] << ", ";
    cout << "\n";
}

// Reduce profile
template <typename T>
void cuda_darray<T> :: get_profile(T* out_profile)
{
	reduce_profile();
	for(unsigned int n = 0; n < get_nx(); n++)
		out_profile[n] = get_profile_h(n);
}


template <typename T>
void cuda_darray<T> :: reduce_profile()
{
	if(!is_reduced_profile)
	{
		// Reduce column-wise for radial profile
		reduce_col_2d1d<d_op1_addassign<T> >();
		// Reduce row-wise for poloidal profile. DO NOT DO THIS. JUST FOR TESTING :)
		// reduce_row_2d1d<d_op2_add<T> >();
		// Multiply result by number of elements
		d_op1_scalar<T, d_op1_mulassign<T> ><<<1, get_nx()>>>(get_profile_d(), T(1. / get_my()), (uint) 1, get_nx());
	    const size_t line_size = get_nx() * sizeof(T);
	    gpuErrchk(cudaMemcpy(h_profile, d_profile, line_size, cudaMemcpyDeviceToHost));
	    is_reduced_profile = true;
	}
}

// Set array data to in_data, column-wise
template <typename T>
void cuda_darray<T> :: upcast_col(T* in_data, const uint Nx_in)
{
	check_bounds(1, Nx_in);
	const size_t line_size = Nx_in * sizeof(T);

	T* d_in_data;
	gpuErrchk(cudaMalloc((void**) &d_in_data, line_size));
	gpuErrchk(cudaMemcpy(d_in_data, in_data, line_size, cudaMemcpyHostToDevice));

	d_op_1d2d_col<T, d_op1_assign<T> ><<<get_grid(), get_block()>>>(get_array_d(), d_in_data, get_nx(), get_my());
	if(d_in_data != nullptr)
		gpuErrchk(cudaFree(d_in_data));
}

// Set array data to in_data, row-wise
template <typename T>
void cuda_darray<T> :: upcast_row(T* in_data, const uint My_in)
{
	check_bounds(My_in, 1);
	const size_t line_size = My_in * sizeof(T);

	T* d_in_data;
	gpuErrchk(cudaMalloc((void**) &d_in_data, line_size));
	gpuErrchk(cudaMemcpy(d_in_data, in_data, line_size, cudaMemcpyHostToDevice));

	d_op_1d2d_row<T, d_op1_assign<T> ><<<get_grid(), get_block()>>>(get_array_d(), d_in_data, get_nx(), get_my());
	if(d_in_data != nullptr)
		gpuErrchk(cudaFree(d_in_data));
}




// Compute maximum
template <typename T>
T cuda_darray<T> :: get_max()
{
	if(!is_reduced_max)
	{
		reduce_row_2d0d<d_op1_maxassign<T> >();
		gpuErrchk(cudaMemcpy(&h_max, d_rval_ptr, sizeof(T), cudaMemcpyDeviceToHost))
		is_reduced_max = true;
	}

	return (h_max);
}


// Compute absolute maximum
template <typename T>
T cuda_darray<T> :: get_absmax()
{
	if(!is_reduced_max)
	{
		reduce_row_2d0d<d_op1_absmaxassign<T> >();
		gpuErrchk(cudaMemcpy(&h_max, d_rval_ptr, sizeof(T), cudaMemcpyDeviceToHost))
		is_reduced_max = true;
	}

	return (h_max);
}

// Compute minimum
template <typename T>
T cuda_darray<T> :: get_min()
{
	if(!is_reduced_min)
	{
		reduce_row_2d0d<d_op1_minassign<T> >();
		gpuErrchk(cudaMemcpy(&h_min, d_rval_ptr, sizeof(T), cudaMemcpyDeviceToHost))
		is_reduced_min = true;
	}
	return (h_min);
}


// Compute absolute minimum
template <typename T>
T cuda_darray<T> :: get_absmin()
{
	if(!is_reduced_min)
	{
		reduce_row_2d0d<d_op1_absminassign<T> >();
		gpuErrchk(cudaMemcpy(&h_min, d_rval_ptr, sizeof(T), cudaMemcpyDeviceToHost))
		is_reduced_min = true;
	}
	return (h_min);
}


// Compute array sum
template <typename T>
T cuda_darray<T> :: get_sum()
{
	if(!is_reduced_sum)
	{
		reduce_row_2d0d<d_op1_addassign<T> >();
		gpuErrchk(cudaMemcpy(&h_sum, d_rval_ptr, sizeof(T), cudaMemcpyDeviceToHost))
		is_reduced_min = true;
	}
	return (h_sum);
}


// Compute mean of the array
template <typename T>
T cuda_darray<T> :: get_mean()
{
	get_sum();
	return(h_sum / T(get_nx() * get_my()));
}


// Set array data to profile
template <typename T>
void cuda_darray<T> :: upcast_profile()
{
	reduce_profile();
    d_op_1d2d_col<T, d_op1_assign<T> ><<<get_grid(), get_block()>>>(get_array_d(), get_profile_d(), get_nx(), get_my());
    is_reduced_profile = false;
    is_reduced_max = false;
    is_reduced_min = false;
    is_reduced_sum = false;
}


// Subtract profile from array data
template <typename T>
void cuda_darray<T> :: subtract_profile()
{
	reduce_profile();
	d_op_1d2d_col<T, d_op1_subassign<T> ><<<get_grid(), get_block()>>>(get_array_d(), get_profile_d(), get_nx(), get_my());
    is_reduced_profile = false;
    is_reduced_max = false;
    is_reduced_min = false;
    is_reduced_sum = false;
}

// Return fluctuating array
template <typename T>
cuda_darray<T> cuda_darray<T> :: tilde() const
{
	// Compute profile
	// Create a new cuda_darray, subtract profile from new cuda_darray
	cuda_darray<T> result(*this);


	// Subtract profile of new array from new array data
	result.reduce_profile();
	d_op_1d2d_col<T, d_op1_subassign<T> ><<<get_grid(), get_block()>>>(result.get_array_d(), result.get_profile_d(), get_nx(), get_my());
	// Unset profile of new array.
	result.is_reduced_profile = false;
	result.reduce_profile();

	// Check that the profile of the tilde array is small
	T profile_sum{0.0};
	for(uint n = 0; n < get_nx(); n++)
		profile_sum += result.h_profile[n];
	assert(profile_sum <= cuda::epsilon);
	return(result);
}


// Return mean profile
template <typename T>
cuda_darray<T> cuda_darray<T> :: bar() const
{
    cuda_darray<T> result(*this);
    result.reduce_profile();
	d_op_1d2d_col<T, d_op1_assign<T> ><<<get_grid(), get_block()>>>(result.get_array_d(), result.get_profile_d(), get_nx(), get_my());
	// Unset profile of new array.
	result.is_reduced_profile = false;
	result.reduce_profile();

    return(result);
}



// Remove background from theta field for logarithmic blob simulations
template <typename T>
void cuda_darray<T> :: remove_bg(const T& bg_level)
{
	// Take exp of entire array
	this -> template op_apply_t<d_op0_expassign<T> >(0);
	this -> template op_scalar_t<d_op1_subassign<T> >(bg_level, 0);
	copy_device_to_host();
	is_reduced_profile = false;
	is_reduced_max = false;
	is_reduced_min = false;
	is_reduced_sum = false;
}


// Add background from logarithmic field
template <typename T>
void cuda_darray<T> :: add_bg(const T& bg_level)
{
	this -> template op_scalar_t<d_op1_addassign<T> >(bg_level, 0);
	this -> template op_apply_t<d_op0_logassign<T> >(0);
	copy_device_to_host();
	is_reduced_profile = false;
	is_reduced_max = false;
	is_reduced_min = false;
	is_reduced_sum = false;
}

#endif // __CUDACC__

#endif // CUDA_DARRAY_H

