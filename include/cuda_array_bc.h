/*
 * cuda_array_bc.h
 *
 *  Created on: May 13, 2016
 *      Author: ralph
 *
 * Array type class that implements Neumann and Dirichlet boundary conditions in the
 * x direction and periodic BC in the y direction.
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
#include <cufft.h>
#include "bounds.h"
#include "error.h"
#include "cuda_types.h"
using namespace std;

typedef unsigned int uint;
typedef double real_t;

enum class bc_t {bc_dirichlet, bc_neumann, bc_periodic};

//constexpr uint num_gp_x{4};
//constexpr uint cuda::gp_offset_x{2};
//constexpr uint num_gp_y{4};
//constexpr uint cuda::gp_offset_y{2};
//
//constexpr uint cuda_cuda::blockdim_col{4};
//constexpr uint cuda_cuda::blockdim_row{1};
//
//
///// Datatype that defines the type of boundary condition on all borders
///// and gives the value of the boundary condition
//template <typename T>
//struct cuda::bvals
//{
//	// The boundary conditions on the domain border
//	bc_t bc_left;
//	bc_t bc_right;
//	bc_t bc_top;
//	bc_t bc_bottom;
//
//	// The boundary values on the domain border
//	T bval_left;
//	T bval_right;
//	T bval_top;
//	T bval_bottom;
//};
//
//
//const std::map<cufftResult, std::string> cufftGetErrorString
//{
//    {CUFFT_SUCCESS, std::string("CUFFT_SUCCESS")},
//    {CUFFT_INVALID_PLAN, std::string("CUFFT_INVALID_PLAN")},
//    {CUFFT_ALLOC_FAILED, std::string("CUFFT_ALLOC_FAILED")},
//    {CUFFT_INVALID_TYPE, std::string("CUFFT_INVALID_TYPE")},
//    {CUFFT_INVALID_VALUE, std::string("CUFFT_INVALID_VALUE")},
//    {CUFFT_INTERNAL_ERROR, std::string("CUFFT_INTERNAL_ERROR")},
//    {CUFFT_EXEC_FAILED, std::string("CUFFT_EXEC_FAILED")},
//    {CUFFT_SETUP_FAILED, std::string("CUFFT_SETUP_FAILED")},
//    {CUFFT_INVALID_SIZE, std::string("CUFFT_INVALID_SIZE")},
//    {CUFFT_UNALIGNED_DATA, std::string("CUFFT_UNALIGNED_DATA")}
//};


class slab_layout_t
{
public:
    // Provide standard ctor for pre-C++11
    slab_layout_t(real_t xl, real_t dx, real_t yl, real_t dy, real_t dt, unsigned int my, unsigned int nx) :
        x_left(xl), delta_x(dx), y_lo(yl), delta_y(dy), delta_t(dt), My(my), Nx(nx) {};
    const real_t x_left;
    const real_t delta_x;
    const real_t y_lo;
    const real_t delta_y;
    const real_t delta_t;
    const unsigned int My;
    const unsigned int Nx;

    friend std::ostream& operator<<(std::ostream& os, const slab_layout_t s)
    {
        os << "x_left = " << s.x_left << "\t";
        os << "delta_x = " << s.delta_x << "\t";
        os << "y_lo = " << s.y_lo << "\t";
        os << "delta_y = " << s.delta_y << "\t";
        os << "delta_t = " << s.delta_t << "\t";
        os << "My = " << s.My << "\n";
        os << "Nx = " << s.Nx << "\t";
        return os;
    }
} __attribute__ ((aligned (8)));


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

/// Device function to compute column and row, taking ghost points into consideration
/// Use offset = gp_offset_[x,y] for not accessing ghost points in kernels launched on grid,block
/// Use offset = 0 for accessing the entire array
__device__ inline int d_get_col(const int offset) {
    return (blockIdx.x * blockDim.x + threadIdx.x + offset);
}


__device__ inline int d_get_row(const int offset) {
    return (blockIdx.y * blockDim.y + threadIdx.y + offset);
}

/// Test if the column and row are ghostpoints or not
/// Returns true if the index pair (col, row) refer to a cellcenter(not a gp)
/// Returns false if the index pair (col, row) refer to a ghost points
__device__ inline bool d_is_cellcenter(const int col, const int row, const int My, const int Nx)
{
	return((col > cuda::gp_offset_y - 1) && 
		   (col < My + cuda::gp_offset_y) &&
		   (row > cuda::gp_offset_x - 1) && 
		   (row < Nx + cuda::gp_offset_x)); 
}


template <typename T>
__global__
void d_alloc_array_d_t(T** array_d_t, T* array, const uint tlevs, const uint My, const uint Nx)
{
    for(uint t = 0; t < tlevs; t++)
    {
        array_d_t[t] = &array[t * (Nx + cuda::num_gp_x) * (My + cuda::num_gp_y)];
    }
}


template <typename T>
__global__
void d_enumerate(T** array_d_t, const uint t, const uint My, const uint Nx)
{
	const int col = d_get_col(cuda::gp_offset_y);
    const int row = d_get_row(cuda::gp_offset_x);
    const int index = row * (My + cuda::num_gp_y) + col;

	if (d_is_cellcenter(col, row, My, Nx))
        array_d_t[t][index] = 10000 * T(t) + T(index);
	
}


template <typename T>
__global__
void d_set_to_zero(T** array_d_t, const uint tlev, const uint My, const uint Nx)
{
	const int col = d_get_col(0);
	const int row = d_get_row(0);
	const int index = row * (My + cuda::num_gp_x) + col;
	
	if (index < (Nx + cuda::num_gp_x) * (My + cuda::num_gp_y))
		array_d_t[tlev][index] = T(0.0);
}
	

template <typename T, typename O>
__global__
void d_evaluate(T** array_d_t, O d_op_fun,  const slab_layout_t geom, const uint tlev)
{
    const int col = d_get_col(cuda::gp_offset_y);
    const int row = d_get_row(cuda::gp_offset_x);
    const int index = row * (geom.My + cuda::num_gp_y) + col;

    if (d_is_cellcenter(col, row, geom.My, geom.Nx))
		array_d_t[tlev][index] = d_op_fun(d_get_col(0), d_get_row(0), geom);
}


template <typename T, typename O>
__global__
void d_evaluate_2(T** array_d_t, O d_op_fun, const slab_layout_t geom, const uint tlev)
{
	const int col = d_get_col(cuda::gp_offset_y);
    const int row = d_get_row(cuda::gp_offset_x);
    const int index = row * (geom.My + cuda::num_gp_y) + col;

    if (d_is_cellcenter(col, row, geom.My, geom.Nx))
		array_d_t[tlev][index] = d_op_fun(array_d_t[tlev][index], d_get_col(0), d_get_row(0), geom);
	
}

/// Update left ghost points for Dirichlet boundary conditions, located in column = cuda::gp_offset_x
///
/// U_1/2 = 1/2(u(0, j) + u(1,j) => u(0,j) = 2 U_1/2 - u(1,j)
/// u[(cuda::gp_offset_x - 1) + (My + cuda::num_gp_y) + m] = 2 U_1/2 - u[cuda::gp_offset_x * (My + cuda::num_gp_y) + m]
/// m = cuda::gp_offset_y ... My + cuda::gp_offset_y - 1

template <typename T>
__global__
void d_update_gp_x_dirichlet_left(T* array, T bval_left, const uint My, const uint Nx)
{
	const int col = d_get_col(cuda::gp_offset_y);
	const int row = cuda::gp_offset_x - 1; // Subtract 1 for zero based indexing
	const int index = row * (My + cuda::num_gp_y) + col;	
	
	if((col + 1 > cuda::gp_offset_y) && (col < My + cuda::gp_offset_y))
		array[index] = 2.0 * bval_left - array[(row + 1) * (My + cuda::num_gp_y) + col];
}

/// Update left ghost points for Neumann boundary conditions, located in  column = cuda::gp_offset_x
///
/// U'_1/2 = 1/2(u(0, j) + u(1,j) => u(0,j) = 2 U_1/2 - u(1,j)
/// u[(cuda::gp_offset_x - 1) + (My + cuda::num_gp_y) + m] = 2 U_1/2 - u[cuda::gp_offset_x * (My + cuda::num_gp_y) + m]
/// m = cuda::gp_offset_y ... My + cuda::gp_offset_y - 1
template <typename T>
__global__
void d_update_gp_x_neumann_left(T* array, T bval_left, T deltax, const uint My, const uint Nx)
{
	const int col = d_get_col(cuda::gp_offset_y);
	const int row = cuda::gp_offset_x - 1;
	const int index = row * (My + cuda::num_gp_y) + col;
	
	if((col + 1 > cuda::gp_offset_y) && (col < My + cuda::gp_offset_y))
		array[index] = -1.0 * bval_left * deltax + array[(row + 1) * (My + cuda::num_gp_y) + col];
}


/// Update right ghost points, Dirichlet boundary conditions, locaed in column (cuda::gp_offset_x + My + 1) 
template <typename T>
__global__
void d_update_gp_x_dirichlet_right(T* array, T bval_right, const uint My, const uint Nx)
{
	const int col = d_get_col(cuda::gp_offset_y);
	const int row = Nx + cuda::gp_offset_x;
	const int index = row * (My + cuda::num_gp_y) + col;
	
	//printf("kernel: row = %d, col = %d, index = %d\n", row, col, index);
	if((col + 1 > cuda::gp_offset_y) && (col < My + cuda::gp_offset_y))	
		array[index] =  2.0 * bval_right - array[(row - 1) * (My + cuda::num_gp_y) + col];
}

template <typename T>
__global__
void d_update_gp_x_neumann_right(T* array, T bval_right, T deltax, const uint My, const uint Nx)
{
	const int col = d_get_col(cuda::gp_offset_y);
	const int row = Nx + cuda::gp_offset_x;
	const int index = row * (My + cuda::num_gp_y) + col;
	
	//printf("kernel: row = %d, col = %d, index = %d\n", row, col, index);
	if((col + 1 > cuda::gp_offset_y) && (col < My + cuda::gp_offset_y))	
		array[index] = bval_right * deltax + array[(row - 1) * (My + cuda::num_gp_y) + col];
}


template <typename T>
__global__
void d_update_gp_y_periodic_bottom(T* array, const uint My, const uint Nx)
{
	const int col = cuda::gp_offset_y - 1;
	// Don't use an offset here! Instead check for row > cuda::gp_offset_x - 2 to access the corner elements.
	const int row = d_get_row(0);
	const int index = row * (My + cuda::num_gp_y) + col;
	//printf("kernel: row = %d, col = %d, index = %d\n", row, col, index);

	if((row > cuda::gp_offset_x - 2) && (row < My + cuda::gp_offset_x + 1))
		array[index] = array[row * (My + cuda::num_gp_y) + col + My];
}


template <typename T>
__global__
void d_update_gp_y_periodic_top(T* array, const uint My, const uint Nx)
{
	const int col = My + cuda::gp_offset_y;
	// Don't use an offset here! Instead check for row > cuda::gp_offset_x - 2 to access the corner elements.
	const int row = d_get_row(0);
	const int index = row * (My + cuda::num_gp_y) + col;
	
	if((row > cuda::gp_offset_x - 2) && (row < My + cuda::gp_offset_y + 1))
		array[index] = array[row * (My + cuda::num_gp_y) + col - My];	
}

#endif // __CUDACC_

template <typename T>
class cuda_array_bc{
public:
	cuda_array_bc(uint, uint, uint, cuda::bvals<T>, slab_layout_t);
	~cuda_array_bc();

	void evaluate(function<T (uint, uint)>, uint);
    void evaluate_device(uint);
	void enumerate();

	inline T& operator() (uint n, uint m) {return(*(array_h + address(n, m)));};
	inline T operator() (uint n, uint m) const {return(*(array_h + address(n, m)));}

    inline T& operator() (uint t, uint n, uint m) {
        //cout << "t = " << t << ", n = " << n << ", m = " << m << ", address = " << (array_h_t[t] + address(n, m)) << endl;
        return(*(array_h_t[t] + address(n, m)));
    };
    inline T operator() (uint t, uint n, uint m) const {
        //cout << "t = " << t << ", n = " << n << ", m = " << m << ", address = " << (array_h_t[t] + address(n, m)) << endl;
        return(*(array_h_t[t] + address(n, m)));
    };

    // Copy device memory to host and print to stdout
    friend std::ostream& operator<<(std::ostream& os, cuda_array_bc& src)
    {
        const uint tl = src.get_tlevs();
        const uint my = src.get_my();
        const uint nx = src.get_nx();
        os << "\n";
        for(uint t = 0; t < tl; t++)
        {
            for(uint n = 0; n < nx; n++)
            {
                for(uint m = 0; m < my; m++)
                {
                    // Remember to also set precision routines in CuCmplx :: operator<<
                    //os << std::setw(cuda::io_w) << std::setprecision(cuda::io_p) << src(t, m, n) << "\t";
                	os << std::setw(8) << std::setprecision(5) << src(n, m) << "\t";
                }
            os << endl;
            }
            os << endl;
        }
        return (os);
    }
	
	
	/// Update ghost points
	void update_ghost_points(uint);

    void dump_full() const;
	
	/// Initialize DFT for in-place transformations along y
	void init_dft();
	void dft_r2c(const uint);
	void dft_c2r(const uint);
	void normalize(const uint);
	
	// Copy entire device data to host
	void copy_device_to_host();
	// Copy entire device data to external data pointer in host memory
	//void copy_device_to_host(T*);
	// Copy device to host at specified time level
	//void copy_device_to_host(uint);

	// Copy deice data at specified time level to external pointer in device memory
	void copy_device_to_device(const uint, T*);

	// Transfer from host to device
	void copy_host_to_device();
	//void copy_host_to_device(uint);
	//void copy_host_to_device(T*);

	// Advance time levels
	//void advance();

	///@brief Copy data from t_src to t_dst
	//void copy(uint t_dst, uint t_src);
	///@brief Copy data from src, t_src to t_dst
	//void copy(uint t_dst, const cuda_array<T>& src, uint t_src);
	///@brief Move data from t_src to t_dst, zero out t_src
	//void move(uint t_dst, uint t_src);
	///@brief swap data in t1, t2
	//void swap(uint t1, uint t2);
	//void normalize();

	//void kill_kx0();
	//void kill_ky0();
	//void kill_k0();

	// Access to private members
	inline uint get_nx() const {return(Nx);};
	inline uint get_my() const {return(My);};
	inline uint get_tlevs() const {return(tlevs);};
	inline uint address(uint n, uint m) const {
        uint retval = (n + cuda::gp_offset_x) * (My + cuda::num_gp_y) + m + cuda::gp_offset_y; 
        //cout << "\taddress: n  = " << n << ", m = " << m << ", address = " << retval << endl;
        return(retval);
    };
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
	inline void check_bounds(uint t, uint n, uint m) const {array_bounds(t, n, m);};
	inline void check_bounds(uint n, uint m) const {array_bounds(n, m);};

	// Number of elements
	inline size_t get_nelem_per_t() const {return ((Nx + cuda::num_gp_x) * (My + cuda::num_gp_y));};

private:
	uint tlevs;
	uint Nx;
	uint My;
	cuda::bvals<T> boundaries;
	slab_layout_t geom;
	bounds array_bounds;

	// block and grid for access without ghost points, use these normally
	dim3 block;
	dim3 grid;
	
	// block and grid access to entire grid, with ghost points
	dim3 grid_gp;

	// Array data is on device
	// Pointer to device data
	T* array_d;
	// Pointer to each time stage. Pointer to array of pointers on device
	T** array_d_t;
	// Pointer to each time stage: Pointer to each time level on host
	T** array_d_t_host;
	T* array_h;
	T** array_h_t;	
	
	// CuFFT related stuff
	cufftHandle plan_fw;
	cufftHandle plan_bw;
};


const map<bc_t, string> bc_str_map
{
	{bc_t::bc_dirichlet, "Dirichlet"},
	{bc_t::bc_neumann, "Neumann"},
	{bc_t::bc_periodic, "Periodic"}
};

template <typename T>
cuda_array_bc<T> :: cuda_array_bc(uint _tlevs, uint _Nx, uint _My, cuda::bvals<T> bvals, slab_layout_t _geom) :
		tlevs(_tlevs), Nx(_Nx), My(_My), boundaries(bvals), geom(_geom), array_bounds(tlevs, Nx, My),
		block(dim3(cuda::blockdim_col, cuda::blockdim_row)),
		grid(dim3(Nx, (My + (cuda::blockdim_row - 1)) / cuda::blockdim_row)),
		grid_gp(dim3(Nx + cuda::num_gp_x, (My + cuda::num_gp_y + (cuda::blockdim_row - 1) / cuda::blockdim_row))),
		array_d(nullptr),
		array_d_t(nullptr),
		array_d_t_host(new T*[tlevs]),
		array_h(new T[tlevs * (Nx + cuda::num_gp_x) * (My + cuda::num_gp_y)]),
		array_h_t(new T*[tlevs])
{
	gpuErrchk(cudaMalloc( (void***) &array_d_t, tlevs * sizeof(T*)));
	gpuErrchk(cudaMalloc( (void**) &array_d, tlevs * get_nelem_per_t() * sizeof(T)));

	d_alloc_array_d_t<<<1, 1>>>(array_d_t, array_d, tlevs, Nx, My);
	gpuErrchk(cudaMemcpy(array_d_t_host, array_d_t, sizeof(T*) * tlevs, cudaMemcpyDeviceToHost));

	for(uint t = 0; t < tlevs; t++)
    {
		d_set_to_zero<<<block, grid_gp>>>(array_d_t, t,  Nx, My);
        array_h_t[t] = &array_h[t * get_nelem_per_t()];
    }

    cout << "array_h at " << array_h << endl;
    for(uint t = 0; t < tlevs; t++)
        cout << "array_h_t[" << t << "] at " << array_h_t[t] << endl;
}

template <typename T>
void cuda_array_bc<T> :: evaluate(function<T (unsigned int, unsigned int)> op_fun, uint tlev)
{
	cout << "evaluating..." << endl;
	for(int n = 0; n < Nx; n++)
		for (int m = 0; m < My; m++)
			(*this)(tlev, n, m) = op_fun(n, m);
}


/// Lambda function capture[=] does not capture Nx, My passed as parameters to the kernel.
/// Solution: pass it to the lambda itself, last two arguments
template <typename T>
void cuda_array_bc<T> :: evaluate_device(uint tlev)
{
    cout << "evaluate_device..." << endl;
	//d_evaluate<<<block, grid>>>([=] __device__ (uint n, uint m, uint Nx, uint My) -> T {return(T(1.2) );}, array_d_t, My, Nx);
	d_evaluate<<<block, grid>>>(get_array_d_t(), [=] __device__ (uint m, uint n, slab_layout_t geom) -> T 
		{
			T x{geom.x_left + (T(n) + 0.5) * geom.delta_x};
			T y{geom.y_lo + (T(m) + 0.5) * geom.delta_y};
			return(sin(2.0 * M_PI * x) * sin(2.0 * M_PI * y));
		}, geom, 0);
}


template <typename T>
void cuda_array_bc<T> :: dump_full() const
{
	for(uint t = 0; t < tlevs; t++)
	{
        cout << "dump_full: t = " << t << endl;
		for (uint n = 0; n < Nx + cuda::num_gp_x; n++)
		{
			for(uint m = 0; m < My + cuda::num_gp_y; m++)
			{
				//cout << (*this)(t, n, m) << "\t";
                cout << std::setw(7) << std::setprecision(5) << *(array_h_t[t] + n * (My + cuda::num_gp_y) + m) << "\t";
			}
			cout << endl;
		}
        cout << endl << endl;
	}
}

template <typename T>
void cuda_array_bc<T> :: init_dft()
{
	cufftResult err;
	int dft_size[1] = {int(My)};
	int dft_onembed[1] = {int(My / 2 + 1)};
	err = cufftPlanMany(&plan_fw, 
		1, //int rank
		dft_size, //int* n
		dft_size, //int* inembed
		1, //int istride
		My + cuda::num_gp_y, //int idist
		dft_onembed, //int* onembed
		1, // int ostride
		My / 2 + 1 + cuda::gp_offset_y / 2, //int odist
		CUFFT_D2Z, //cufftType type
		Nx); //int batch
	
	if (err != 0)
	{
		stringstream err_str;
        err_str << "Error planning D2Z DFT: " << err << "\n";
        throw gpu_error(err_str.str());
	}
	//cufftResult cufftPlanMany(cufftHandle *plan, int rank, int *n, int *inembed,
    //int istride, int idist, int *onembed, int ostride,
    //int odist, cufftType type, int batch);
	
	err = cufftPlanMany(&plan_bw,
		1, //int rank
		dft_size, //int* n
		dft_onembed, //int* inembed
		1, //int istride
		My / 2 + 1 + cuda::gp_offset_y / 2, //int idist
		dft_size, //int* onembed
		1, //int ostride
		My + cuda::num_gp_y, //int odist
		CUFFT_Z2D, //cufftType type
		Nx); //int batch
		
	if(err != 0)
	{
		stringstream err_str;
        err_str << "Error planning Z2D DFT: " << err << "\n";
        throw gpu_error(err_str.str());
	}
}


/// Perform DFT in y-direction, row-wise
template <typename T>
void cuda_array_bc<T> :: dft_r2c(const uint tlev)
{
	const int offset = cuda::gp_offset_x * (My + cuda::num_gp_y) + cuda::gp_offset_y; 
	cufftResult err;
	
	/* Use the CUFFT plan to transform the signal in place. */
	err = cufftExecD2Z(plan_fw, 
	                 get_array_d(tlev) + offset, 
					 (cufftDoubleComplex*) get_array_d(tlev) + offset / 2);
	if(err != CUFFT_SUCCESS)
    {
        stringstream err_str;
        err_str << "Error planning D2Z DFT: " << cufftGetErrorString.at(err) << endl;
        throw gpu_error(err_str.str());
    }
}


/// Perform iDFT in y-direction, row-wise
template <typename T>
void cuda_array_bc<T> :: dft_c2r(const uint tlev)
{
	const int offset = cuda::gp_offset_x * (My + cuda::num_gp_y) + cuda::gp_offset_y; 
	cufftResult err;

	/* Use the CUFFT plan to transform the signal in place. */
	err = cufftExecZ2D(plan_bw,
					 (cufftDoubleComplex*) get_array_d(tlev) + offset / 2,
					 get_array_d(tlev) + offset);
	if(err != CUFFT_SUCCESS)
    {
        stringstream err_str;
        err_str << "Error planning Z2D DFT: " << cufftGetErrorString.at(err) << endl;
        throw gpu_error(err_str.str());
    }
}


template <typename T>
void cuda_array_bc<T> :: normalize(const uint tlev)
{
	d_evaluate_2<<<block, grid>>>(get_array_d_t(), [=] __device__ (T in, uint m, uint n, slab_layout_t geom) -> T 
	{
		return(in / T(geom.My));
	}, geom, 0);
}


template <typename T>
void cuda_array_bc<T>::copy_device_to_host()
{
    for(uint t = 0; t < tlevs; t++)
	{
		cout << t << ", " << get_array_d(t) << endl;
        gpuErrchk(cudaMemcpy(&array_h[t * get_nelem_per_t()], get_array_d(t), sizeof(T) * get_nelem_per_t(), cudaMemcpyDeviceToHost));	
	}
}


template <typename T>
void cuda_array_bc<T> :: copy_host_to_device()
{ 
    for(uint t = 0; t < tlevs; t++)
        gpuErrchk(cudaMemcpy(&array_h[t * get_nelem_per_t()], get_array_d(t), sizeof(T) * get_nelem_per_t(), cudaMemcpyDeviceToHost));
}


template <typename T>
void cuda_array_bc<T> :: enumerate()
{
	for(uint t = 0; t < tlevs; t++)
	{
		cout << t << endl;
		d_enumerate<<<block, grid_gp>>>(array_d_t, t, Nx, My);
	}
}


template <typename T>
void cuda_array_bc<T> :: update_ghost_points(uint tlev)
{
	/// Grid for updating ghost points in x-direction, per row, fixed column
	static dim3 block_gp_row(1, cuda::blockdim_col);
	static dim3 grid_gp_row(1, (Nx + cuda::num_gp_x + cuda::blockdim_col - 1) / cuda::blockdim_col);
	
	/// Grid for updating ghost points in y-direction, one column, fixed row
	static dim3 block_gp_col(cuda::blockdim_col);
	static dim3 grid_gp_col((My + cuda::num_gp_y + cuda::blockdim_col - 1) / cuda::blockdim_col);
	
	//cout << "updating ghost points\n";
	//cout << "block_gp_row = (" << block_gp_row.x << ", " << block_gp_row.y << ")" << endl;
	//cout << "grid_gp_row = (" << grid_gp_row.x << ", " << grid_gp_row.y << ")" << endl;
	//cout << "block_gp_col = (" << block_gp_col.x << "), grid_gp_col = " << grid_gp_col.x << endl;

	// Update left/right boundaries first. Value in the corner (0,0), (My, Nx) are updated 
	// using periodic boundary conditions later.	
	switch(boundaries.bc_left)
	{
        case cuda::bc_t::bc_dirichlet:
			d_update_gp_x_dirichlet_left<<<block_gp_col, grid_gp_col>>>(get_array_d(tlev), boundaries.bval_left, My, Nx);
			break;
        case cuda::bc_t::bc_neumann:
			d_update_gp_x_neumann_left<<<block_gp_col, grid_gp_col>>>(get_array_d(tlev), boundaries.bval_left, geom.delta_x, My, Nx);
			break;
	}
	
	switch (boundaries.bc_right)
	{
        case cuda::bc_t::bc_dirichlet:
			d_update_gp_x_dirichlet_right<<<block_gp_col, grid_gp_col>>>(get_array_d(tlev), boundaries.bval_right, My, Nx);
			break;
        case cuda::bc_t::bc_neumann:
			d_update_gp_x_neumann_right<<<block_gp_col, grid_gp_col>>>(get_array_d(tlev), boundaries.bval_right, geom.delta_x, My, Nx);
			break;
	}
	// Update top/bottom boundaries. Use periodic boundary conditions to also update corner
	// elements at (0,0), (My, Nx)
	d_update_gp_y_periodic_top<<<block_gp_row, grid_gp_row>>>(get_array_d(tlev), My, Nx);
	d_update_gp_y_periodic_bottom<<<block_gp_row, grid_gp_row>>>(get_array_d(tlev), My, Nx);
}

template <typename T>
cuda_array_bc<T> :: ~cuda_array_bc()
{
	if(array_h != nullptr)
		delete [] array_h;
	if(array_h_t != nullptr)
		delete [] array_h_t;
	if(array_d != nullptr)
		cudaFree(array_d);
	if(array_d_t != nullptr)
		cudaFree(array_d_t);
		
	cufftDestroy(plan_fw);
	cufftDestroy(plan_bw);
}

#endif /* cuda_array_bc_H_ */
