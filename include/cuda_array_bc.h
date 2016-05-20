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

//enum class bc_t {bc_dirichlet, bc_neumann, bc_periodic};


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
/// Returns true if the index pair (row, col) refer to a cellcenter(not a gp)
/// Returns false if the index pair (row, col) refer to a ghost points
__device__ inline bool d_is_cellcenter(const int row, const int col, const int Nx, const int My)
{
	return((col > cuda::gp_offset_y - 1) && 
		   (col < cuda::gp_offset_y + My) &&
		   (row > cuda::gp_offset_x - 1) && 
		   (row < cuda::gp_offset_x + Nx)); 
}


template <typename T>
__global__
void d_alloc_array_d_t(T** array_d_t, T* array, const uint tlevs, const uint Nx, const uint My)
{
    for(uint t = 0; t < tlevs; t++)
    {
        array_d_t[t] = &array[t * (Nx + cuda::num_gp_x) * (My + cuda::num_gp_y)];
    }
}


template <typename T>
__global__
void d_enumerate(T** array_d_t, const uint t, const uint Nx, const uint My)
{
	const int col = d_get_col(cuda::gp_offset_y);
    const int row = d_get_row(cuda::gp_offset_x);
    const int index = row * (My + cuda::num_gp_y) + col;

	if (d_is_cellcenter(col, row, My, Nx))
        array_d_t[t][index] = 10000 * T(t) + T(index);
	
}


template <typename T>
__global__
void d_set_to_zero(T** array_d_t, const uint tlev, const uint Nx, const uint My)
{
	const int col = d_get_col(0);
	const int row = d_get_row(0);
	const int index = row * (My + cuda::num_gp_x) + col;
	
	if (index < (Nx + cuda::num_gp_x) * (My + cuda::num_gp_y))
		array_d_t[tlev][index] = T(0.0);
}
	


/// Evaluate the function lambda d_op_fun (with type given by template parameter O)
/// on the cell centers
template <typename T, typename O>
__global__
void d_evaluate(T** array_d_t, O d_op_fun, const cuda::slab_layout_t geom, const uint tlev)
{
    //const int col = d_get_col(cuda::gp_offset_y);
    //const int row = d_get_row(cuda::gp_offset_x);
    const int col = d_get_col(0);
    const int row = d_get_row(0);
    const int index = row * (geom.My + cuda::num_gp_y) + col;

    if (d_is_cellcenter(col, row, geom.My, geom.Nx))
		array_d_t[tlev][index] = d_op_fun(d_get_row(0), d_get_col(0), geom);
}


/// Evaluate the function lambda d_op_fun (with type given by template parameter O)
/// on the cell centers. Same as d_evaluate, but pass the value at the cell center
/// to the lambda function 
template <typename T, typename O>
__global__
void d_evaluate_2(T** array_d_t, O d_op_fun, const cuda::slab_layout_t geom, const uint tlev)
{
	const int col = d_get_col(0);
    const int row = d_get_row(0);
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
void d_update_gp_x_dirichlet_left(T* array, T bval_left, const uint Nx, const uint My)
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
void d_update_gp_x_neumann_left(T* array, T bval_left, T deltax, const uint Nx, const uint My)
{
	const int col = d_get_col(cuda::gp_offset_y);
	const int row = cuda::gp_offset_x - 1;
	const int index = row * (My + cuda::num_gp_y) + col;
	
	if((col + 1 > cuda::gp_offset_y) && (col < My + cuda::gp_offset_y))
		array[index] = -1.0 * bval_left * deltax + array[(row + 1) * (My + cuda::num_gp_y) + col];
}


/// Update left ghost points using periodic boundary conditions
/// u[row * (My + cuda::num_gp_y) + gp_offset_x - 1] = u[(row + My) * (My + cuda::num_gp_y) + gp_offset_x - 1]
template <typename T>
__global__
void d_update_gp_x_periodic_left(T* array, const uint Nx, const uint My)
{
    const int col = d_get_col(cuda::gp_offset_y);
    const int row = cuda::gp_offset_x - 1;
    const int index = row * (My + cuda::num_gp_y) + col;

    if((col + 1 > cuda::gp_offset_y) && (col < My + cuda::gp_offset_y))
        array[index] = array[(row + Nx) * (My + cuda::num_gp_y) + col];
}


/// Update right ghost points, Dirichlet boundary conditions, located in column (cuda::gp_offset_x + My + 1) 
template <typename T>
__global__
void d_update_gp_x_dirichlet_right(T* array, T bval_right, const uint Nx, const uint My)
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
void d_update_gp_x_neumann_right(T* array, T bval_right, T deltax, const uint Nx, const uint My)
{
	const int col = d_get_col(cuda::gp_offset_y);
	const int row = Nx + cuda::gp_offset_x;
	const int index = row * (My + cuda::num_gp_y) + col;
	
	//printf("kernel: row = %d, col = %d, index = %d\n", row, col, index);
	if((col + 1 > cuda::gp_offset_y) && (col < My + cuda::gp_offset_y))	
		array[index] = bval_right * deltax + array[(row - 1) * (My + cuda::num_gp_y) + col];
}


/// Update left ghost points using periodic boundary conditions
/// u[row * (My + cuda::num_gp_y) + My + gp_offset_y] = u[row * (My + cuda::num_gp_y) + My + gp_offset_y - 1]
template <typename T>
__global__
void d_update_gp_x_periodic_right(T* array, const uint Nx, const uint My)
{
    const int col = d_get_col(cuda::gp_offset_y);
    const int row = Nx + cuda::gp_offset_x;
    const int index = row * (My + cuda::num_gp_y) + col;

    if((col + 1 > cuda::gp_offset_y) && (col < My + cuda::gp_offset_y))
        array[index] = array[(row - Nx) * (My + cuda::num_gp_y) + col];
}


template <typename T>
__global__
void d_update_gp_y_periodic_bottom(T* array, const uint Nx, const uint My)
{
	const int col = cuda::gp_offset_y - 1;
	// Don't use an offset here! Instead check for row > cuda::gp_offset_x - 2 to access the corner elements.
	const int row = d_get_row(0);
	const int index = row * (My + cuda::num_gp_y) + col;

	if((row > cuda::gp_offset_x - 2) && (row < Nx + cuda::gp_offset_x + 1))
    {
        printf("kernel: row = %d, col = %d, index = %d\n", row, col, index);
		array[index] = array[row * (My + cuda::num_gp_y) + col + My];
    }
}


template <typename T>
__global__
void d_update_gp_y_periodic_top(T* array, const uint Nx, const uint My)
{
	const int col = My + cuda::gp_offset_y;
	// Don't use an offset here! Instead check for row > cuda::gp_offset_x - 2 to access the corner elements.
	const int row = d_get_row(0);
	const int index = row * (My + cuda::num_gp_y) + col;
	
	if((row > cuda::gp_offset_x - 2) && (row < Nx + cuda::gp_offset_x + 1))
		array[index] = array[row * (My + cuda::num_gp_y) + col - My];	
}

#endif // __CUDACC_

template <typename T>
class cuda_array_bc{
public:
	cuda_array_bc(uint, uint, uint, cuda::bvals<T>, cuda::slab_layout_t);
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
                	os << std::setw(cuda::io_w) << std::setprecision(cuda::io_p) << std::fixed << src(n, m) << "\t";
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
    cuda::slab_layout_t geom;
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


const map<cuda::bc_t, string> bc_str_map
{
	{cuda::bc_t::bc_dirichlet, "Dirichlet"},
	{cuda::bc_t::bc_neumann, "Neumann"},
	{cuda::bc_t::bc_periodic, "Periodic"}
};

template <typename T>
cuda_array_bc<T> :: cuda_array_bc(uint _tlevs, uint _Nx, uint _My, cuda::bvals<T> bvals, cuda::slab_layout_t _geom) :
		tlevs(_tlevs), Nx(_Nx), My(_My), boundaries(bvals), geom(_geom), array_bounds(tlevs, Nx, My),
        block(dim3(cuda::blockdim_col, cuda::blockdim_row)),
		grid(dim3((My + cuda::num_gp_y + cuda::blockdim_col - 1) / cuda::blockdim_col, (Nx + cuda::num_gp_x + cuda::blockdim_row - 1) / cuda::blockdim_row)),
		array_d(nullptr),
		array_d_t(nullptr),
		array_d_t_host(new T*[tlevs]),
		array_h(new T[tlevs * (Nx + cuda::num_gp_x) * (My + cuda::num_gp_y)]),
		array_h_t(new T*[tlevs])
{
	gpuErrchk(cudaMalloc( (void***) &array_d_t, tlevs * sizeof(T*)));
	gpuErrchk(cudaMalloc( (void**) &array_d, tlevs * get_nelem_per_t() * sizeof(T)));

    //cout << "cuda_array_bc<T> ::cuda_array_bc<T>\t";
    //cout << "Nx = " << Nx << ", My = " << My << endl;
    //cout << "block = ( " << block.x << ", " << block.y << ")" << endl;
    //cout << "grid = ( " << grid.x << ", " << grid.y << ")" << endl;
    //cout << geom << endl;

	d_alloc_array_d_t<<<1, 1>>>(array_d_t, array_d, tlevs, Nx, My);
	gpuErrchk(cudaMemcpy(array_d_t_host, array_d_t, sizeof(T*) * tlevs, cudaMemcpyDeviceToHost));

	for(uint t = 0; t < tlevs; t++)
    {
		d_set_to_zero<<<grid, block>>>(array_d_t, t,  Nx, My);
        array_h_t[t] = &array_h[t * get_nelem_per_t()];
    }

    for(uint t = 0; t < tlevs; t++)

    cout << endl;
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
#ifdef DEGUG
    cout << "evaluate_device..." << endl;
	cout << "block: block.x = " << block.x << ", block.y = " << block.y << endl;
    cout << "grid: grid.x = " << grid.x << ", grid.y = " << grid.y << endl;
#endif //DEBUG
    d_evaluate<<<grid, block>>>(get_array_d_t(), [=] __device__ (uint n, uint m, cuda::slab_layout_t geom) -> T 
		{
			T x{geom.x_left + (T(n) + 0.0) * geom.delta_x};
			T y{geom.y_lo + (T(m) + 0.0) * geom.delta_y};
		    return(sin(2.0 * M_PI * x) + sin(2.0 * M_PI * y));
            //return(T(1000 * n + m));
		}, 
        geom, 0);
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


/// Initialize the spectral transformation depending on the boundary conditions.
/// In case of Dirichlet and Neumann BCs initialize a 1d transform along the y direction (consecutive elements)
/// In case of Periodic BCs initialize a 2d transformation on both directions

template <typename T>
void cuda_array_bc<T> :: init_dft()
{
	cufftResult err;
    int dft_size[2] = {0, 0};       // Size of the transformation
    int dft_onembed[2] = {0, 0};    // Input, embedded size of the transformation
    int dft_inembed[2] = {0, 0};    // Output, embedded size of the transformation
    int dist_real{0};               // Distance between two vectors inpu vectors for DFT, in units of double
    int dist_cplx{0};               // Distance between two complex input vectors for iDFT, in units of (2 * double)
    int istride{1};                 // Distance between two successive input and output elements in the least significant (that is, the innermost) dimension
    int ostride{1};
    switch(boundaries.bc_left)
    {
        case cuda::bc_t::bc_dirichlet:
            // fall through
        case cuda::bc_t::bc_neumann:
            // Initialize a 1d transformation
            dft_size[0] = int(My);                                  
            dft_onembed[0] = int(My / 2 + 1); 
            dist_real = int(My + cuda::num_gp_y); 
            dist_cplx = int(My / 2 + cuda::gp_offset_y / 2 + 1); 

            // Plan the DFT, D2Z
            if ((err = cufftPlanMany(&plan_fw, 
                                     1,             //int rank
                                     dft_size,      //int* n
                                     dft_size,      //int* inembed
                                     istride,       //int istride
                                     dist_real,     //int idist
                                     dft_onembed,   //int* onembed
                                     ostride,       //int ostride
                                     dist_cplx,     //int odist
                                     CUFFT_D2Z,     //cufftType type
                                     Nx)            //int batch
                ) != CUFFT_SUCCESS)
            {
                stringstream err_str;
                err_str << "Error planning 1d D2Z DFT: " << err << "\n";
                throw gpu_error(err_str.str());
            }
           
            // Plan the iDFT, Z2D 
            if((err = cufftPlanMany(&plan_bw,
                                    1,              //int rank
                                    dft_size,       //int* n
                                    dft_onembed,    //int* inembed
                                    istride,        //int istride
                                    dist_cplx,      //int idist
                                    dft_size,       //int* onembed
                                    ostride,        //int ostride
                                    dist_real,      //int odist
                                    CUFFT_Z2D,      //cufftType type
                                    Nx)             //int batch
               ) != CUFFT_SUCCESS)
            {
                stringstream err_str;
                err_str << "Error planning 1d Z2D DFT: " << err << "\n";
                throw gpu_error(err_str.str());
            }
            break;
        case cuda::bc_t::bc_periodic:
            // Initialize 2d transformation
            dft_size[1] = int(My);
            dft_size[0] = int(Nx);
            dft_inembed[1] = My + cuda::num_gp_y;
            dft_onembed[1] = My / 2 + cuda::gp_offset_y / 2 + 1;
            istride = 1;
            ostride = 1;

            if((err = cufftPlanMany(&plan_fw,
                                    2,              //int rank
                                    dft_size,       //int* n
                                    dft_inembed,    //int* inembed
                                    istride,        //int istride
                                    (My + cuda::num_gp_y) * (Nx + cuda::num_gp_x), //int idist
                                    dft_onembed,    //int* onembed
                                    ostride,        //int ostride
                                    (My / 2 + cuda::num_gp_y / 2 + 1) * (Nx + cuda::num_gp_x), //int odist
                                    CUFFT_D2Z,      //cufftType typ
                                    1)              //int batch
               ) != CUFFT_SUCCESS)
            {
                stringstream err_str;
                err_str << "Error planning 2d D2Z DFT: " << err << "\n";
                throw gpu_error(err_str.str());
            }

            // Plan inverse transformation
            if((err = cufftPlanMany(&plan_bw,
                                    2,
                                    dft_size,
                                    dft_onembed,
                                    ostride,
                                    (My / 2 + cuda::num_gp_y / 2 + 1) * (Nx + cuda::num_gp_x),
                                    dft_inembed,
                                    istride,
                                    (My + cuda::num_gp_y) * (Nx + cuda::num_gp_x),
                                    CUFFT_Z2D,
                                    1)
              ) != CUFFT_SUCCESS)
            {
                stringstream err_str;
                err_str << "Error planning 2d Z2D DFT: " << err << "\n";
                throw gpu_error(err_str.str());
            }


            break;
    }

}


    /// Perform DFT in y-direction, row-wise
    template <typename T>
    void cuda_array_bc<T> :: dft_r2c(const uint tlev)
    {
        const int offset = cuda::gp_offset_x * (My + cuda::num_gp_y) + cuda::gp_offset_y; 
        cufftResult err;
	
	/* Use the CUFFT plan to transform the signal in place. */
	if ((err = cufftExecD2Z(plan_fw, 
	                 get_array_d(tlev) + offset, 
					 (cufftDoubleComplex*) get_array_d(tlev) + offset / 2)
        ) != CUFFT_SUCCESS)
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
    // If we made a 1d DFT, normalize by My. Otherwise nomalize by Nx * My
    switch (boundaries.bc_left)
    {
        case cuda::bc_t::bc_dirichlet:
            // fall through
        case cuda::bc_t::bc_neumann:
            d_evaluate_2<<<block, grid>>>(get_array_d_t(), [=] __device__ (T in, uint n, uint m, cuda::slab_layout_t geom) -> T 
            {
                return(in / T(geom.My));
            }, geom, 0);
            break;
        case cuda::bc_t::bc_periodic:
            d_evaluate_2<<<block, grid>>>(get_array_d_t(), [=] __device__ (T in, uint n, uint m, cuda::slab_layout_t geom) -> T 
            {
                return(in / T(geom.Nx * geom.My));
            }, geom, 0);
            break;
    }
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

#ifdef DEBUG    
	cout << "updating ghost points\n";
	cout << "block_gp_row = (" << block_gp_row.x << ", " << block_gp_row.y << ")" << endl;
	cout << "grid_gp_row = (" << grid_gp_row.x << ", " << grid_gp_row.y << ")" << endl;
	cout << "block_gp_col = (" << block_gp_col.x << "), grid_gp_col = " << grid_gp_col.x << endl;
#endif //DEBUG

	// Update left/right boundaries first. Value in the corner (0,0), (My, Nx) are updated 
	// using periodic boundary conditions later.	
	switch(boundaries.bc_left)
	{
        case cuda::bc_t::bc_dirichlet:
			d_update_gp_x_dirichlet_left<<<grid_gp_col, block_gp_col>>>(get_array_d(tlev), boundaries.bval_left, Nx, My);
			break;
        case cuda::bc_t::bc_neumann:
			d_update_gp_x_neumann_left<<<grid_gp_col, block_gp_col>>>(get_array_d(tlev), boundaries.bval_left, geom.delta_x, Nx, My);
			break;
        case cuda::bc_t::bc_periodic:
            d_update_gp_x_periodic_left<<<grid_gp_col, block_gp_col>>>(get_array_d(tlev), Nx, My);
            break;
	}
	
	switch (boundaries.bc_right)
	{
        case cuda::bc_t::bc_dirichlet:
			d_update_gp_x_dirichlet_right<<<grid_gp_col, block_gp_col>>>(get_array_d(tlev), boundaries.bval_right, Nx, My);
			break;
        case cuda::bc_t::bc_neumann:
			d_update_gp_x_neumann_right<<<grid_gp_col, block_gp_col>>>(get_array_d(tlev), boundaries.bval_right, geom.delta_x, Nx, My);
			break;
        case cuda::bc_t::bc_periodic:
            d_update_gp_x_periodic_right<<<grid_gp_col, block_gp_col>>>(get_array_d(tlev), Nx, My);
            break;
	}
	// Update top/bottom boundaries. Use periodic boundary conditions to also update corner
	// elements at (0,0), (My, Nx)
	d_update_gp_y_periodic_top<<<grid_gp_row, block_gp_row>>>(get_array_d(tlev), Nx, My);
	d_update_gp_y_periodic_bottom<<<grid_gp_row, block_gp_row>>>(get_array_d(tlev), Nx, My);
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
