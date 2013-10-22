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


// Kernels to be used below
__global__ void enumerate(cmplx_t* array, int Nx, int My){
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int index = row * (gridDim.x * blockDim.x + 1) + col;
    double val = 0.0;
    if (index < Nx * (My / 2 + 1)){
        val = double(threadIdx.x + 10 * threadIdx.y + 1000 * blockIdx.x + 10000 * blockIdx.y);
        array[index] = make_cuDoubleComplex(val, -1.0);
    }
}

__global__ void normalize(cmplx_t* array, int Nx, int My){
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int index = row * (gridDim.x * blockDim.x + 1) + col;
    const double norm = double(Nx * My);

    if (index < Nx * (My / 2 + 1)){
        array[index] = make_cuDoubleComplex(array[index].x / norm, array[index].y / norm);
        //array[index] /= norm;
    }
}

__global__ void init_sine(cmplx_t* array, int Nx, int My, const double PI){
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int index = row * (gridDim.x * blockDim.x + 1) + col;

    const double dx = 2.0 * PI / double(Nx);
    cmplx_t val = make_cuDoubleComplex( sin(double(2*col) * dx), sin(double(2 * col + 1) * dx));
    //cmplx_t val = std::complex<double>(sin(double(2*col) * dx), sin(double(2 * col + 1) * dx));

    if (index < Nx * (My / 2 + 1)){
            array[index] = val;
    }
}

// Template kernels
template <typename T>
__global__ void enumerate(T* array, int Nx, int My)
{
    const int row = blockIdx.y * blockDim.y + threadIdx.x;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int index = row * (gridDim.x * blockDim.x + 1) + col;
    if (index < Nx * My)
        array[index] = double(threadIdx.x + 10 * threadIdx.y + 1000 * blockIdx.x + 10000 * blockIdx.y);
}


template <typename T>
__global__ void set_constant(T* array, T val, int Nx, int My)
{
    const int row = blockIdx.y * blockDim.y + threadIdx.x;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int index = row * (gridDim.x * blockDim.x + 1) + col;
    if (index < Nx * My)
        array[index] = val;
}


// The default constructor does nothing
template <class T>
cuda_array<T> :: cuda_array() :
    Nx(16),
    My(16),
    tlevs(3),
    array_d(NULL), array_d_t(NULL), array_h(NULL), array_h_t(NULL)
{
}


// Use the constructor that explicitly defines array size and time levels
// Pad Nx and My to multiples of block_nx, block_my defined in
// cuda_array.h
template <class T>
cuda_array<T> :: cuda_array(unsigned int nx, unsigned int my, unsigned int t) :
    Nx(nx), My(my), tlevs(t), bounds(tlevs, Nx, My),
    array_d(NULL), array_d_t(NULL),
    array_h(NULL), array_h_t(NULL)
{
    //cudaError_t err;
    // Determine grid size for kernel launch
    block = dim3(cuda_blockdim_my, cuda_blockdim_nx);
    grid = dim3(My / (2 * cuda_blockdim_my), Nx / cuda_blockdim_nx);
    //grid = dim3(1, 1);

    cout << "Array size: " << Nx << "," << My << "," << tlevs << "\n";
    // Allocate device memory
    size_t nelem = sizeof(T) * tlevs * Nx * My;
    gpuErrchk( cudaMalloc( (void**) &array_d, nelem) );
    array_h = (T*) calloc(Nx * My * tlevs, sizeof(T));

    cout << "Device data at " << array_d << "\t";
    cout << "Host data at " << array_h << "\n";
    cout << nelem << " bytes of data\n";
    // array_[hd]_t is an array of pointers allocated on the host
    array_d_t = (T**) calloc(tlevs, sizeof(T*));
    array_h_t = (T**) calloc(tlevs, sizeof(T*));

    // array_t[i] points to the i-th time level
    for(unsigned int tl = 0; tl < tlevs; tl++){
        array_d_t[tl] = &array_d[tl * Nx * My];
        array_h_t[tl] = &array_h[tl * Nx * My];
        cout << "time level " << t << " at " << array_d_t[t] << "(device)\t";
        cout << array_h_t[t] << "(host)\n";
    }

    //array_h_d = reinterpret_cast<double*> (malloc(Nx * My * sizeof(double)));

//    // Create cufft plans
//    cout << "Planning FFTs\n";
//    int transform_size[2] = {Nx, My};
//
//    cufftCall( cufftPlanMany(&plan_fw, 2, transform_size,
//                             NULL, 1, 0,
//                             NULL, 1, 0,
//                             CUFFT_D2Z, 1) );
//    cufftCall( cufftPlanMany(&plan_bw, 2, transform_size,
//                             NULL, 1, 0,
//                             NULL, 1, 0,
//                             CUFFT_Z2D, 1) );
//    // Set memory layout to FFTW native
//    cufftCall( cufftSetCompatibilityMode(plan_fw, CUFFT_COMPATIBILITY_FFTW_PADDING) );
//    cufftCall( cufftSetCompatibilityMode(plan_bw, CUFFT_COMPATIBILITY_FFTW_PADDING) );
}

template <class T>
cuda_array<T> :: ~cuda_array(){
    cudaFree(array_d);
    free(array_h);
    free(array_d_t);
    free(array_h_t);
}


// Access functions for private members

template <class T>
void cuda_array<T> :: enumerate_array(const int t)
{
    // Call enumeration kernel
    //enumerate<<<grid, block>>>(array_d, Nx, My);
    // ToDo: write enumerate as a template function
}

// Operators
template <class T>
cuda_array<T>& cuda_array<T> :: operator= (const cuda_array<T>& rhs)
{
    // Copy array data from one array to another array
    // Ensure dimensions are equal
    /*if ( (void*) this != (void*) &rhs) {
        return *this;
    }*/
	cout << "If i had a kernel I would set the array to" << rhs << "\n";
    return *this;
}

template <class T>
cuda_array<T>& cuda_array<T> :: operator= (const T& rhs)
{
    set_constant<<<grid, block>>>(array_d, rhs, Nx, My);
    return *this;
}


template <class T>
T& cuda_array<T> :: operator()(unsigned int t, unsigned int n, unsigned int m)
{
	if (!bounds(t, n, m))
		throw out_of_bounds_err(string("T& cuda_array<T> :: operator()(uint, uint, uint): out of bounds\n";));
	return (*(array_t[t] + address(n, m)));
}


template <class T>
T cuda_array<T> :: operator()(unsigned int t, unsigned int n, unsigned int m)
{
	if (!bounds(t, n, m))
		throw out_of_bounds_err(string("T cuda_array<T> :: operator()(uint, uint, uint): out of bounds\n";));
	return (*(array_t[t] + address(n, m)));

}



//void cuda_array :: fft_forward(){
//    cufftCall( cufftExecD2Z(plan_fw, reinterpret_cast<double*>(array_d), array_d) );
//    //cufftCall( cufftExecD2Z(plan_fw, reinterpret_cast<double*>(array_d), reinterpret_cast<cufftDoubleComplex*>(array_d)) );
//}

//void cuda_array :: fft_backward(){
//    cufftCall( cufftExecZ2D(plan_bw, array_d, reinterpret_cast<double*>(array_d)) );
//    //cufftCall( cufftExecZ2D(plan_bw, reinterpret_cast<cufftDoubleComplex*> (array_d), reinterpret_cast<double*>(array_d)) );
//    normalize<<<grid, block>>>(array_d, Nx, My);
//}

// Copy full complex array from device to host
// The array is contiguous, so use only one memcpy
template <class T>
void cuda_array<T> :: copy_device_to_host() const {
    const size_t size_line = Nx * My * tlevs * sizeof(T);
    cout << "Copying " << size_line << " bytes to " << array_h << " (host) to ";
    cout << array_d << " (device)\n";
    gpuErrchk(cudaMemcpy(array_h, array_d, size_line, cudaMemcpyDeviceToHost));
}

/*
// Data transfer from device to host
void cuda_array :: copy_device_to_host_d(const int tlev) const {
    // Create an image of the device data on host, omitt padding data

    int offset_host = 0;
    int offset_device = 0;
    // Argument to memcpy, how many bytes to copy. Include sizeof.
    const size_t size_line = My * sizeof(double);
    // Use these to increase address. sizeof is implicitly included in pointer arithmetic,
    // that is address + stride points to address[stride]. Don't include sizeof(%) here :)
    // stride_host is just My, array_h_d does not included padding for fourier trafos
    const size_t stride_host = My;
    // stride on device has to consider the padding used for fourier trafo
    // Also note the reinterpret_cast<double> in the Memcpy call below.
    // Since array_d is a pointer to complex, interpret it as a double so that offset_device
    // is counted in sizeof(double) instead of sizeof(cmplx_t)
    const size_t stride_device = (My + 2);
    cout << "Copying " << Nx << " lines of " << size_line << " bytes from device to host\n";
    // Copy array slice-wise from device

    for(int n = 0; n < Nx; n++){
        if ((n % 16) == 0) {
            cout << "Iteration " << n << ": Copying " << size_line << " bytes from ";
            cout << array_d + offset_device << "(device) -> " << array_h_d + offset_host << "(host)\n";
        }
        gpuErrchk( cudaMemcpy(array_h_d + offset_host, reinterpret_cast<double*> (array_d) + offset_device, size_line, cudaMemcpyDeviceToHost) );
        offset_host += stride_host;
        offset_device += stride_device;
    }
}
*/

/*
void cuda_array :: copy_host_to_device_d(const int tlev) const {
    // Copy contents of host array to device
    // Use loop since line_size < stride_size
    int offset_host = 0;
    int offset_device = 0;
    const int size_line = My * sizeof(double);
    const int stride_host = My;
    const int stride_device = (My + 2);
    // Copy slive-wise from host to device
    cout << "Copying " << Nx << " lines of " << size_line << " bytes from host to device\n";
    for(int n = 0; n < Nx; n++){
        if( (n%16) == 0){
            cout << "Iteration " << n << ": Copying " << size_line << " bytes: ";
            cout << array_h_d + offset_host << "(host) -> " << array_d + offset_device << "(device)\n";
        }
        gpuErrchk( cudaMemcpy(reinterpret_cast<double*> (array_d) + offset_device, array_h_d + offset_host, size_line, cudaMemcpyHostToDevice) );
        offset_host += stride_host;
        offset_device += stride_device;
    }
}
*/

/*
// Dump host array in double precision
void cuda_array :: dump_array_d() const {
    class min_eps{
        public:
            min_eps(double e) : eps(e) {};
            double operator()(double val) {return (fabs(val) < eps ? 0.0 : val);}
        private:
            const double eps;
    };
    min_eps cout_eps(1e-6);

    for(int n = 0; n < Nx; n++){
        cout << n << ":\t";
        for(int m = 0; m < My; m++){
            cout << cout_eps(*(array_h_d + n * (My) + m)) << "\t";
        }
        cout << "\n\n";
    }
}
*/

/*
void cuda_array :: init_arr_d_host() {
    for(int n = 0; n < Nx; n++){
        for(int m = 0; m < My; m++){
            array_h_d[n * My + m] = double(n) * 1000.0 + double(m);
        }
    }
}
*/

/*
void cuda_array :: init_arr_d_host(double val){
    for(int n = 0; n < Nx; n++){
        for(int m = 0; m < My; m++){
            array_h_d[n * My + m] = val;
        }
    }
}
*/

/*
void cuda_array :: init_arr_c_host(val){
    for(int n = 0; n < Nx; n++){
        for(int m = 0; m < My / 2 + 1; m++){
            array_h_c[n * My + m] = val;
        }
    }
}
*/

/*
void cuda_array :: init_arr_d_sine(){
    init_sine<<<grid, block>>>(array_d, Nx, My, PI);
}
*/

template class cuda_array<cmplx_t>;
template class cuda_array<double>;

// End of file cuda_array.cpp



