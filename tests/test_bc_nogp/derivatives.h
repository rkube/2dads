/*
 * Interface to derivation functions
 */

#include "cuda_types.h"
#include "cuda_array_bc_nogp.h"

__device__ inline size_t d_get_col_2(){
    return (blockIdx.x * blockDim.x + threadIdx.x); 
}


__device__ inline size_t d_get_row_2(){
    return (blockIdx.y * blockDim.y + threadIdx.y); 
}


__device__ inline bool good_idx2(size_t row, size_t col, const cuda::slab_layout_t geom){
    return((row < geom.Nx) && (col < geom.My));
}  


// compute the first derivative in x-direction in rows 1..Nx-2
// no interpolation needed
template <typename T>
__global__ 
void d_dx1_center(T* in, T* out, const cuda::bvals_t<T> bc, const cuda::slab_layout_t geom)
{
    const size_t col{d_get_col_2()};
    const size_t row{d_get_row_2()};
    const size_t index{row * (geom.My + geom.pad_y) + col};
    const T inv_2_dx{0.5 / geom.delta_x};

    if(row > 0 && row < geom.Nx - 1 && col < geom.My)
    {
        // Index of element to the "right" (in x-direction)
        const size_t idx_r{(row + 1) * (geom.My + geom.pad_y) + col};
        // Index of element to the "left" (in x-direction)
        const size_t idx_l{(row - 1) * (geom.My + geom.pad_y) + col};

        out[index] = (in[idx_r] - in[idx_l]) * inv_2_dx;
    }   
}


// Compute first derivative in row n=0
// Interpolate to the value in row n=-1 get the ghost point value
template <typename T>
__global__ 
void d_dx1_boundary_left(T* in, T* out, const cuda::bvals_t<T> bc, const cuda::slab_layout_t geom)
{
    const size_t col{d_get_col_2()};
    const size_t row{0};
    const size_t index{col};
    const T inv_2_dx{0.5 / geom.delta_x};

    if(col < geom.My)
    {
        // The value to the right in column 1
        const T val_r{in[(geom.My + geom.pad_y) + col]};
        // Interpolate the value in column -1
        T val_l{-1.0};
        switch(bc.bc_left)
        {
            case cuda::bc_t::bc_dirichlet:
                 val_l = 2.0 * bc.bval_left - in[index];
                break;
            case cuda::bc_t::bc_neumann:
                val_l = -1.0 * geom.delta_x * bc.bval_left + in[index];
                break;
            case cuda::bc_t::bc_periodic:
                val_l = in[(geom.Nx - 1) * (geom.My + geom.pad_y) + col];
                break;
        }
        out[index] = (val_r - val_l) * inv_2_dx;
    }

}


// Compute first derivative in row n = Nx - 1
// Interpolate the value at row n = Nx to get the ghost point value
template <typename T>
__global__ 
void d_dx1_boundary_right(T* in, T* out, const cuda::bvals_t<T> bc, const cuda::slab_layout_t geom)
{
    const size_t col{d_get_col_2()};
    const size_t row{geom.Nx - 1};
    const size_t index{row * (geom.My + geom.pad_y) + col};
    const T inv_2_dx{0.5 / geom.delta_x};

    if(col < geom.My)
    {
        // The value to the left in column Nx - 2
        const T val_l{in[(geom.Nx - 2) * (geom.My + geom.pad_y) + col]};
        // Interpolate the value in column -1
        T val_r{-1.0};
        switch(bc.bc_left)
        {
            case cuda::bc_t::bc_dirichlet:
                 val_r = 2.0 * bc.bval_right - in[index];
                break;
            case cuda::bc_t::bc_neumann:
                val_r = geom.delta_x * bc.bval_right + in[index];
                break;
            case cuda::bc_t::bc_periodic:
                val_r = in[col];
                break;
        }
        out[index] = (val_r - val_l) * inv_2_dx;
    }
}


template <typename T>
void dx_1(cuda_array_bc_nogp<T>& in, cuda_array_bc_nogp<T>& out, const size_t tlev,
             const cuda::slab_layout_t sl, const cuda::bvals_t<T> bc)
{
    cout << "Computing x derivative\n";

    // Size of the grid for boundary kernels in x-direction
    dim3 gridsize_line(int((sl.My + cuda::blockdim_row - 1) / cuda::blockdim_row));


    d_dx1_center<T> <<<in.get_grid(), in.get_block()>>>(in.get_array_d(tlev), out.get_array_d(0), bc, sl);
    d_dx1_boundary_left<<<gridsize_line, cuda::blockdim_row>>>(in.get_array_d(tlev), out.get_array_d(0), bc, sl);
    d_dx1_boundary_right<<<gridsize_line, cuda::blockdim_row>>>(in.get_array_d(tlev), out.get_array_d(0), bc, sl);
}


//
//template <typename T>
//class derivs
//{
//    public:
//        derivs(const cuda::slab_layout_t);
//        ~derivs();
//
//        /// @brief Compute first order x- and y-derivatives
//        /// @detailed Allocates memory for Fourier coefficients.
//        /// @detailed If spectral representation is available, use
//        /// @d_dx1_dy1 where they are passed as arguments instead
//        void d_dx1_dy1(cuda_array<T>&, cuda_array<T>&, cuda_array<T>&);
//        void d_dx1_dy1(const cuda_array<CuCmplx<T> >&,  cuda_array<CuCmplx<T> >&, cuda_array<CuCmplx<T> >&, const uint);
//        void d_dx1_dy1(const cuda_array<CuCmplx<T> >*,  cuda_array<CuCmplx<T> >*, cuda_array<CuCmplx<T> >*, const uint);
//        /// @brief Compute second order x- and y-derivatives
//        void d_dx2_dy2(cuda_array<T>&, cuda_array<T>&, cuda_array<T>&);
//        void d_dx2_dy2(cuda_array<CuCmplx<T> >&,  cuda_array<CuCmplx<T> >&, cuda_array<CuCmplx<T> >&, const uint);
//        /// @brief Compute Laplacian
//        void d_laplace(cuda_array<T>&, cuda_array<T>&, const uint);
//        void d_laplace(cuda_array<CuCmplx<T> >&, cuda_array<CuCmplx<T> >&, const uint);
//        void d_laplace(cuda_array<CuCmplx<T> >*, cuda_array<CuCmplx<T> >*, const uint);
//        /// @brief Invert Laplace equation
//        void inv_laplace(cuda_array<T>&, cuda_array<T>&, const uint);
//        void inv_laplace(cuda_array<CuCmplx<T> >&, cuda_array<CuCmplx<T> >&, const uint);
//        void inv_laplace(cuda_array<CuCmplx<T> >*, cuda_array<CuCmplx<T> >*, const uint);
//
//        void dft_r2c(T* in, CuCmplx<T>* out);
//        void dft_c2r(CuCmplx<T>* in, T* out);
//
//    private:
//        const unsigned int Nx;
//        const unsigned int My;
//        const T Lx;
//        const T Ly;
//        const T dx;
//        const T dy;
//
//        dim3 grid_my_nx21;
//        dim3 block_my_nx21;
//
//        cuda_array<CuCmplx<T> > kmap_dx1_dy1;
//        cuda_array<CuCmplx<T> > kmap_dx2_dy2;
//
//        cufftHandle plan_r2c;
//        cufftHandle plan_c2r;
//
//        void init_dft();
//};
//
//
//#ifdef __CUDACC__
//
//template <typename T>
//derivs<T> :: derivs(const cuda::slab_layout_t sl) :
//    Nx(sl.Nx), My(sl.My),
//    Lx(T(sl.Nx) * T(sl.delta_x)),
//    Ly(T(sl.My) * T(sl.delta_y)),
//    dx(T(sl.delta_x)), dy(T(sl.delta_y)),
//    kmap_dx1_dy1(1, My, Nx / 2 + 1),
//    kmap_dx2_dy2(1, My, Nx / 2 + 1)
//{
//    init_dft();
//    // Generate first and second derivative map;
//    gen_kmap_dx1_dy1<<<kmap_dx1_dy1.get_grid(), kmap_dx1_dy1.get_block()>>>(kmap_dx1_dy1.get_array_d(), cuda::TWOPI / Lx,
//                                                                            cuda::TWOPI / Ly, My, Nx / 2 + 1);
//    gen_kmap_dx2_dy2<<<kmap_dx2_dy2.get_grid(), kmap_dx2_dy2.get_block()>>>(kmap_dx2_dy2.get_array_d(), cuda::TWOPI / Lx,
//                                                                            cuda::TWOPI / Ly, My, Nx / 2 + 1);
//    //ostream of;
//    //of.open("k2map.dat");
//    //of << kmap_dx2_dy2 << endl;
//    //of.close()
//    gpuStatus();
//}


