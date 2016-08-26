/*
 * Sample slab implementation for methods requiring several cuda_array_bc types
 *
 */
 
 
#include "cuda_array_bc.h"
#include "cucmplx.h"


enum class field_t{f1, f2};

#ifdef __CUDACC__


//__device__ inline int d_get_col(const int offset) {
//    return (blockIdx.x * blockDim.x + threadIdx.x + offset);
//}
//
//
//__device__ inline int d_get_row(const int offset) {
//    return (blockIdx.y * blockDim.y + threadIdx.y + offset);
//}
//
//
//__device__ inline bool d_is_cellcenter(const int col, const int row, const int My, const int Nx)
//{
//	return((col > gp_offset_y - 1) && 
//		   (col < My + gp_offset_y) &&
//		   (row > gp_offset_x - 1) && 
//		   (row < Nx + gp_offset_x)); 
//}

template <typename T>
__global__
void d_deriv_x(T* in, T* out, T inv_delta_x_2, slab_layout_t slab_layout)
{
   	const int col = d_get_col(cuda::gp_offset_y);
    const int row = d_get_row(cuda::gp_offset_x);
    const int index = row * (slab_layout.My + cuda::num_gp_y) + col;

	if (d_is_cellcenter(col, row, slab_layout.My, slab_layout.Nx))
        out[index] = (in[(row + 1) * (slab_layout.My + cuda::num_gp_y) + col] - in[(row - 1) * (slab_layout.My + cuda::num_gp_y) + col]) * inv_delta_x_2;
}


template <typename T>
__global__
void d_compute_dy1(CuCmplx<T>* in_arr,  
        CuCmplx<T>* out_y_arr,
		CuCmplx<T>* kmap, 
        const uint Nx, const uint My21)
{
	const uint row = blockIdx.y * blockDim.y + threadIdx.y;
	const uint col = blockIdx.x * blockDim.x + threadIdx.x;
    const uint idx = row * My21 + col;
   
    if(d_is_cellcenter(col, row, My21, Nx))
    {
            out_y_arr[idx] = in_arr[idx] * CuCmplx<T>(0.0, kmap[idx].im());
    }
}
template <typename T>
__global__
void d_compute_dy1(CuCmplx<T>* in_arr,  
        CuCmplx<T>* out_y_arr,
		CuCmplx<T>* kmap, 
        const uint Nx, const uint My21)
{
	const uint row = blockIdx.y * blockDim.y + threadIdx.y;
	const uint col = blockIdx.x * blockDim.x + threadIdx.x;
    const uint idx = row * My21 + col;
   
    if(d_is_cellcenter(col, row, My21, Nx))
    {
            out_y_arr[idx] = in_arr[idx] * CuCmplx<T>(0.0, kmap[idx].im());
    }
}
            //out_y_arr[idx] = in_arr[idx] * kmap[idx].im();


#endif //__CUDACC__

template <typename T>
class slab_bc{
    public:
        slab_bc(slab_layout_t, cuda::bvals<T>);
        
        void test_d_dx(uint tlev);
        void print_field(field_t);
        
        void test_d_dy(uint tlev);
        
    private:
        slab_layout_t my_layout;
        cuda::bvals<T> my_bv;
        cuda_array_bc<T> field1;
        cuda_array_bc<T> field2;
};


template <typename T>
slab_bc<T> :: slab_bc(slab_layout_t _slab_layout, cuda::bvals<T> _bv) :
    my_layout(_slab_layout), my_bv(_bv), 
    field1(1, my_layout.Nx, my_layout.My, my_bv, my_layout),
    field2(1, my_layout.Nx, my_layout.My, my_bv, my_layout)
{
    cout << "Creating slab" << endl;    
    field1.evaluate_device(0);
}


template <typename T>
void slab_bc<T> :: test_d_dx(uint tlev)
{
    static T inv_delta_x_2{0.5 / my_layout.delta_x};
    cout << "d_dx, dx = " << my_layout.delta_x << ", 1 / (2dx) = " << inv_delta_x_2 << endl;
    field1.update_ghost_points(0);
    d_deriv_x<<<field1.get_grid(), field1.get_block()>>>(field1.get_array_d(tlev), field2.get_array_d(tlev), inv_delta_x_2, my_layout);
}


template <typename T>
void slab_bc<T> :: test_d_dy(uint tlev)
{
    cerr << "test_d_dy: dummy routine" << endl;
    //field1.dft_r2c(0);
    //d_deriv_y<<<field1.get_grid(), field2.get_block()>>>((CuCmplx<T>*) field1.get_array_d(tlev), (CuCmplx<T>*) field2.get_array_d(tlev), my_layout);
    //field2.dft_c2r(0);
}

template <typename T>
void slab_bc<T> :: print_field(field_t f)
{
    cuda_array_bc<T>* f_ptr;
    switch(f)
    {
        case field_t::f1:
            f_ptr = &field1;
            break;
        case field_t::f2:
            f_ptr = &field2;
            break;
    }
    f_ptr -> copy_device_to_host();
    f_ptr -> dump_full();
}

