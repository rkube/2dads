#ifndef UTILITY_H
#define UTILITY_H

#include "cuda_array_bc_nogp.h"


namespace device{
#ifdef __CUDACC__
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

// Compute wavenumbers for derivation in Fourier-space
// kmap_d1 holds (kx, ky) wavenumbers
// kmap_d2 holds (kx^2, ky^2) wavenumbers 
//
// To compute derivatives in Fourier space do:
//      u_x_hat[index] = u_hat[index] * complex(0.0, kmap_d1[index].re())
//      u_y_hat[index] = u_hat[index] * complex(0.0, kmap_d1[index].im())
//
//      u_xx_hat[index] = u_hat[index] * kmap_d2.re()
//      u_yy_hat[index] = u_hat[index] * kmap_d2.im()
//
// for the entire array 
//

template <typename T>
__global__
void kernel_gen_coeffs(CuCmplx<T>* kmap_d1, CuCmplx<T>* kmap_d2, twodads::slab_layout_t geom)
{
    const size_t col{cuda :: thread_idx :: get_col()};
    const size_t row{cuda :: thread_idx :: get_row()};
    const size_t index{row * (geom.get_my() + geom.get_pad_y()) + col}; 
    const T two_pi_Lx{twodads::TWOPI / geom.get_Lx()};
    const T two_pi_Ly{twodads::TWOPI / (static_cast<T>((geom.get_my() - 1) * 2) * geom.get_deltay())}; 

    CuCmplx<T> tmp1(0.0, 0.0);
    CuCmplx<T> tmp2(0.0, 0.0);

    if(row < geom.get_nx() / 2)
        tmp1.set_re(two_pi_Lx * T(row));

    else if (row == geom.get_nx() / 2)
        tmp1.set_re(0.0);
    else
        tmp1.set_re(two_pi_Lx * (T(row) - T(geom.get_nx())));

    if(col < geom.get_my() - 1)
    {
        tmp1.set_im(two_pi_Ly * T(col));
        tmp2.set_im(-1.0 * two_pi_Ly * two_pi_Ly * T(col * col));
    }
    else
    {
        tmp2.set_im(-1.0 * two_pi_Ly * two_pi_Ly * T(col * col));
        tmp1.set_im(0.0);
    }

    if(col < geom.get_my() && row < geom.get_nx())
    {
        kmap_d1[index] = tmp1;
        kmap_d2[index] = tmp2;
    }
}

// Multiply the input array with the real/imaginary part of the map. Store result in output
template <typename T, typename O>
__global__
void kernel_multiply_map(CuCmplx<T>* in, CuCmplx<T>* map, CuCmplx<T>* out, O op_func,
                            twodads::slab_layout_t geom)
{
    const size_t col{cuda :: thread_idx :: get_col()};
    const size_t row{cuda :: thread_idx :: get_row()};
    const size_t index{row * (geom.get_my() + geom.get_pad_y()) + col}; 

    if(col < geom.get_my() && row < geom.get_nx())
    {
        out[index] = op_func(in[index], map[index]);
    }
}


#endif //__CUDACC__
}

namespace utility
{
    template <typename T>
    void print(const cuda_array_bc_nogp<T, allocator_host>& vec, const size_t tidx, std::ostream& os)
    {
        address_t<T>* address = vec.get_address_ptr();
        const size_t nelem_m{vec.is_transformed(tidx) ? vec.get_geom().get_my()  + vec.get_geom().get_pad_y(): vec.get_geom().get_my()};

        for(size_t n = 0; n < vec.get_geom().get_nx(); n++)
        {
            for(size_t m = 0; m < nelem_m; m++)
            {
                os << std::setw(twodads::io_w) << std::setprecision(twodads::io_p) << std::fixed << (*address).get_elem(vec.get_tlev_ptr(tidx), n, m) << "\t";
            }
            os << std::endl;
        }
    }


    template <typename T>
    void print(const cuda_array_bc_nogp<T, allocator_host>& vec, const size_t tidx, std::string fname)
    {
        std::ofstream os(fname, std::ofstream::trunc);
        address_t<T>* address = vec.get_address_ptr();
        const size_t nelem_m{vec.is_transformed(tidx) ? vec.get_geom().get_my() : vec.get_geom().get_my() + vec.get_geom().get_pad_y()};

        for(size_t n = 0; n < vec.get_geom().get_nx(); n++)
        {
            for(size_t m = 0; m < nelem_m; m++)
            {
                os << std::setw(twodads::io_w) << std::setprecision(twodads::io_p) << std::fixed << (*address).get_elem(vec.get_tlev_ptr(tidx), n, m) << "\t";
            }
            os << std::endl;
        }
        os.close();
    }


    template <typename T>
    void normalize(cuda_array_bc_nogp<T, allocator_host>& vec, const size_t tlev)
    {
        switch(vec.get_bvals().get_bc_left())
        {
            case twodads::bc_t::bc_dirichlet:
            // fall through
            case twodads::bc_t::bc_neumann:
                vec.apply([] (T value, const size_t n, const size_t m, twodads::slab_layout_t geom) -> T
                          {return(value / T(geom.get_nx()));},
                          tlev);
                break;
            case twodads::bc_t::bc_periodic:
                vec.apply([] (T value, const size_t n, const size_t m, twodads::slab_layout_t geom) -> T
                          {return(value / T(geom.get_nx() * geom.get_my()));},
                          tlev);
            break;
        }

    }

    template <typename T>
    T L2(cuda_array_bc_nogp<T, allocator_host>&vec, const size_t tlev)
    {
        T tmp{0.0};
        address_t<T>* addr{vec.get_address_ptr()};
        T* data_ptr = vec.get_tlev_ptr(tlev); 

        for(size_t n = 0; n < vec.get_geom().get_nx(); n++)
        {
            for(size_t m = 0; m < vec.get_geom().get_my(); m++)
            {
                tmp += fabs(addr -> get_elem(data_ptr, n, m) * addr -> get_elem(data_ptr, n, m));
            }
            //std::cout << "L2: " << addr -> get_elem(data_ptr, n, 0) << ", " << fabs(addr -> get_elem(data_ptr, n, 0)) << ", " << abs(addr -> get_elem(data_ptr, n, 0) * addr -> get_elem(data_ptr, n, 0)) << std::endl;
        }

        tmp = sqrt(tmp / T(vec.get_geom().get_nx() * vec.get_geom().get_my()));
        return tmp;
    }

    template <typename T>
    T mean(cuda_array_bc_nogp<T, allocator_host>&vec, const size_t tlev)
    {
        T sum{0.0};
        address_t<T>* addr{vec.get_address_ptr()};
        T* data_ptr = vec.get_tlev_ptr(tlev);

        for(size_t n = 0; n < vec.get_geom().get_nx(); n++)
        {
            for(size_t m = 0; m < vec.get_geom().get_my(); m++)
            {
                sum += addr -> get_elem(data_ptr, n, m);
            }
        }
        sum = sum / T(vec.get_geom().get_nx() * vec.get_geom().get_my());
        return sum;
    }


#ifdef __CUDACC__
    template <typename T>
    cuda_array_bc_nogp<T, allocator_host> create_host_vector(cuda_array_bc_nogp<T, allocator_device>& src)
    {
        cuda_array_bc_nogp<T, allocator_host> res (src.get_geom(), src.get_bvals(), src.get_tlevs());
        for(size_t t = 0; t < src.get_tlevs(); t++)
        {
            gpuErrchk(cudaMemcpy(res.get_tlev_ptr(t), src.get_tlev_ptr(t), src.get_geom().get_nelem_per_t() * sizeof(T), cudaMemcpyDeviceToHost));
        }
        return(res);
    }

    template <typename T>
    void update_host_vector(cuda_array_bc_nogp<T, allocator_host>& dst, cuda_array_bc_nogp<T, allocator_device>& src)
    {
        assert(dst.get_geom() == src.get_geom());
        for(size_t t = 0; t < src.get_tlevs(); t++)
        {
            gpuErrchk(cudaMemcpy(dst.get_tlev_ptr(t), src.get_tlev_ptr(t), src.get_geom().get_nelem_per_t() * sizeof(T), cudaMemcpyDeviceToHost));
        }
    }


    template <typename T>
    void print(cuda_array_bc_nogp<T, allocator_device>& vec, const size_t tlev, std::string fname)
    {
        print(create_host_vector(vec), tlev, fname);
    }


    template <typename T>
    void print(cuda_array_bc_nogp<T, allocator_device>& vec, const size_t tlev, std::ostream& os)
    {
        cuda_array_bc_nogp<T, allocator_host> tmp = create_host_vector(vec);
        print(tmp, tlev, os);
    }



    template <typename T>
    T L2(cuda_array_bc_nogp<T, allocator_device>& vec, const size_t tlev)
    {
        // Configuration for reduction kernel
        T* data_ptr = vec.get_tlev_ptr(tlev);
        const size_t shmem_size_row = vec.get_geom().get_nx() * sizeof(T);
        //const dim3 grid(vec.get_grid());
        //const dim3 block(vec.get_block());
        const dim3 blocksize_row(static_cast<int>(vec.get_geom().get_nx()), 1, 1);
        const dim3 gridsize_row(1, static_cast<int>(vec.get_geom().get_my()), 1);

        T rval{0.0};

        // temporary value profile
        //T* h_tmp_profile(new T[Nx]);
        T* d_rval_ptr{nullptr};
        T* d_tmp_profile{nullptr};
        T* device_copy{nullptr};

        // Result from 1d->0d reduction on device
        gpuErrchk(cudaMalloc((void**) &d_rval_ptr, sizeof(T)));
        // Result from 2d->1d reduction on device
        gpuErrchk(cudaMalloc((void**) &d_tmp_profile, vec.get_geom().get_nx() * sizeof(T)));
        // Copy data to non-strided memory layout
        gpuErrchk(cudaMalloc((void**) &device_copy, vec.get_geom().get_nx() * vec.get_geom().get_my() * sizeof(T)));

        // Geometry of the temporary array, no padding
        twodads::slab_layout_t tmp_geom{vec.get_geom().get_xleft(), vec.get_geom().get_deltax(), vec.get_geom().get_ylo(), 
                                        vec.get_geom().get_deltay(), vec.get_geom().get_nx(), 0, vec.get_geom().get_my(), 0, 
                                        vec.get_geom().get_grid()};

        // Create device copy column-wise, ignore padding
        for(size_t n = 0; n < vec.get_geom().get_nx(); n++)
        {
            gpuErrchk(cudaMemcpy((void*) (device_copy + n * tmp_geom.get_my()),
                                 (void*) (data_ptr + n * (vec.get_geom().get_my() + vec.get_geom().get_pad_y())), 
                                 vec.get_geom().get_my() * sizeof(T), 
                                 cudaMemcpyDeviceToDevice));
        }

        // Take the square of the absolute value
        device :: kernel_apply_single<<<vec.get_grid(), vec.get_block()>>>(device_copy,
                                                       [] __device__ (T in, const size_t n, const size_t m, const twodads::slab_layout_t& geom ) -> T 
                                                       {return(abs(in) * abs(in));}, 
                                                       tmp_geom);
        gpuErrchk(cudaPeekAtLastError());
        //T* tmp_arr(new T[Nx * My]);
        //gpuErrchk(cudaMemcpy(tmp_arr, device_copy.get(), get_nx() * get_my() * sizeof(T), cudaMemcpyDeviceToHost));
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
        device :: kernel_reduce<<<gridsize_row, blocksize_row, shmem_size_row>>>(device_copy, d_tmp_profile, 
                                                                       [=] __device__ (T op1, T op2) -> T {return(op1 + op2);},
                                                                       1, tmp_geom.get_nx(), tmp_geom.get_nx(), tmp_geom.get_my());
        gpuErrchk(cudaPeekAtLastError());
        //gpuErrchk(cudaMemcpy(h_tmp_profile, d_tmp_profile.get(), get_nx() * sizeof(T), cudaMemcpyDeviceToHost));
        //for(size_t n = 0; n < Nx; n++)
        //{
        //    cout << n << ": " << h_tmp_profile[n] << endl;
        //}
        // Perform 1d -> 0d reduction
        device :: kernel_reduce<<<1, tmp_geom.get_nx(), shmem_size_row>>>(d_tmp_profile, d_rval_ptr, 
                                                       [=] __device__ (T op1, T op2) -> T {return(op1 + op2);},
                                                       1, tmp_geom.get_nx(), tmp_geom.get_nx(), 1);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaMemcpy(&rval, (void*) d_rval_ptr, sizeof(T), cudaMemcpyDeviceToHost));

        cudaFree(device_copy);
        cudaFree(d_tmp_profile);
        cudaFree(d_rval_ptr);

        return(sqrt(rval / static_cast<T>(tmp_geom.get_nx() * tmp_geom.get_my())));
    }


    template <typename T>
    cuda_array_bc_nogp<T, allocator_host> create_host_vector(cuda_array_bc_nogp<T, allocator_device>* src)
    {
        cuda_array_bc_nogp<T, allocator_host> res (src -> get_geom(), src -> get_bvals(), src -> get_tlevs());
        for(size_t t = 0; t < src -> get_tlevs(); t++)
        {
            gpuErrchk(cudaMemcpy(res.get_tlev_ptr(t), src -> get_tlev_ptr(t), (src -> get_geom()).get_nelem_per_t() * sizeof(T), cudaMemcpyDeviceToHost));
        }
        return(res);

    }


    template <typename T>
    void normalize(cuda_array_bc_nogp<T, allocator_device>& vec, const size_t tlev)
    {
        switch (vec.get_bvals().get_bc_left())
        {
            case twodads::bc_t::bc_dirichlet:
            case twodads::bc_t::bc_neumann:
                vec.apply([] __device__ (T value, const size_t n, const size_t m, twodads::slab_layout_t geom) -> T
                          {return(value / T(geom.get_my()));}, 
                          tlev);
                break;
            case twodads::bc_t::bc_periodic:
                vec.apply([] __device__ (T value, const size_t n, const size_t m, twodads::slab_layout_t geom) -> T
                          {return value / T(geom.get_nx() * geom.get_my());},
                          tlev);
                break; 
        }
    }
#endif //CUDACC


    namespace bispectral
    {
#ifdef __CUDACC__
    template <typename T>
    void init_deriv_coeffs(cuda_array_bc_nogp<CuCmplx<T>, allocator_device>& coeffs_dy1,
                           cuda_array_bc_nogp<CuCmplx<T>, allocator_device>& coeffs_dy2,
                           const twodads::slab_layout_t geom_my21,
                           allocator_device<T>)
    {
        const dim3 block_my21(cuda::blockdim_col, cuda::blockdim_row);
        const dim3 grid_my21((geom_my21.get_my() + cuda::blockdim_col - 1) / cuda::blockdim_col,
                             (geom_my21.get_nx() + cuda::blockdim_row - 1) / (cuda::blockdim_row));

        device :: kernel_gen_coeffs<<<grid_my21, block_my21>>>(coeffs_dy1.get_tlev_ptr(0), coeffs_dy2.get_tlev_ptr(0), geom_my21);
        gpuErrchk(cudaPeekAtLastError());    
    }
#endif //__CUDACC__

#ifndef __CUDACC__
    template <typename T>
    void init_deriv_coeffs(cuda_array_bc_nogp<CuCmplx<T>, allocator_host>& coeffs_dy1,
                           cuda_array_bc_nogp<CuCmplx<T>, allocator_host>& coeffs_dy2,
                           const twodads::slab_layout_t& geom_my21,
                           allocator_host<T>)
    {
        const T two_pi_Lx{twodads::TWOPI / geom_my21.get_Lx()};
        const T two_pi_Ly{twodads::TWOPI / (static_cast<T>((geom_my21.get_my() - 1) * 2) * geom_my21.get_deltay())};

        size_t n{0};
        size_t m{0};
        // Access data in coeffs_dy via T get_elem function below.
        address_t<CuCmplx<T>>* arr_dy1{coeffs_dy1.get_address_ptr()};
        address_t<CuCmplx<T>>* arr_dy2{coeffs_dy2.get_address_ptr()};
  
        CuCmplx<T>* dy1_data = coeffs_dy1.get_tlev_ptr(0);
        CuCmplx<T>* dy2_data = coeffs_dy2.get_tlev_ptr(0);
        
        /////////////////////////////////////////////////////////////////////////////////////////////
        // n = 0..nx/2-1
        for(n = 0; n < geom_my21.get_nx() / 2; n++)
        {
            for(m = 0; m < geom_my21.get_my() - 1; m++)
            {
                (*arr_dy1).get_elem(dy1_data, n, m).set_re(two_pi_Lx * T(n)); 
                (*arr_dy1).get_elem(dy1_data, n, m).set_im(two_pi_Ly * T(m));

                (*arr_dy2).get_elem(dy2_data, n, m).set_re(-1.0 * two_pi_Lx * two_pi_Lx * T(n * n));
                (*arr_dy2).get_elem(dy2_data, n, m).set_im(-1.0 * two_pi_Ly * two_pi_Ly * T(m * m));
            }
            m = geom_my21.get_my() - 1;
            (*arr_dy1).get_elem(dy1_data, n, m).set_re(two_pi_Lx * T(n));
            (*arr_dy1).get_elem(dy1_data, n, m).set_im(T(0.0));

            (*arr_dy2).get_elem(dy2_data, n, m).set_re(-1.0 * two_pi_Lx * two_pi_Lx * T(n * n));
            (*arr_dy2).get_elem(dy2_data, n, m).set_im(-1.0 * two_pi_Ly * two_pi_Ly * T(m * m));
        }
        
        /////////////////////////////////////////////////////////////////////////////////////////////
        // n = nx/2
        n = geom_my21.get_nx() / 2;
        for(m = 0; m < geom_my21.get_my() - 1; m++)
        {
            (*arr_dy1).get_elem(dy1_data, n, m).set_re(T(0.0)); 
            (*arr_dy1).get_elem(dy1_data, n, m).set_im(two_pi_Ly * T(m));

            (*arr_dy2).get_elem(dy2_data, n, m).set_re(-1.0 * two_pi_Lx * two_pi_Lx * T(n * n));
            (*arr_dy2).get_elem(dy2_data, n, m).set_im(-1.0 * two_pi_Ly * two_pi_Ly * T(m * m));
        }
        m = geom_my21.get_my() - 1;

        (*arr_dy1).get_elem(dy1_data, n, m).set_re(T(0.0));
        (*arr_dy1).get_elem(dy1_data, n, m).set_im(T(0.0));

        (*arr_dy2).get_elem(dy2_data, n, m).set_re(-1.0 * two_pi_Lx * two_pi_Lx * T(n * n));
        (*arr_dy2).get_elem(dy2_data, n, m).set_im(-1.0 * two_pi_Ly * two_pi_Ly * T(m * m));

        /////////////////////////////////////////////////////////////////////////////////////////////
        // n = nx/2+1 .. Nx-2
        for(n = geom_my21.get_nx() / 2 + 1; n < geom_my21.get_nx(); n++)
        {
            for(m = 0; m < geom_my21.get_my() - 1; m++)
            {
                (*arr_dy1).get_elem(dy1_data, n, m).set_re(two_pi_Lx * (T(n) - T(geom_my21.get_nx())));
                (*arr_dy1).get_elem(dy1_data, n, m).set_im(two_pi_Ly * T(m));

                (*arr_dy2).get_elem(dy2_data, n, m).set_re(-1.0 * two_pi_Lx * two_pi_Lx * (T(geom_my21.get_nx()) - T(n)) * (T(geom_my21.get_nx()) - T(n)) );
                (*arr_dy2).get_elem(dy2_data, n, m).set_im(-1.0 * two_pi_Ly * two_pi_Ly * T(m * m));
            }
            
            m = geom_my21.get_my() - 1;
            (*arr_dy1).get_elem(dy1_data, n, m).set_re(two_pi_Lx * (T(n) - T(geom_my21.get_nx())));
            (*arr_dy1).get_elem(dy1_data, n, m).set_im(T(0.0));

            (*arr_dy2).get_elem(dy2_data, n, m).set_re(-1.0 * two_pi_Lx * two_pi_Lx * (T(geom_my21.get_nx()) - T(n)) * (T(geom_my21.get_nx()) - T(n)) );
            (*arr_dy2).get_elem(dy2_data, n, m).set_im(-1.0 * two_pi_Ly * two_pi_Ly * T(m * m));
        }
    }
#endif // __CUDACC__

    }

}

#endif //UTILITY_H