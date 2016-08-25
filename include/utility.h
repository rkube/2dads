#ifndef UTILITY_H
#define UTILITY_H

#include "cuda_array_bc_nogp.h"


namespace utility
{
    template <typename T>
    void print(cuda_array_bc_nogp<T, allocator_host>& vec, const size_t tlev, std::ostream& os)
    {
        address_t<T>* address = vec.get_address_ptr();
        for(size_t n = 0; n < vec.get_geom().get_nx(); n++)
        {
            for(size_t m = 0; m < vec.get_geom().get_my(); m++)
            {
                os << std::setw(twodads::io_w) << std::setprecision(twodads::io_p) << std::fixed << (*address)(vec.get_tlev_ptr(tlev), n, m) << "\t";
            }
            os << std::endl;
        }
    }


    template <typename T>
    void print(const cuda_array_bc_nogp<T, allocator_host>& vec, const size_t tlev, std::string fname)
    {
        std::ofstream os(fname, std::ofstream::trunc);
        address_t<T>* address = vec.get_address_ptr();
        for(size_t n = 0; n < vec.get_geom().get_nx(); n++)
        {
            for(size_t m = 0; m < vec.get_geom().get_my(); m++)
            {
                os << std::setw(twodads::io_w) << std::setprecision(twodads::io_p) << std::fixed << (*address)(vec.get_tlev_ptr(tlev), n, m) << "\t";
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
        const dim3 grid(vec.get_grid());
        const dim3 block(vec.get_block());
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
        device :: kernel_apply<<<grid, block>>>(device_copy,
                                                [=] __device__ (T in, size_t n, size_t m, twodads::slab_layout_t geom ) -> T 
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



}

#endif //UTILITY_H