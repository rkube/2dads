///
/// Two-dimensional slab consisting of cuda_arrays for all variables
/// Dynamic variables are omega and theta
///


#ifndef SLAB_CUDA_H
#define SLAB_CUDA_H

#include <string>
#include <fstream>
#include <map>
#include "cufft.h"
#include "cuda_types.h"
#include "2dads_types.h"
#include "cuda_array4.h"
#include "slab_config.h"
#include "initialize.h"
#include "cufft.h"



class slab_cuda
{
    public:
        typedef void (slab_cuda::*rhs_fun_ptr)(uint);
        typedef cuda_array<cuda::cmplx_t> cuda_arr_cmplx;
        typedef cuda_array<cuda::real_t> cuda_arr_real;
        slab_cuda(const slab_config&); ///< Standard constructors
        ~slab_cuda();

        bool init_dft();  ///< Initialize cuFFT
        void finish_dft(); ///< Clean-up for cuFFT
        void test_slab_config(); ///< Test if the slab_configuration gives a good slab

        /// @brief initialize slab to start with time integration 
        /// @detailed After running initialize, the following things are set up:
        /// @detailed theta, omega, strmf: Either by function specified through init_function string in 
        /// @detailed input.ini or by taking the iDFT of their respective spectral field
        /// @detailed theta_hat, omega_hat: non-zero values at last time index, tlev-1
        /// @detailed omega_rhs_hat, theta_rhs_hat: computed as specified by omega/theta_rhs
        /// @detailed non-zero value at last time index, tlev-2
        void initialize(); 

        
        /// @brief Move data from time level t_src to t_dst
        /// @param fname field name 
        /// @param t_dst destination time index
        /// @param t_src source time index
        void move_t(const twodads::field_k_t, const uint, const uint);
        /// @brief Copy data from time level t_src to t_dst
        /// @param fname field name 
        /// @param t_dst destination time index
        /// @param t_src source time index
        void copy_t(const twodads::field_k_t, const uint, const uint);
        /// @brief Set fname to a constant value at time index tlev
        /// @param fname field name 
        /// @param val constant complex number
        /// @param t_src time index
        void set_t(const twodads::field_k_t, const cuda::cmplx_t, const uint);
        /// @brief Set fname to a constant value at time index tlev
        /// @param fname field name 
        /// @param val constant complex number
        /// @param t_src time index
        void set_t(const twodads::field_t, const cuda::real_t, const uint);

        /// @brief compute spectral derivative in x direction
        void d_dx(const twodads::field_k_t, const twodads::field_k_t, const uint);
        /// @brief compute spectral derivative in y direction
        void d_dy(const twodads::field_k_t, const twodads::field_k_t, const uint);
        /// @brief Solve laplace equation in k-space
        void inv_laplace(const twodads::field_k_t, const twodads::field_k_t, const uint); ///< Invert Laplace operators

        /// Debug functions that only enumerate the array by row and col number
        /// format: cmplx: (sector * 1000 + col, row)
        ///         real : (1000*col + row)
        void enumerate(const twodads::field_k_t f_name);
        void enumerate(const twodads::field_t f_name);
        void d_dx_enumerate(const twodads::field_k_t, uint); ///< Compute x derivative
        void d_dy_enumerate(const twodads::field_k_t, uint); ///< Compute y derivative
        // Solve laplace equation in k-space
        void inv_laplace_enumerate(const twodads::field_k_t, uint); ///< Invert Laplace operators
        
        /// @brief Advance all member fields with multiple time levels: theta_hat, omega_hat, theta_rhs_hat and omega_rhs_hat
        void advance(); 
        /// @brief Call RHS_fun pointers
        /// @param t_src The most current time level, only important for first transient time steps
        void rhs_fun(const uint);
        /// @brief Update real fields theta, theta_x, theta_y, etc.
        /// @param tlev: The time level used from theta_hat, omega_hat as input for inverse DFT
        void update_real_fields(const uint);


        /// @brief calls copy_device_to_host for the specified field
        void copy_device_to_host(const twodads::output_t);

        void integrate_stiff(const twodads::field_k_t, const uint); ///< Time step
        void integrate_stiff_ky0(const twodads::field_k_t, const uint); ///< Time integration of modes with ky=0
        void integrate_stiff_enumerate(twodads::field_k_t, const uint); ///< Only enumerate modes
        void integrate_stiff_debug(const twodads::field_k_t, const uint, const uint, const uint); ///< Integrate, with full debugging output
        /// @brief execute DFT real to complex
        /// @param fname_r real field type
        /// @param fname_c complex field type
        /// @param t_src time index of complex field used as target for DFT
        void dft_r2c(const twodads::field_t, const twodads::field_k_t, const uint);
        /// @brief execute iDFT (complex to real) and normalize the resulting real field
        /// @param fname_c complex field type
        /// @param fname_r real field type
        /// @param t time index of complex field used as source for iDFT
        void dft_c2r(const twodads::field_k_t, const twodads::field_t, const uint);

        /// @brief print real field on terminal
        void print_field(const twodads::field_t) const;
        /// @brief print real field on to ascii file
        void print_field(const twodads::field_t, const string) const;
        /// @brief print complex field to terminal
        void print_field(const twodads::field_k_t) const;
        /// @brief print complex field to ascii file
        void print_field(const twodads::field_k_t, string) const;

        /// @brief Copy data from a real field to a buffer in host memory
        /// @param twodads::field_t fname: Name of the field to be copied
        /// @param cuda_array<T> buffer: buffer in which array data is to be copied
        void get_data_host(const twodads::field_t, cuda_array<twodads::real_t>&) const;
        void get_data_host(const twodads::field_t, cuda::real_t*, const uint, const uint) const;

        /// @brief Copy data from a real field to a buffer in device memory
        /// @param twodads::field_t fname: Name of the field to be copied
        /// @param cuda_array<real_t>* buffer: Cuda array in which to copy data
        /// @detailed: Calls copy(0, buffer, 0) method from cuda_array
        void get_data_device(const twodads::field_t, cuda::real_t*, const uint, const uint) const;

        /// @brief get address of a cuda_array member with updated host data
        /// @detailed Call to get_array_ptr copies device data to host
        cuda_arr_real*  get_array_ptr(const twodads::output_t fname);
        cuda_arr_real*  get_array_ptr(const twodads::field_t fname);
        /// @brief Print the addresses of member variables in host memory
        void print_address() const;
        /// @brief  Print the grid sizes used for cuda kernel calls
        void print_grids() const;
        // Output methods
        friend class output_h5;
        //friend class diagnostics;

    private:
        slab_config config; ///< slab configuration
        const uint Nx; ///< Number of discretization points in x-direction
        const uint My; ///< Number of discretization points in y=direction
        const uint tlevs; ///< Number of time levels stored, order of time integration + 1

        cuda_array<cuda::real_t> theta, theta_x, theta_y; ///< Real fields assoc. with theta
        cuda_array<cuda::real_t> omega, omega_x, omega_y; ///< Real fields assoc. with omega
        cuda_array<cuda::real_t> strmf, strmf_x, strmf_y; ///< Real fields assoc.with stream function
        cuda_array<cuda::real_t> tmp_array; ///< temporary data
        cuda_array<cuda::real_t> theta_rhs, omega_rhs; ///< Real fields for RHS, for debugging

        cuda_array<cuda::cmplx_t> theta_hat, theta_x_hat, theta_y_hat; ///< Fourier fields assoc. with theta
        cuda_array<cuda::cmplx_t> omega_hat, omega_x_hat, omega_y_hat; ///< Fourier fields assoc. with omega
        cuda_array<cuda::cmplx_t> strmf_hat, strmf_x_hat, strmf_y_hat; ///< Fourier fields assoc. with strmf
        cuda_array<cuda::cmplx_t> tmp_array_hat; ///< Temporary data, Fourier field

        cuda_array<cuda::cmplx_t> theta_rhs_hat; ///< non-linear RHS for time integration of theta
        cuda_array<cuda::cmplx_t> omega_rhs_hat; ///< non-linear RHS for time integration of omea

        // Arrays that store data on CPU for diagnostic functions
        rhs_fun_ptr theta_rhs_func; ///< Evaluate explicit part for time integrator
        rhs_fun_ptr omega_rhs_func; ///< Evaluate explicit part for time integrator

        // DFT plans
        cufftHandle plan_r2c; ///< Plan used for all D2Z DFTs used by cuFFT
        cufftHandle plan_c2r; ///< Plan used for all Z2D iDFTs used by cuFFT

        bool dft_is_initialized; ///< True if cuFFT is initialized
        //output_h5 slab_output; ///< Handels internals of output to hdf5
        //diagnostics slab_diagnostic; ///< Handles internals of diagnostic output

        // Parameters for stiff time integration
        const cuda::stiff_params_t stiff_params; ///< Coefficients for time integration routine
        const cuda::slab_layout_t slab_layout; ///< Slab layout passed to cuda kernels and initialization routines

        ///@brief Block and grid dimensions for kernels operating on My*Nx arrays.
        ///@brief For kernels where every element is treated alike
        const dim3 block_my_nx;
        const dim3 grid_my_nx;

        ///@ brief Block and grid dimensions for kernels operating on My * Nx/2+1 arrays
        const dim3 block_my_nx21;
        const dim3 grid_my_nx21;

        /// @brief Block and grid dimensions for arrays My * Nx/2+1
        /// @brief Row-like blocks, spanning 0..Nx/2 in multiples of cuda::blockdim_nx
        /// @brief effectively wasting blockdim_nx-1 threads in the last call. But memory is
        /// @brief coalesced :)
       
        /// @brief 
        const dim3 block_nx21; ///< blocksize is (cuda::blockdim_nx, 1)
        const dim3 grid_nx21_sec1; ///< Sector 1: ky > 0, kx < Nx / 2
        const dim3 grid_nx21_sec2; ///< Sector 2: ky < 0, kx < Nx / 2
        const dim3 grid_nx21_sec3; ///< Sector 3: ky > 0, kx = Nx / 2
        const dim3 grid_nx21_sec4; ///< Sector 4: ky < 0, kx = Nx / 2

        dim3 grid_dx_half; ///< All modes with kx < Nx / 2
        dim3 grid_dx_single; ///< All modes with kx = Nx / 2

        dim3 grid_dy_half; ///< My/2 rows, all columns
        dim3 grid_dy_single; ///< 1 row, all columns


        /// @brief Grid dimensions for access on all {kx, 0} modes, stored in the first Nx/2+1 elements of an array
        dim3 grid_ky0;
        
        //make rhs_func_map static, because all slabs have the same RHS functions
        static std::map<twodads::rhs_t, rhs_fun_ptr> rhs_func_map;
        const std::map<twodads::field_k_t, cuda_arr_cmplx*> rhs_array_map;
        const std::map<twodads::field_k_t, cuda_arr_cmplx*> get_field_k_by_name;
        const std::map<twodads::field_t, cuda_arr_real*> get_field_by_name;
        const std::map<twodads::output_t, cuda_arr_real*> get_output_by_name;


        cuda::real_t* d_ss3_alpha; ///< Coefficients for implicit part of time integration
        cuda::real_t* d_ss3_beta; ///< Coefficients for explicit part of time integration
        void theta_rhs_ns(uint); ///< Navier-Stokes 
        void theta_rhs_lin(uint); ///< Small amplitude blob
        void theta_rhs_log(uint); ///< Arbitrary amplitude blob
        void theta_rhs_hw(uint); ///< Hasegawa-Wakatani model 
        void theta_rhs_hwmod(uint); ///< Modified Hasegawa-Wakatani model
        void theta_rhs_null(uint); ///< Zero explicit term

        void omega_rhs_ns(uint); ///< Navier Stokes
        void omega_rhs_hw(uint); ///< Hasegawa-Wakatani model
        void omega_rhs_hwmod(uint); ///< Modified Hasegawa-Wakatani model
        void omega_rhs_hwzf(uint); /// Modified Hasegawa-Wakatani, supress zonal flows
        void omega_rhs_ic(uint); ///< Interchange turbulence
        void omega_rhs_null(uint); ///< Zero explicit term

        static map<twodads::rhs_t, rhs_fun_ptr> create_rhs_func_map()
        {
            map<twodads::rhs_t, rhs_fun_ptr> my_map;
            my_map[twodads::rhs_t::theta_rhs_ns] = &slab_cuda::theta_rhs_ns;
            my_map[twodads::rhs_t::theta_rhs_lin] = &slab_cuda::theta_rhs_lin;
            my_map[twodads::rhs_t::theta_rhs_log] = &slab_cuda::theta_rhs_log;
            my_map[twodads::rhs_t::theta_rhs_null] = &slab_cuda::theta_rhs_null;
            my_map[twodads::rhs_t::theta_rhs_hw] =  &slab_cuda::theta_rhs_hw;
            my_map[twodads::rhs_t::theta_rhs_hwmod] = &slab_cuda::theta_rhs_hwmod;
            my_map[twodads::rhs_t::omega_rhs_ns] = &slab_cuda::omega_rhs_ns;
            my_map[twodads::rhs_t::omega_rhs_hw] = &slab_cuda::omega_rhs_hw;
            my_map[twodads::rhs_t::omega_rhs_hwmod] = &slab_cuda::omega_rhs_hwmod;
            my_map[twodads::rhs_t::omega_rhs_hwzf] = &slab_cuda::omega_rhs_hwzf;
            my_map[twodads::rhs_t::omega_rhs_null] = &slab_cuda::omega_rhs_null;
            my_map[twodads::rhs_t::omega_rhs_ic] = &slab_cuda::omega_rhs_ic;
            return (my_map);
        }
};

#endif //SLAB_CUDA_H
// End of file slab_cuda.h
