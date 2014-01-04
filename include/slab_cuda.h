/*
 * slab_cuda.h
 *
 */

#include "cufft.h"
#include "cuda_types.h"
#include "2dads_types.h"
#include "cuda_array2.h"
#include "slab_config.h"
#include "output.h"

class slab_cuda
{
    public:
        typedef void (slab_cuda::*rhs_fun_ptr)();
        slab_cuda(slab_config);
        ~slab_cuda();

        bool init_dft();
        void finish_dft();
        void test_slab_config();

        void initialize();

        void move_t(twodads::field_k_t, uint, uint);
        void move_t(twodads::field_t, uint, uint);
        void copy_t(twodads::field_k_t, uint, uint);
        void set_t(twodads::field_k_t, cuda::cmplx_t, uint);

        // compute spectral derivative
        void d_dx(twodads::field_k_t, twodads::field_k_t);
        void d_dy(twodads::field_k_t, twodads::field_k_t);
        // Solve laplace equation in k-space
        void inv_laplace(twodads::field_k_t, twodads::field_k_t, uint);
        
        // Advance all fields with multiple time levels
        void advance();
        // Compute RHS function into tlev0 of theta_rhs_hat, omega_rhs_hat
        void rhs_fun();
        // Compute new theta_hat, omega_hat into tlev0.
        void integrate_stiff(twodads::dyn_field_t, uint);

        // Carry out DFT
        void dft_r2c(twodads::field_t, twodads::field_k_t, uint);
        void dft_c2r(twodads::field_k_t, twodads::field_t, uint);

        void dump_field(twodads::field_t);
        void dump_field(twodads::field_k_t);

        // Output methods
        // Make output_h5 a pointer since we deleted the default constructor
        void write_output(twodads::real_t);
        void write_diagnostics(twodads::real_t);

        void dump_address();
    private:
        slab_config config;
        const uint Nx;
        const uint My;
        const uint tlevs;

        cuda_array<cuda::real_t> theta, theta_x, theta_y;
        cuda_array<cuda::real_t> omega, omega_x, omega_y;
        cuda_array<cuda::real_t> strmf, strmf_x, strmf_y;
        cuda_array<cuda::real_t> tmp_array;

        cuda_array<cuda::cmplx_t> theta_hat, theta_x_hat, theta_y_hat;
        cuda_array<cuda::cmplx_t> omega_hat, omega_x_hat, omega_y_hat;
        cuda_array<cuda::cmplx_t> strmf_hat, strmf_x_hat, strmf_y_hat;
        cuda_array<cuda::cmplx_t> tmp_array_hat;

        cuda_array<cuda::cmplx_t> theta_rhs_hat;
        cuda_array<cuda::cmplx_t> omega_rhs_hat;
        rhs_fun_ptr theta_rhs_fun;
        rhs_fun_ptr omega_rhs_fun;

        // DFT plans
        cufftHandle plan_r2c;
        cufftHandle plan_c2r;

        bool dft_is_initialized;
        output_h5 slab_output;

        // Parameters for stiff time integration
        const cuda::stiff_params_t stiff_params;
        const cuda::slab_layout_t slab_layout;
        cuda::cmplx_t* d_ss3_alpha;
        cuda::cmplx_t* d_ss3_beta;

        // Block and grid dimensions for arrays Nx * My/2+1
        dim3 block_my21_sec1;
        dim3 block_my21_sec2;
        dim3 grid_my21_sec1;
        dim3 grid_my21_sec2;
        
        // Block and grid sizes for inv_lapl and integrate_stiff kernels
        dim3 block_sec12;
        dim3 grid_sec1;
        dim3 grid_sec2;

        dim3 block_sec3;
        dim3 block_sec4;
        dim3 grid_sec3;
        dim3 grid_sec4;

        // Get cuda_array corresponding to field type, real field
        cuda_array<cuda::real_t>* get_field_by_name(twodads::field_t);
        // Get cuda_array corresponding to field type, complex field
        cuda_array<cuda::cmplx_t>* get_field_by_name(twodads::field_k_t);
        // Get cuda_array corresponding to output field type 
        cuda_array<cuda::real_t>* get_field_by_name(twodads::output_t);
        // Get dynamic field corresponding to field name for time integration
        cuda_array<cuda::cmplx_t>* get_field_by_name(twodads::dyn_field_t);
        // Get right-hand side for time integration corresponding to dynamic field 
        cuda_array<cuda::cmplx_t>* get_rhs_by_name(twodads::dyn_field_t);

        //cuda::stiff_params stiff_params_omega;
        void theta_rhs_lin();
        void theta_rhs_log();
        void theta_rhs_null();

        void omega_rhs_lin();
        //void omega_rhs_hw();
        //void omega_rhs_hwmod();
        void omega_rhs_null();
        void omega_rhs_ic();
        // Parameters for stiffly stable time integration
        // Time-integration parameters on device
        //cuda::real_t** d_alpha;
        //cuda::real_t** d_beta;
        //cuda::stiff_params d_stiff_params_theta;
        //cuda::stiff_params d_stiff_params_omega;
};

// End of file slab_cuda.h
