/*
 * slab_cuda.h
 *
 */

#include "cufft.h"
#include "cuda_types.h"
#include "2dads_types.h"
#include "cuda_array2.h"
#include "slab_config.h"
#include "spectral_kernels.h"

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

        void d_dx(twodads::field_k_t, twodads::field_k_t);
        void d_dy(twodads::field_k_t, twodads::field_k_t);
        void inv_laplace(twodads::field_k_t, twodads::field_k_t);
        
        void advance(twodads::field_k_t);
        void rhs_fun();
        void integrate_stiff(twodads::dyn_field_t, uint);

        void dft_r2c(twodads::field_t, twodads::field_k_t, uint);
        void dft_c2r(twodads::field_k_t, twodads::field_t, uint);

        //output_h5 slab_output;
        //void write_output(twodads::real_t);
        //void write_diagnostics(twodads::real_t);
        void dump_field(twodads::field_t);
        void dump_field(twodads::field_k_t);

    private:
        //slab_config(config);
        slab_config config;
        const uint Nx;
        const uint My;
        const uint tlevs;

        cuda_array<cuda::real_t> theta, theta_x, theta_y;
        cuda_array<cuda::real_t> tmp_array;
        cuda_array<cuda::cmplx_t> theta_hat, theta_x_hat, theta_y_hat;
        cuda_array<cuda::cmplx_t> tmp_array_hat;

        cuda_array<cuda::cmplx_t> theta_rhs_hat;
        rhs_fun_ptr theta_rhs_fun;
        //rhs_fun_ptr omega_rhs_fun;

        // DFT plans
        cufftHandle plan_r2c;
        cufftHandle plan_c2r;

        bool dft_is_initialized;

        cuda_array<cuda::real_t>* get_field_by_name(twodads::field_t);
        cuda_array<cuda::cmplx_t>* get_field_by_name(twodads::field_k_t);

        void theta_rhs_lin();
        void theta_rhs_log();
        void rhs_null();
        // Parameters for stiffly stable time integration
        twodads::stiff_params stiff_params_theta;
        twodads::stiff_params stiff_params_omega;
};

// End of file slab_cuda.h
