/*
 * slab_cuda.h
 *
 */

#include "cufft.h"
#include "cuda_types.h"
#include "2dads_types.h"
#include "cuda_array2.h"
#include "slab_config.h"


class slab_cuda
{
    public:
        slab_cuda(slab_config);
        ~slab_cuda();

        bool init_dft();
        void finish_dft();
        void test_slab_config();

        void initialize();

        //void move_t(twodads::field_k_t, uint, uint);
        //void move_t(twodads::field_t, uint, uint);

        //void d_dx(twodads::field_k_t, field_k_t);
        //void d_dy(twodads::field_k_t, field_k_t);
        //void inv_laplace(twodads::field_k_t, twodads::field_k_t, uint);
        
        void advance();
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
        slab_config(config);
        const unsigned int Nx;
        const unsigned int My;
        const unsigned int tlevs;

        cuda_array<cuda::real_t> theta;
        cuda_array<cuda::cmplx_t> theta_hat;

        // DFT plans
        cufftHandle plan_r2c;
        cufftHandle plan_c2r;

        bool dft_is_initialized;

        cuda_array<cuda::real_t>* get_field_by_name(twodads::field_t);
        cuda_array<cuda::cmplx_t>* get_field_by_name(twodads::field_k_t);

};




