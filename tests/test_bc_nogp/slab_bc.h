/*
 * Test a slab where the arrays support various boundary conditions
 *
 */


#ifndef SLAB_BC_H
#define SLAB_BC_H

#include <string>
#include <cufft.h>
#include "derivatives.h"
#include "cuda_types.h"
#include "cuda_array_bc_nogp.h"


class slab_bc
{
    public:
        typedef cuda_array_bc_nogp<cuda::cmplx_t> cuda_arr_cmpl;
        typedef cuda_array_bc_nogp<cuda::real_t> cuda_arr_real;

        slab_bc(const cuda::slab_layout_t, const cuda::bvals_t<real_t>);
        ~slab_bc();

        void init_dft();
        void dft_r2c(const size_t);
        void dft_c2r(const size_t);
        void finish_dft();

        void d_dx_dy(const size_t);

        void dump_arr1();
        void dump_arr1x();
        void dump_arr1y();

    private:
        const size_t Nx;
        const size_t My;
        const size_t tlevs;

        const cuda::bvals_t<cuda::real_t> boundaries;
        const cuda::slab_layout_t geom;

        cuda_array_bc_nogp<cuda::real_t> arr1;
        cuda_array_bc_nogp<cuda::real_t> arr1_x;
        cuda_array_bc_nogp<cuda::real_t> arr1_y;

        cufftHandle plan_r2c;
        cufftHandle plan_c2r;

        bool dft_is_initialized;
};

#endif // SLAB_BC_H
