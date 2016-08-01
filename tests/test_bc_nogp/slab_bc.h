/*
 * Test a slab where the arrays support various boundary conditions
 *
 */


#ifndef SLAB_BC_H
#define SLAB_BC_H

#include <fstream>
#include <string>
#include <map>
#include <cufft.h>
#include "cuda_types.h"
#include "cuda_array_bc_nogp.h"
#include "derivatives.h"


namespace test_ns{
    enum class field_t {arr1, arr1_x, arr1_y, arr2, arr2_x, arr2_y, arr3, arr3_x, arr3_y};
}

class slab_bc
{
    public:
        using value_t = cuda::real_t;

        typedef cuda_array_bc_nogp<my_allocator_device<cuda::cmplx_t>> cuda_arr_cmpl;
        typedef cuda_array_bc_nogp<my_allocator_device<cuda::real_t>> cuda_arr_real;

        slab_bc(const cuda::slab_layout_t, const cuda::bvals_t<real_t>);
        ~slab_bc();

        void init_dft();
        void dft_r2c(const test_ns::field_t, const size_t);
        void dft_c2r(const test_ns::field_t, const size_t);
        void finish_dft();

        void initialize_invlaplace(const test_ns::field_t);
        void initialize_sine(const test_ns::field_t);
        void initialize_arakawa(const test_ns::field_t, const test_ns::field_t);

        void invert_laplace(const test_ns::field_t, const test_ns::field_t, const size_t);
        void d_dx_dy(const size_t);

        void print_field(const test_ns::field_t) const;
        void print_field(const test_ns::field_t, const string) const;

    private:
        const size_t Nx;
        const size_t My;
        const size_t tlevs;

        const cuda::bvals_t<cuda::real_t> boundaries;
        const cuda::slab_layout_t geom;

        derivs<my_allocator_device<cuda::real_t>> der;

        cuda_arr_real arr1;
        cuda_arr_real arr1_x;
        cuda_arr_real arr1_y;

        cuda_arr_real arr2;
        cuda_arr_real arr2_x;
        cuda_arr_real arr2_y;

        cuda_arr_real arr3;
        cuda_arr_real arr3_x;
        cuda_arr_real arr3_y;

        cufftHandle plan_r2c;
        cufftHandle plan_c2r;

        const std::map<test_ns::field_t, cuda_arr_real*> get_field_by_name;

        bool dft_is_initialized;
};

#endif // SLAB_BC_H
