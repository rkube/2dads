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
#include "integrators.h"


namespace test_ns{
    enum class field_t {arr1, arr1_x, arr1_y, arr2, arr2_x, arr2_y, arr3, arr3_x, arr3_y};
}

class slab_bc
{
    public:
        using value_t = cuda::real_t;

        using cuda_arr_real = cuda_array_bc_nogp<my_allocator_device<cuda::real_t>>;
        using cuda_arr_cmpl = cuda_array_bc_nogp<my_allocator_device<cuda::cmplx_t>>;

        slab_bc(const cuda::slab_layout_t, const cuda::bvals_t<real_t>);
        ~slab_bc();

        void dft_r2c(const test_ns::field_t, const size_t);
        void dft_c2r(const test_ns::field_t, const size_t);

        void initialize_invlaplace(const test_ns::field_t);
        void initialize_sine(const test_ns::field_t);
        void initialize_arakawa(const test_ns::field_t, const test_ns::field_t);
        void initialize_derivatives(const test_ns::field_t, const test_ns::field_t);
        void initialize_dfttest(const test_ns::field_t);
        void initialize_gaussian(const test_ns::field_t);

        void invert_laplace(const test_ns::field_t, const test_ns::field_t, const size_t, const size_t);

        void d_dx(const test_ns::field_t, const test_ns::field_t, const size_t, const size_t, const size_t);
        void d_dy(const test_ns::field_t, const test_ns::field_t, const size_t, const size_t, const size_t);
        void arakawa(const test_ns::field_t, const test_ns::field_t, const test_ns::field_t, const size_t, const size_t);

        void print_field(const test_ns::field_t) const;
        void print_field(const test_ns::field_t, const string) const;

        void initialize_tint();
        void integrate(const test_ns::field_t);

        cuda_arr_real* get_array_ptr(const test_ns::field_t fname) const {return(get_field_by_name.at(fname));};
        inline cuda::slab_layout_t get_geom() const {return(geom);};
        inline cuda::bvals_t<cuda::real_t> get_bvals() const {return(boundaries);};
    private:
        const size_t Nx;
        const size_t My;
        const size_t tlevs;

        const cuda::bvals_t<cuda::real_t> boundaries;
        const cuda::slab_layout_t geom;

        derivs<my_allocator_device<cuda::real_t>> der;
        dft_object_t<cuda::real_t> myfft;
        integrator<my_allocator_device<cuda::real_t>>* tint;

        cuda_arr_real arr1;
        cuda_arr_real arr1_x;
        cuda_arr_real arr1_y;

        cuda_arr_real arr2;
        cuda_arr_real arr2_x;
        cuda_arr_real arr2_y;

        cuda_arr_real arr3;
        cuda_arr_real arr3_x;
        cuda_arr_real arr3_y;

        const std::map<test_ns::field_t, cuda_arr_real*> get_field_by_name;
};

#endif // SLAB_BC_H
