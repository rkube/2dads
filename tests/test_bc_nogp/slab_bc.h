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
#include "2dads_types.h"
#include "cuda_types.h"
#include "cuda_array_bc_nogp.h"
#include "dft_type.h"
//#include "solvers.h"
//#include "derivatives.h"
//#include "integrators.h"


namespace test_ns{
    enum class field_t {arr1, arr1_x, arr1_y, arr2, arr2_x, arr2_y, arr3, arr3_x, arr3_y};
}

class slab_bc
{
    public:
        using value_t = twodads::real_t;

        using cuda_arr_real = cuda_array_bc_nogp<twodads::real_t, allocator_device>;
        using cuda_arr_cmpl = cuda_array_bc_nogp<twodads::cmplx_t, allocator_device>;

        slab_bc(const twodads::slab_layout_t, const twodads::bvals_t<twodads::real_t>, const twodads::stiff_params_t);
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
        void print_field(const test_ns::field_t, const std::string) const;

        void initialize_tint(const test_ns::field_t);
        void integrate(const test_ns::field_t);

        cuda_arr_real* get_array_ptr(const test_ns::field_t fname) const {return(get_field_by_name.at(fname));};
        inline twodads::slab_layout_t get_geom() const {return(geom);};
        inline twodads::bvals_t<twodads::real_t> get_bvals() const {return(boundaries);};
    private:
        const size_t Nx;
        const size_t My;
        const size_t tlevs;

        const twodads::bvals_t<twodads::real_t> boundaries;
        const twodads::slab_layout_t geom;

        //solvers::cublas_handle_t cublas_handle;
        //solvers::cusparse_handle_t cusparse_handle;

        //derivs<allocator_device<twodads::real_t>> der;
        dft_object_t<twodads::real_t>* myfft;
        //integrator<my_allocator_device<twodads::real_t>>* tint;

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
