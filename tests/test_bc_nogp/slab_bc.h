/*
 * Test a slab where the arrays support various boundary conditions
 *
 */


#ifndef SLAB_BC_H
#define SLAB_BC_H

#include <fstream>
#include <string>
#include <map>
#include "2dads_types.h"
#include "cuda_array_bc_nogp.h"
#include "utility.h"
#include "dft_type.h"
#include "solvers.h"
#include "derivatives.h"
#include "integrators.h"

#ifdef __CUDACC__
#include "cuda_types.h"
#endif //__CUDACC__


namespace test_ns{
    enum class field_t {arr1, arr1_x, arr1_y, arr2, arr2_x, arr2_y, arr3, arr3_x, arr3_y};
}

#ifndef __CUDACC__
#define LAMBDACALLER
#endif

#ifdef __CUDACC__
#define LAMBDACALLER __device__
#endif


class slab_bc
{
    public:
        using value_t = twodads::real_t;
        using cmplx_t = CuCmplx<twodads::real_t>;
        using cmplx_ptr_t = CuCmplx<twodads::real_t>*;

#ifdef DEVICE
        using cuda_arr_real = cuda_array_bc_nogp<twodads::real_t, allocator_device>;
        using cuda_arr_cmpl = cuda_array_bc_nogp<twodads::cmplx_t, allocator_device>;
        using dft_library_t = cufft_object_t<twodads::real_t>;
#endif //DEVICE
#ifdef HOST
        using cuda_arr_real = cuda_array_bc_nogp<twodads::real_t, allocator_host>;
        using cuda_arr_cmpl = cuda_array_bc_nogp<twodads::cmplx_t, allocator_host>;
        using dft_library_t = fftw_object_t<twodads::real_t>;
#endif //HOST

        slab_bc(const twodads::slab_layout_t, const twodads::bvals_t<twodads::real_t>, const twodads::stiff_params_t);
        ~slab_bc();

        void dft_r2c(const test_ns::field_t, const size_t);
        void dft_c2r(const test_ns::field_t, const size_t);

        void initialize_invlaplace(const test_ns::field_t fname, const size_t tlev=0);
        void initialize_sine(const test_ns::field_t fnam, const size_t tlev=0);
        void initialize_arakawa(const test_ns::field_t fname1, const test_ns::field_t fname2, const size_t tlev=0);
        void initialize_derivatives(const test_ns::field_t fname1, const test_ns::field_t fname2, const size_t tlev=0);
        void initialize_dfttest(const test_ns::field_t fname, const size_t tlev=0);
        void initialize_gaussian(const test_ns::field_t fname, const size_t tlev=0);

        void invert_laplace(const test_ns::field_t, const test_ns::field_t, const size_t, const size_t);

        void d_dx(const test_ns::field_t, const test_ns::field_t, const size_t, const size_t, const size_t);
        void d_dy(const test_ns::field_t, const test_ns::field_t, const size_t, const size_t, const size_t);
        void arakawa(const test_ns::field_t, const test_ns::field_t, const test_ns::field_t, const size_t, const size_t);

        void integrate(const test_ns::field_t, const size_t);

        void advance();

        cuda_arr_real* get_array_ptr(const test_ns::field_t fname) const {return(get_field_by_name.at(fname));};
        inline twodads::slab_layout_t get_geom() const {return(geom);};
        inline twodads::bvals_t<twodads::real_t> get_bvals() const {return(boundaries);};
        inline twodads::stiff_params_t get_tint_params() const {return(tint_params);};
    private:

        const twodads::bvals_t<twodads::real_t> boundaries;
        const twodads::slab_layout_t geom;
        const twodads::stiff_params_t tint_params;

        dft_object_t<twodads::real_t>* myfft;

#ifdef DEVICE
        deriv_t<twodads::real_t, allocator_device> my_derivs;
        integrator_karniadakis<twodads::real_t, allocator_device> tint;
#endif //DEVICE
#ifdef HOST
        deriv_t<twodads::real_t, allocator_host> my_derivs;
        integrator_karniadakis<twodads::real_t, allocator_host> tint;
#endif //HOST

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
