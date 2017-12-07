/*
 * Driver slab where the arrays support various boundary conditions
 */


#ifndef SLAB_BC_H
#define SLAB_BC_H

#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <map>
#include "2dads_types.h"
#include "cuda_array_bc_nogp.h"
#include "utility.h"
#include "dft_type.h"
#include "solvers.h"
#include "derivatives.h"
#include "integrators.h"
#include "slab_config.h"
#include "output.h"
#include "diagnostics.h"

#ifdef __CUDACC__
#include "cuda_types.h"
#endif //__CUDACC__


#ifndef __CUDACC__
#define LAMBDACALLER
#endif

#ifdef __CUDACC__
#define LAMBDACALLER __device__
#endif


class slab_bc
{
    /**
     .. cpp:namespace:: ::slab_bc

    */
    public:
        /**
         .. cpp:type:: value_t=twodads::real_t

         Use double precision.

        */
        using value_t = twodads::real_t;

        /**
         .. cpp:type:: cmplx_t=CuCmplx<twodads::real_t>

         Use custom CuCmplx<T> as the complex data type. Clang should std::cmplx<T>
         on the GPU. Future releases might change this.

        */
        using cmplx_t = CuCmplx<twodads::real_t>;

        /**
         .. cpp:type:: cmplx_ptr_t=CuCmplx<twodads::real_t>*

         Short hand notation for pointers to complex data.

        */
        using cmplx_ptr_t = CuCmplx<twodads::real_t>*;

#ifdef DEVICE
        using arr_real = cuda_array_bc_nogp<value_t, allocator_device>;
        using arr_cmpl = cuda_array_bc_nogp<cmplx_t, allocator_device>;
        using dft_t = cufft_object_t<value_t>;
        using deriv_t = deriv_fd_t<value_t, allocator_device>;
#endif //DEVICE

#ifdef HOST
        /**
         .. cpp:type:: arr_real = cuda_array_bc_nogp<value_t, allocator_host>

         Data type for real arrays.

        */
        using arr_real = cuda_array_bc_nogp<value_t, allocator_host>;

        /**
         .. cpp:type:: arr_cmpl = cuda_array_bc_nogp<cmplx_t, allocator_host>

         Data type for complex arrays.

        */
        using arr_cmpl = cuda_array_bc_nogp<cmplx_t, allocator_host>;

        /**
        .. cpp:type:: dft_t = fftw_object_t<value_t>

        Data type for DFTs.

        */
        using dft_t = fftw_object_t<value_t>;

        /**
         .. cpp:type deriv_t = deriv_fd_t<value_t, allocator_host>

         Data type for derivatives.

        */
        using deriv_t = deriv_fd_t<value_t, allocator_host>; 
#endif //HOST

        // typedef calls to functions that compute the implicit part for time integration.
        // These functions have 2 time indices as the argument.
        // 1: t_dst gives the index where the result is written to
        // 2: t_src gives the index where the input is if the array has more than one time index (i.e. the fields that are integrated)
        // for all other fields, theta_[xy], strmf, strmf_[xy] this defaults to 0.
        typedef void(slab_bc ::*rhs_func_ptr) (const size_t, const size_t); 

        /**
         .. cpp:function:: slab_bc(const twodads::slab_layout_t, const twodads::bvals_t<value_t>, const twodads::stiff_params_t)

         Construct a slab, explicitly from a slab_layout_t, boundary values, and stiff_params_t.

        */
        slab_bc(const twodads::slab_layout_t, const twodads::bvals_t<value_t>, const twodads::stiff_params_t);

        /**
         .. cpp:function:: slab_bc(const slab_config_js& cfg)

         :param const slab_config_js& cfg: Configuration for the slab.

         Construct a slab from a config object.

        */
        slab_bc(const slab_config_js& cfg);
        ~slab_bc();

        /**
         .. cpp:function:: void dft_r2c(const twodads::field_t, const size_t)

         :param const field_t fname: Name of the field to transform
         :param const size_t tidx: Time index for the input data.

         Compute a real to complex DFT of field at time level. Marks the field as transformed.

        */
        void dft_r2c(const twodads::field_t, const size_t);

        /**
         .. cpp:function:: void dft_c2r(const twodads::field_t, const size_t)

         :param const field_t fname: Name of the field to transform
         :param const size_t tidx: Time index at which to transform

         Compute complex to real DFt of field at time level. Unmark the field as transformed and normalize.

        */
        void dft_c2r(const twodads::field_t, const size_t);

        /**
         .. cpp:function:: void initialize()

         Initializes the slab according to the information in the slab_config_js object.
        
        */
        void initialize();

        /**
         .. cpp:function:: void invert_laplace(const twodads::field_t, const twodads::field_t, const size_t, const size_t)

         :param const field_t fname_src: Name of input field
         :param const field_t fname_dst: Name of output field
         :param const size_t t_src: Time index of input field
         :param const size_t t_ds: Time index of output field

         Inverts the laplace equation. This is a wrapper for the invert_laplace of derivs_t member.

        */
        void invert_laplace(const twodads::field_t, const twodads::field_t, const size_t, const size_t);

        /**
         .. cpp:function:: void d_dx(const twodads::field_t, const twodads::field_t, const size_t, const size_t, const size_t)

         Computes derivatives of a field in the x-direction. This is a wrapper for the dx member of the
         derivs_t member.

        */
        void d_dx(const twodads::field_t, const twodads::field_t, const size_t, const size_t, const size_t);

        /**
         .. cpp:function:: void d_dy(const twodads::field_t fname_src, const twodads::field_t fname_dst, const size_t order, const size_t, const size_t)

         Computes derivatives of a field in the y-direction. This is a wrapper for the dx member of the
         derivs_t member.

        */
        void d_dy(const twodads::field_t, const twodads::field_t, const size_t, const size_t, const size_t);

        /**
         .. cpp:function:: void integrate(const twodads::dyn_field_t, const size_t)

         Integrates a field in time with a given order. This is a wrapper for the integrate
         function of the tint member.

        */
        void integrate(const twodads::dyn_field_t, const size_t);

        /**
         .. cpp:function:: void update_real_fields(const size_t)

         Updates real fields after time integration.

        */
        void update_real_fields(const size_t);

        /**
         .. cpp:function:: void advance()

         Advances time index of dynamic fields.
        
        */
        void advance();

        /**
         .. cpp:function:: write_output(const size_t, const twodads::real_t)

         Writes the output specified in the slab_config_js member.

        */
        void write_output(const size_t, const twodads::real_t);

        /**
         .. cpp:function:: diagnose(const size_te, const twodads::real_t)

         Calls the diagnostic functions specified in the slab_config_js member.

        */
        void diagnose(const size_t, const twodads::real_t);

        arr_real* get_array_ptr(const twodads::field_t fname) const {return(get_field_by_name.at(fname));};

        const slab_config_js& get_config() {return(conf);};

        /**
         .. cpp:function:: void rhs(const size_t, const size_t)

         Calls the RHS function for the dynamic fields specified in the slab_config_js member.

        */
        void rhs(const size_t, const size_t);

        /**
         .. cpp:function void rhs_theta_null(const size_t, const size_t)

         RHS for diffusion equation. No explicit part. Should not be called directly but from void rhs(const size_t, const size_t).

        */
        void rhs_theta_null(const size_t, const size_t);

        /**
         .. cpp:function void rhs_theta_lin(const size_t, const size_t)

         RHS for linear interchange model, advection with electric drift. Should not be called directly but from void rhs(const size_t, const size_t).

        */
        void rhs_theta_lin(const size_t, const size_t);

        /**
         .. cpp:function void rhs_theta_log(const size_t, const size_t)

         RHS for logarithmic interchange model, advection with electric drift. Should not be called directly but from void rhs(const size_t, const size_t).

        */
        void rhs_theta_log(const size_t, const size_t);


        /**
         .. cpp:function void rhs_omega_null(const size_t, const size_t)

         RHS for diffusion equation. No explicit part. Should not be called directly but from void rhs(const size_t, const size_t).

        */
        void rhs_omega_null(const size_t, const size_t);

        /**
         .. cpp:function void rhs_theta_null(const size_t, const size_t)

         RHS for interchange model. Should not be called directly but from void rhs(const size_t, const size_t).

        */
        void rhs_omega_ic(const size_t, const size_t);

        /**
         .. cpp:function void rhs_tau_null(const size_t, const size_t)

         RHS for diffusion equation. No explicit part. Should not be called directly but from void rhs(const size_t, const size_t).

        */
        void rhs_tau_null(const size_t, const size_t);

        /**
         .. cpp:function void rhs_tau_log(const size_t, const size_t)

         RHS for logarithmic interchange model, advection with electric drift. Should not be called directly but from void rhs(const size_t, const size_t).

        */
        void rhs_tau_log(const size_t, const size_t);
    private:

        const slab_config_js conf;
        output_h5_t output;
        diagnostic_t diagnostic;
        dft_object_t<twodads::real_t>* myfft;

#ifdef DEVICE
        deriv_base_t<value_t, allocator_device>* my_derivs;
        integrator_base_t<value_t, allocator_device>* tint_theta;
        integrator_base_t<value_t, allocator_device>* tint_omega;
        integrator_base_t<value_t, allocator_device>* tint_tau;
#endif //DEVICE
#ifdef HOST
        deriv_base_t<value_t, allocator_host>* my_derivs;
        integrator_base_t<value_t, allocator_host>* tint_theta;
        integrator_base_t<value_t, allocator_host>* tint_omega;
        integrator_base_t<value_t, allocator_host>* tint_tau;
#endif //HOST

        arr_real theta;
        arr_real theta_x;
        arr_real theta_y;
        arr_real omega;
        arr_real omega_x;
        arr_real omega_y;
        arr_real tau;
        arr_real tau_x;
        arr_real tau_y;
        arr_real tmp;
        arr_real strmf;
        arr_real strmf_x;
        arr_real strmf_y;
        arr_real theta_rhs;
        arr_real omega_rhs;
        arr_real tau_rhs;

        const std::map<twodads::field_t, arr_real*> get_field_by_name;
        const std::map<twodads::dyn_field_t, arr_real*> get_dfield_by_name;
        const std::map<twodads::output_t, arr_real*> get_output_by_name;
        
        rhs_func_ptr theta_rhs_func;
        rhs_func_ptr omega_rhs_func;
        rhs_func_ptr tau_rhs_func;

        static std::map<twodads::rhs_t, rhs_func_ptr> rhs_func_map;

        static std::map<twodads::rhs_t, rhs_func_ptr> create_rhs_func_map()
        {
            std::map<twodads::rhs_t, rhs_func_ptr> my_map;
            my_map[twodads::rhs_t::rhs_theta_null] = &slab_bc::rhs_theta_null;
            my_map[twodads::rhs_t::rhs_theta_lin]  = &slab_bc::rhs_theta_lin;
            my_map[twodads::rhs_t::rhs_theta_log]  = &slab_bc::rhs_theta_log;
            my_map[twodads::rhs_t::rhs_omega_null] = &slab_bc::rhs_omega_null;
            my_map[twodads::rhs_t::rhs_omega_ic]   = &slab_bc::rhs_omega_ic;
            my_map[twodads::rhs_t::rhs_tau_null]   = &slab_bc::rhs_tau_null;
            my_map[twodads::rhs_t::rhs_tau_log]    = &slab_bc::rhs_tau_log;
            return(my_map);
        }

        /**
         .. cpp:namespace-pop::

        */
};

#endif // SLAB_BC_H
