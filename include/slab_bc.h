/*
 * Test a slab where the arrays support various boundary conditions
 *
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
    public:
        using value_t = twodads::real_t;
        using cmplx_t = CuCmplx<twodads::real_t>;
        using cmplx_ptr_t = CuCmplx<twodads::real_t>*;

#ifdef DEVICE
        using arr_real = cuda_array_bc_nogp<value_t, allocator_device>;
        using arr_cmpl = cuda_array_bc_nogp<cmplx_t, allocator_device>;
        using dft_t = cufft_object_t<value_t>;
        using deriv_t = deriv_fd_t<value_t, allocator_device>;
        using integrator_fd_t = integrator_karniadakis_fd_t<value_t, allocator_device>;
        using integrator_bs_t = integrator_karniadakis_bs_t<value_t, allocator_device>;
#endif //DEVICE

#ifdef HOST
        using arr_real = cuda_array_bc_nogp<value_t, allocator_host>;
        using arr_cmpl = cuda_array_bc_nogp<cmplx_t, allocator_host>;
        using dft_t = fftw_object_t<value_t>;
        using deriv_t = deriv_fd_t<value_t, allocator_host>;
        using integrator_fd_t = integrator_karniadakis_fd_t<value_t, allocator_host>;
        using integrator_bs_t = integrator_karniadakis_bs_t<value_t, allocator_host>;
#endif //HOST

        // typedef calls to functions that compute the implicit part for time integration.
        // These functions have 2 time indices as the argument.
        // t_dst gives the index where the result is written to
        // t_src gives the index where the input is if the array has more than one time index (i.e. the fields that are integrated)
        // for all other fields, theta_[xy], strmf, strmf_[xy] this defaults to 0.
        typedef void(slab_bc ::*rhs_func_ptr) (const size_t, const size_t); 

        slab_bc(const twodads::slab_layout_t, const twodads::bvals_t<value_t>, const twodads::stiff_params_t);
        slab_bc(const slab_config_js& cfg);
        ~slab_bc();

        void dft_r2c(const twodads::field_t, const size_t);
        void dft_c2r(const twodads::field_t, const size_t);

        void initialize();

        void invert_laplace(const twodads::field_t, const twodads::field_t, const size_t, const size_t);

        void d_dx(const twodads::field_t, const twodads::field_t, const size_t, const size_t, const size_t);
        void d_dy(const twodads::field_t, const twodads::field_t, const size_t, const size_t, const size_t);
        void pbracket(const twodads::field_t, const twodads::field_t, const twodads::field_t, const size_t, const size_t, const size_t);

        void integrate(const twodads::dyn_field_t, const size_t);
        void update_real_fields(const size_t);

        void advance();

        void write_output(const size_t);
        void write_output(const size_t, const twodads::real_t);

        arr_real* get_array_ptr(const twodads::field_t fname) const {return(get_field_by_name.at(fname));};

        const slab_config_js& get_config() {return(conf);};

        void rhs(const size_t, const size_t);

        // The RHS functions below should not be called directly but via the pointers
        // stored in ***_rhs_func
        void rhs_theta_null(const size_t t_dst, const size_t t_src)
        {
            // Do nothing
        };

        void rhs_omega_null(const size_t t_dst, const size_t t_src)
        {
            // Do nothing
        };

        void rhs_omega_ic(const size_t, const size_t);

        void rhs_tau_null(const size_t t_dst, const size_t t_src)
        {
            // Do nothing
        };

        void rhs_theta_lin(const size_t, const size_t);

    private:

        const slab_config_js conf;
        output_h5_t output;
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
            my_map[twodads::rhs_t::rhs_omega_null] = &slab_bc::rhs_omega_null;
            my_map[twodads::rhs_t::rhs_omega_ic]   = &slab_bc::rhs_omega_ic;
            my_map[twodads::rhs_t::rhs_tau_null]   = &slab_bc::rhs_tau_null;
            return(my_map);
        }
};

#endif // SLAB_BC_H