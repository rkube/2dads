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
        using cuda_arr_real = cuda_array_bc_nogp<value_t, allocator_device>;
        using cuda_arr_cmpl = cuda_array_bc_nogp<cmplx_t, allocator_device>;
        using dft_t = cufft_object_t<value_t>;
        using deriv_t = deriv_fd_t<value_t, allocator_device>;
        using integrator_t = integrator_karniadakis_t<value_t, allocator_device>;
#endif //DEVICE

#ifdef HOST
        using cuda_arr_real = cuda_array_bc_nogp<value_t, allocator_host>;
        using cuda_arr_cmpl = cuda_array_bc_nogp<cmplx_t, allocator_host>;
        using dft_t = fftw_object_t<value_t>;
        using deriv_t = deriv_fd_t<value_t, allocator_host>;
        using integrator_t = integrator_karniadakis_t<value_t, allocator_host>;
#endif //HOST

        typedef void(slab_bc ::*rhs_func_ptr) (const size_t); 

        slab_bc(const twodads::slab_layout_t, const twodads::bvals_t<value_t>, const twodads::stiff_params_t);
        slab_bc(const slab_config_js cfg);
        ~slab_bc();

        void dft_r2c(const twodads::field_t, const size_t);
        void dft_c2r(const twodads::field_t, const size_t);

        void initialize_invlaplace(const twodads::field_t, const size_t);
        void initialize_sine(const twodads::field_t, const size_t);
        void initialize_arakawa(const twodads::field_t, const twodads::field_t, const size_t);
        void initialize_derivativesx(const twodads::field_t, const size_t);
        void initialize_derivativesy(const twodads::field_t, const size_t);
        void initialize_dfttest(const twodads::field_t, const size_t);
        void initialize_gaussian(const twodads::field_t, const size_t);

        void invert_laplace(const twodads::field_t, const twodads::field_t, const size_t, const size_t);

        void d_dx(const twodads::field_t, const twodads::field_t, const size_t, const size_t, const size_t);
        void d_dy(const twodads::field_t, const twodads::field_t, const size_t, const size_t, const size_t);
        void arakawa(const twodads::field_t, const twodads::field_t, const twodads::field_t, const size_t, const size_t);

        void integrate(const twodads::field_t, const size_t);
        void update_real_fields(const size_t);
        void rhs(const size_t);

        void advance();

        cuda_arr_real* get_array_ptr(const twodads::field_t fname) const {return(get_field_by_name.at(fname));};
        inline twodads::slab_layout_t get_geom() const {return(geom);};
        inline twodads::bvals_t<value_t> get_bvals_theta() const {return(boundaries_theta);};
        inline twodads::bvals_t<value_t> get_bvals_omega() const {return(boundaries_omega);};
        inline twodads::bvals_t<value_t> get_bvals_tau() const {return(boundaries_tau);};
        inline twodads::bvals_t<value_t> get_bvals_strm() const {return(boundaries_strmf);};
        inline twodads::stiff_params_t get_tint_params() const {return(tint_params);};
    private:

        const twodads::bvals_t<value_t> boundaries_theta;
        const twodads::bvals_t<value_t> boundaries_omega;
        const twodads::bvals_t<value_t> boundaries_tau;
        const twodads::bvals_t<value_t> boundaries_strmf;

        const twodads::slab_layout_t geom;
        const twodads::stiff_params_t tint_params;

        dft_object_t<twodads::real_t>* myfft;

#ifdef DEVICE
        deriv_base_t<value_t, allocator_device>* my_derivs;
        integrator_base_t<value_t, allocator_device>* tint;
#endif //DEVICE
#ifdef HOST
        deriv_base_t<value_t, allocator_host>* my_derivs;
        integrator_base_t<value_t, allocator_host>* tint;
#endif //HOST

        cuda_arr_real theta;
        cuda_arr_real theta_x;
        cuda_arr_real theta_y;

        cuda_arr_real omega;
        cuda_arr_real omega_x;
        cuda_arr_real omega_y;

        cuda_arr_real tau;
        cuda_arr_real tau_x;
        cuda_arr_real tau_y;

        cuda_arr_real strmf;
        cuda_arr_real strmf_x;
        cuda_arr_real strmf_y;

        cuda_arr_real theta_rhs;
        cuda_arr_real omega_rhs;
        cuda_arr_real tau_rhs;

        const std::map<twodads::field_t, cuda_arr_real*> get_field_by_name;
        
        rhs_func_ptr theta_rhs_func;
        rhs_func_ptr omega_rhs_func;
        rhs_func_ptr tau_rhs_func;

        
        static std::map<twodads::rhs_t, rhs_func_ptr> rhs_func_map;

        void theta_rhs_null(const size_t tsrc)
        {
            std::cout << "theta_rhs_null" << tsrc << std::endl;
        };

        void omega_rhs_null(const size_t tsrc)
        {
            std::cout << "omega_rhs_null" << tsrc << std::endl;
        };

        void tau_rhs_null(const size_t tsrc)
        {
            std::cout << "tau_rhs_null" << tsrc << std::endl;
        };

        static std::map<twodads::rhs_t, rhs_func_ptr> create_rhs_func_map()
        {
            std::map<twodads::rhs_t, rhs_func_ptr> my_map;
            my_map[twodads::rhs_t::theta_rhs_null] = &slab_bc::theta_rhs_null;
            my_map[twodads::rhs_t::omega_rhs_null] = &slab_bc::tau_rhs_null;
            my_map[twodads::rhs_t::tau_rhs_null]   = &slab_bc::tau_rhs_null;
            return(my_map);
        }
};

#endif // SLAB_BC_H