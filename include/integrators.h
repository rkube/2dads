/*
 * Provide time integrators to be used by slab data types
 *
 */

#include <iostream>
#include "cuda_array_bc_nogp.h"
#include "error.h"

template <typename allocator>
class integrator 
{
    using value_t = typename my_allocator_traits<allocator> :: value_type;
    public:
        integrator(const cuda::slab_layout_t _geom, 
                   const cuda::bvals_t<cuda::real_t> _bvals_theta, 
                   const cuda::stiff_params_t _stiff_params) : 
            geom(_geom), params(_stiff_params), 
            theta(get_geom(), _bvals_theta, params.get_tlevs())
            //theta_rhs(get_geom(), _bvals_theta, params.get_tlevs() - 1))
            {};

        // Integrate the field in time, over-write the result
        virtual void integrate(cuda_array_bc_nogp<allocator>&) = 0;

        const cuda::slab_layout_t get_geom() const {return(geom);};
        const cuda::stiff_params_t get_params() const {return(params);};
        cuda_array_bc_nogp<allocator>& get_array() const {return(theta);};
        //cuda_array_bc_nogp<allocator>& get_array_rhs() const {return(theta_rhs);};
    private:
        const cuda::slab_layout_t geom;
        const cuda::stiff_params_t params;
        cuda_array_bc_nogp<allocator> theta;
        //cuda_array_bc_nogp<allocator> theta_rhs;
};



template <typename allocator>
class integrator_karniadakis : public integrator<allocator>
{
    public:
        using value_t = typename my_allocator_traits<allocator> :: value_type;

        integrator_karniadakis(const cuda::slab_layout_t _geom, 
                               const cuda::bvals_t<cuda::real_t> _bvals_theta, 
                               const cuda::stiff_params_t _stiff_params) : 
            My_int{static_cast<int>(_geom.get_my())},
            My21_int{static_cast<int>((_geom.get_my() + _geom.get_pad_y()) / 2)},
            Nx_int{static_cast<int>(_geom.get_nx())},
            inv_dx2{1.0 / (_geom.get_deltax() * _geom.get_deltax())}, 
            integrator<allocator>(_geom, _bvals_theta, _stiff_params) {};
        ~integrator_karniadakis();
        
        virtual void integrate(cuda_array_bc_nogp<allocator>&);

    private:
        const int My_int;
        const int My21_int;
        const int Nx_int;
        const value_t inv_dx2;
};


// Integrate the fields in time.
// see also derivative :: invert_laplac
template <typename allocator>
void integrator_karniadakis<allocator> :: integrate(cuda_array_bc_nogp<allocator>& in) 
{


}



