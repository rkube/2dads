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
    public:
        using value_t = typename my_allocator_traits<allocator> :: value_type;
        integrator(const cuda::slab_layout_t _geom, 
                   const cuda::bvals_t<cuda::real_t> _bvals_theta, 
                   const cuda::stiff_params_t _stiff_params) : 
            geom(_geom), params(_stiff_params), bvals(_bvals_theta),
            theta(get_geom(), get_bvals(), params.get_tlevs())
            //theta_rhs(get_geom(), _bvals_theta, params.get_tlevs() - 1))
            {};

        // Integrate the field in time, over-write the result
        virtual void integrate(cuda_array_bc_nogp<allocator>*) = 0;

        virtual void initialize_field(const cuda_array_bc_nogp<allocator>*, const size_t) = 0;

        const cuda::slab_layout_t get_geom() const {return(geom);};
        const cuda::stiff_params_t get_params() const {return(params);};
        const cuda::bvals_t<cuda::real_t> get_bvals() const {return(bvals);};
        cuda_array_bc_nogp<allocator>& get_array() {return(theta);};
        //cuda_array_bc_nogp<allocator>* get_array() {return(&theta);};
        //cuda_array_bc_nogp<allocator>& get_array_rhs() {return(theta_rhs);};
        
    private:
        const cuda::slab_layout_t geom;
        const cuda::stiff_params_t params;
        const cuda::bvals_t<cuda::real_t> bvals;
        cuda_array_bc_nogp<allocator> theta;
        //cuda_array_bc_nogp<allocator> theta_rhs;
};



template <typename allocator>
class integrator_karniadakis : public integrator<allocator>
{
    public:
        using value_t = typename integrator<allocator>::value_t;

        using integrator<allocator>::get_array;
        using integrator<allocator>::get_geom;
        using integrator<allocator>::get_params;
        using integrator<allocator>::get_bvals;

        integrator_karniadakis(const cuda::slab_layout_t _geom, 
                               const cuda::bvals_t<cuda::real_t> _bvals_theta, 
                               const cuda::stiff_params_t _stiff_params) : 
            My_int{static_cast<int>(_geom.get_my())},
            My21_int{static_cast<int>((_geom.get_my() + _geom.get_pad_y()) / 2)},
            Nx_int{static_cast<int>(_geom.get_nx())},
            inv_dx2{1.0 / (_geom.get_deltax() * _geom.get_deltax())}, 
            integrator<allocator>(_geom, _bvals_theta, _stiff_params) {};

        ~integrator_karniadakis();
        
        virtual void initialize_field(const cuda_array_bc_nogp<allocator>*, const size_t);
        virtual void integrate(cuda_array_bc_nogp<allocator>*);


        inline int get_my_int() const {return(My_int);};
        inline int get_my21_int() const {return(My21_int);};
        inline int get_nx_int() const {return(Nx_int);};
        inline value_t get_inv_dx2() const {return(inv_dx2);};

    private:
        const int My_int;
        const int My21_int;
        const int Nx_int;
        const value_t inv_dx2;
};


template <typename allocator>
void integrator_karniadakis<allocator> :: initialize_field(const cuda_array_bc_nogp<allocator>* src, const size_t t_src)
{
    std::cout << "Initializing field" << std::endl;
    // Copy input data to last time level
    get_array().copy(get_params().get_tlevs() - 1, (*src), t_src);
}


// Integrate the fields in time. Returns the 1d-dft into out
// see also derivative :: invert_laplace
template <typename allocator>
void integrator_karniadakis<allocator> :: integrate(cuda_array_bc_nogp<allocator>* out) 
{


}



