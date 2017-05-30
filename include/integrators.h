/*
 * Provide time integrators to be used by slab data types
 *
 */


#ifndef INTEGRATORS_H
#define INTEGRATORS_H

#include <iostream>
#include <string>
#include "error.h"
#include "cuda_array_bc_nogp.h"
#include "dft_type.h"
#include "solvers.h"
#include "2dads_types.h"


namespace detail
{
#ifdef __CUDACC__
    template <typename T>
    void impl_solve_tridiagonal(cuda_array_bc_nogp<T, allocator_device>& field,
                                const cuda_array_bc_nogp<CuCmplx<T>, allocator_device>& diag_u,
                                const cuda_array_bc_nogp<CuCmplx<T>, allocator_device>& diag,
                                const cuda_array_bc_nogp<CuCmplx<T>, allocator_device>& diag_l,
                                const size_t t_dst, 
                                solvers :: elliptic_base_t* ell_solver,
                                allocator_device<T>)
    {   
        ell_solver -> solve(reinterpret_cast<CuCmplx<T>*>(field.get_tlev_ptr(t_dst)), 
                            reinterpret_cast<CuCmplx<T>*>(field.get_tlev_ptr(t_dst)),
                            diag_l.get_tlev_ptr(0), 
                            diag.get_tlev_ptr(0), 
                            diag_u.get_tlev_ptr(0));
    }

#endif //__CUDACC__

#ifndef __CUDACC__

    template <typename T>
    void impl_solve_tridiagonal(cuda_array_bc_nogp<T, allocator_host>& field,
                                const cuda_array_bc_nogp<CuCmplx<T>, allocator_host>& diag_u,
                                const cuda_array_bc_nogp<CuCmplx<T>, allocator_host>& diag,
                                const cuda_array_bc_nogp<CuCmplx<T>, allocator_host>& diag_l,
                                const size_t t_dst, 
                                solvers :: elliptic_base_t* ell_solver,
                                allocator_host<T>)
    {
        ell_solver -> solve(nullptr,
                            reinterpret_cast<CuCmplx<T>*>(field.get_tlev_ptr(t_dst)),
                            diag_l.get_tlev_ptr(0) + 1,
                            diag.get_tlev_ptr(0),
                            diag_u.get_tlev_ptr(0));  
    }
#endif //__CUDACC__
}



/*
 * Interface to time integrator routines
 *
 */
template <typename T, template<typename> class allocator>
class integrator_base_t
{
    public:
        integrator_base_t(){}
        virtual ~integrator_base_t() {}
        
        virtual void integrate(cuda_array_bc_nogp<T, allocator>&, 
                               const cuda_array_bc_nogp<T, allocator>&, 
                               const size_t, const size_t, const size_t, const size_t, const size_t) = 0;
};


// Karniadakis stiffly-stable time integrator. Sub-class of integrator
template <typename T, template<typename> class allocator>
class integrator_karniadakis_fd_t : public integrator_base_t<T, allocator>
{
    public:
#ifdef DEVICE
    using dft_t = cufft_object_t<T>;
    using elliptic_t = solvers :: elliptic_cublas_t;
#endif // DEVICE

#ifdef HOST
    using dft_t = fftw_object_t<T>;
    using elliptic_t = solvers :: elliptic_mkl_t;
#endif // HOST

        integrator_karniadakis_fd_t(const twodads::slab_layout_t& _sl, const twodads::bvals_t<T>& _bv, const twodads::stiff_params_t& _sp) :
            geom{_sl}, bvals{_bv}, stiff_params{_sp},  
            geom_transpose{get_geom().get_ylo(),
                           get_geom().get_deltay(),
                           get_geom().get_xleft(),
                           get_geom().get_deltax(),
                           (get_geom().get_my() + get_geom().get_pad_y()) / 2, 0,
                           get_geom().get_nx(), 0,
                           get_geom().get_grid()},
            myfft{new dft_t(get_geom(), twodads::dft_t::dft_1d)},   
            my_solver{new elliptic_t(get_geom())},
            diag_order{1},
            // Pass a complex bvals_t to these guys. They don't really need it though.
            diag(get_geom_transpose(), twodads::bvals_t<CuCmplx<T>>(twodads::bc_t::bc_null, twodads::bc_t::bc_null, CuCmplx<T>{0.0}, CuCmplx<T>{0.0}), 1),
            diag_l(get_geom_transpose(), twodads::bvals_t<CuCmplx<T>>(twodads::bc_t::bc_null, twodads::bc_t::bc_null, CuCmplx<T>{0.0}, CuCmplx<T>{0.0}), 1),
            diag_u(get_geom_transpose(), twodads::bvals_t<CuCmplx<T>>(twodads::bc_t::bc_null, twodads::bc_t::bc_null, CuCmplx<T>{0.0}, CuCmplx<T>{0.0}), 1)
        {
            init_diagonal(1, bvals.get_bc_left(), bvals.get_bc_right());
            init_diagonals_ul();
        }
        ~integrator_karniadakis_fd_t() 
        {
            delete my_solver;
            delete myfft;
        }
        
        void integrate(cuda_array_bc_nogp<T, allocator>&, const cuda_array_bc_nogp<T, allocator>&, const size_t, const size_t, const size_t, const size_t, const size_t);

        // init_diagonals() initializes the diagonal elements used for elliptic solver.
        // The main diagonal depends on the order of the integrator and is called in constructor for first level
        // and subsequently the first time when a higher level integrator routine is called
        void init_diagonal(const size_t, const twodads::bc_t, const twodads::bc_t);
        // It should really be private, but then nvcc complains
        // An explicit __device__ lambda cannot be defined in a member function that has private or protected access within its class
        // The upper and lower diagonals are the same for all time levels and are initialized once by the constructor
        void init_diagonals_ul();

        inline twodads::slab_layout_t get_geom() const {return(geom);};
        inline twodads::bvals_t<T> get_bvals() const {return(bvals);};
        inline twodads::stiff_params_t get_tint_params() const {return(stiff_params);};
        inline twodads::slab_layout_t get_geom_transpose() const {return(geom_transpose);};

        inline cuda_array_bc_nogp<CuCmplx<T>, allocator> get_diag() const {return(diag);};
        inline cuda_array_bc_nogp<CuCmplx<T>, allocator> get_diag_u() const {return(diag_u);};
        inline cuda_array_bc_nogp<CuCmplx<T>, allocator> get_diag_l() const {return(diag_l);};

        inline T get_rx() const {return(get_tint_params().get_diff() * get_tint_params().get_deltat() / (get_geom().get_deltax() * get_geom().get_deltax()));};

        inline elliptic_t* get_ell_solver() {return(my_solver);};
    private:
        // Diagonal elements for elliptic solver
        const twodads::slab_layout_t geom;
        const twodads::bvals_t<twodads::real_t> bvals;
        const twodads::stiff_params_t stiff_params;

        // Transposed complex layout for the diagonals. See derivatives.h
        const twodads::slab_layout_t geom_transpose;

        // Fourier transformation happens in the time integration where we solve
        // in each fourier mode
        dft_t* myfft;
        elliptic_t* my_solver;

        size_t diag_order;
        void set_diag_order(const size_t o) {diag_order = o;};
        size_t get_diag_order() const {return(diag_order);};

        cuda_array_bc_nogp<CuCmplx<T>, allocator> diag;
        cuda_array_bc_nogp<CuCmplx<T>, allocator> diag_l;
        cuda_array_bc_nogp<CuCmplx<T>, allocator> diag_u;
};


template <typename T, template<typename> class allocator>
void integrator_karniadakis_fd_t<T, allocator> :: init_diagonal(const size_t order,
                                                                const twodads::bc_t bc_left,
                                                                const twodads::bc_t bc_right)
{
    // Get values from members not passed to the lambda so we can pass them by value into the lambda function, [=] capture
    const T rx{get_rx()};
    const T alpha{twodads::alpha[order - 1][0]};
    
    // Initialize the main diagonal to alpha_0 + 2 * rx + ky^2 * diff * dt
    // The first and last element on the main diagonal depend on boundary condition
    // 3 * rx for bc_dirichlet
    // 1 * rx for bc_neumann
    T val_left{0.0};
    T val_right{0.0};

    switch(bc_left)
    {
        case twodads::bc_t::bc_dirichlet:
            val_left = 3.0;
            break;
        case twodads::bc_t::bc_neumann:
            val_right = 1.0;
            break;
        case twodads::bc_t::bc_periodic:
            // Fall through
        default:
            throw not_implemented_error("integrator_karniadakis_fd_t does not handle periodic boundary conditions");
    }

    switch(bc_right)
    {
        case twodads::bc_t::bc_dirichlet:
            val_right = 3.0;
            break;
        case twodads::bc_t::bc_neumann:
            val_right = 1.0;
            break;
        case twodads::bc_t::bc_periodic:
            // Fall through
        default:
            throw not_implemented_error("integrator_karniadakis_fd_t does not handle periodic boundary conditions");        
    }

    diag.apply([=] LAMBDACALLER (CuCmplx<T> input, const size_t n, const size_t m, twodads::slab_layout_t geom) -> CuCmplx<T>
    {
        const T Lx{geom.get_deltax() * 2 * (geom.get_nx() - 1)};
        const T ky2{twodads::TWOPI * twodads::TWOPI * static_cast<T>(n * n) / (Lx * Lx)};
        
        if (m == 0)
            return(CuCmplx<T>(val_left * rx + ky2 * rx * geom.get_deltax() * geom.get_deltax() + alpha, 0.0));
        else if (m == geom.get_my() - 1)
            return(CuCmplx<T>(val_right * rx + ky2 * rx * geom.get_deltax() * geom.get_deltax() + alpha, 0.0));
        else
            return(CuCmplx<T>(2.0 * rx + ky2 * rx * geom.get_deltax() * geom.get_deltax() + alpha, 0.0));
    }, 0);

    set_diag_order(order);
}


template <typename T, template<typename> class allocator>
void integrator_karniadakis_fd_t<T, allocator> :: init_diagonals_ul()
{
    // diag_[lu] are transposed. Use m = 0..Nx-1, n = 0..My/2 and interchange x and y
    // ->  Lx = dx * (2 * nx - 1) as we have cut nx roughly in half

    const T rx{get_rx()};
    diag_l.apply([=] LAMBDACALLER (CuCmplx<T> dummy, const size_t n, const size_t m, twodads::slab_layout_t geom) -> CuCmplx<T>
    {
        // CUBLAS requires the first element in the lower diagonal to be zero.
        // Remember to shift the pointer in the MKL implementation when passing to the
        // MKL caller routine in solver as to skip the first element
        if(m == 0)
            return(0.0);
        return(CuCmplx<T>{-1. * rx, 0.0});
    }, 0);

    diag_u.apply([=] LAMBDACALLER (CuCmplx<T> dummy, const size_t n, const size_t m, twodads::slab_layout_t geom) -> CuCmplx<T>
    {
        // CUBLAS requires the last element in the upper diagonal to be zero.
        // The MKL solver doesn't care about the last element.
        if(m == geom.get_my() - 1)
            return(0.0);  
        return(CuCmplx<T>{-1. * rx, 0.0});  
    }, 0);
}


// Perform karniadakis stiffly-stable time integration
// Given the solution in the previous K time steps, u^{-1}, u^{-2}, ... u^{-K} 
// and the explicit terms of the previous K time steps NL{-1}, NL^{-2}, ... NL^{-L} 
// compute the solution for the next time step u^{0}
//
// field: array where time data is stored. 
// explicit_part: Explicit part in karniadakis scheme
// t_src1: index where u^{-1} is located
// t_src2: index where u^{-2} is located
// t_src3: index where u^{-3} is located
//
// data of explicit part is stored at indices with offset -1:
// N^{-1} : t_src1-1
// N^{-2} : t_src2-1
// N^{-3} : t_src3-1
//
// t_dst: time index to store the result in
// order: Order of time integration

template <typename T, template<typename> class allocator>
void integrator_karniadakis_fd_t<T, allocator> :: integrate(cuda_array_bc_nogp<T, allocator>& field,
                                                         const cuda_array_bc_nogp<T, allocator>& explicit_part,  
                                                         const size_t t_src1, const size_t t_src2, const size_t t_src3, 
                                                         const size_t t_dst, const size_t order) 
{
    // Set up the data for time integrations:
    // Sum up the implicit and explicit terms into t_dst 
    // Fourier transform the sum
    // Add boundary terms to ky=0 mode for n=0, Nx-1
    // Call tridiagonal solver
    // Return result in Fourier space


    // In the following we define local constants for the coefficients such that the
    // __device__ lambdas can capture them by value, [=] capture.
    // They are stored in a host array, which __device__ functions cannot capture by value yet :/

    if(order == 1)
    {
        assert(field.is_transformed(t_src1) == false);
        assert(explicit_part.is_transformed(t_dst) == false);
        // Initialize the main diagonal for first order time step
        if(get_diag_order() != 1) 
            init_diagonal(1, field.get_bvals().get_bc_left(), field.get_bvals().get_bc_right());

        const T alpha1{twodads::alpha[0][1]};
        const T beta1_dt{twodads::beta[0][0] * get_tint_params().get_deltat()};

        // u^{0} = alpha_1 u^{-1}
        field.elementwise([=] LAMBDACALLER(T lhs, T rhs) -> T {return(rhs * alpha1);}, 
                          t_dst, t_src1);

        // u^{0} += beta_1 N^{-1}
        field.elementwise([=] LAMBDACALLER(T lhs, T rhs) -> T{return(lhs + rhs * beta1_dt);}, 
                          explicit_part, t_dst, t_src1 - 1);

    }
    else if(order == 2)
    {
        // Initialize main diagonal for second order time step
        if(get_diag_order() != 2) 
            init_diagonal(2, field.get_bvals().get_bc_left(), field.get_bvals().get_bc_right());

        // Sum up previous time steps in t_dst:
        const T alpha2{twodads::alpha[1][2]};
        const T alpha1{twodads::alpha[1][1]};

        const T beta2_dt{twodads::beta[1][1] * get_tint_params().get_deltat()};
        const T beta1_dt{twodads::beta[1][0] * get_tint_params().get_deltat()};

        // u^{0} = alpha_2 * u^{-2}
        field.elementwise([=] LAMBDACALLER(T lhs, T rhs) -> T { return(rhs * alpha2);}, t_dst, t_src2);
        // u^{0} += alpha_1 * u^{-1}
        field.elementwise([=] LAMBDACALLER(T lhs, T rhs) -> T { return(lhs + rhs * alpha1);}, t_dst, t_src1);
        // u^{0} += dt * beta_2 * N^{-2}
        field.elementwise([=] LAMBDACALLER(T lhs, T rhs) -> T{return(lhs + rhs * beta2_dt);}, explicit_part, t_dst, t_src2 - 1);
        // u^{0} += dt * beta_1 * N^{-1}
        field.elementwise([=] LAMBDACALLER(T lhs, T rhs) -> T{return(lhs + rhs * beta1_dt);}, explicit_part, t_dst, t_src1 - 1);
    }

    else if (order == 3)
    {   
        if(get_diag_order() != 3) 
            init_diagonal(3, field.get_bvals().get_bc_left(), field.get_bvals().get_bc_right());

        // Sum up previous time steps in t_dst:
        const T alpha3{twodads::alpha[2][3]};
        const T alpha2{twodads::alpha[2][2]};
        const T alpha1{twodads::alpha[2][1]};

        const T beta3_dt{twodads::beta[2][2] * get_tint_params().get_deltat()};
        const T beta2_dt{twodads::beta[2][1] * get_tint_params().get_deltat()};
        const T beta1_dt{twodads::beta[2][0] * get_tint_params().get_deltat()};

        // u^{0} = alpha_3 * u^{-3}
        field.elementwise([=] LAMBDACALLER(T lhs, T rhs) -> T { return(rhs * alpha3);}, t_dst, t_src3);
        // u^{0} += alpha_2 * u^{-2}
        field.elementwise([=] LAMBDACALLER(T lhs, T rhs) -> T { return(lhs + rhs * alpha2);}, t_dst, t_src2);        
        // u^{0} += alpha_1 * u^{-1}
        field.elementwise([=] LAMBDACALLER(T lhs, T rhs) -> T { return(lhs + rhs * alpha1);}, t_dst, t_src1);
        // u^{0} += dt * beta_3 * N^{-3}
        field.elementwise([=] LAMBDACALLER(T lhs, T rhs) -> T{ return(lhs + rhs * beta3_dt);}, explicit_part, t_dst, t_src3 - 1);
        // u^{0} += dt * beta_2 * N^{-2}
        field.elementwise([=] LAMBDACALLER(T lhs, T rhs) -> T{ return(lhs + rhs * beta2_dt);}, explicit_part, t_dst, t_src2 - 1);
        // u^{0} += dt * beta_1 * N^{-1}
        field.elementwise([=] LAMBDACALLER(T lhs, T rhs) -> T{ return(lhs + rhs * beta1_dt);}, explicit_part, t_dst, t_src1 - 1);
    }

    (*myfft).dft_r2c(field.get_tlev_ptr(t_dst), reinterpret_cast<CuCmplx<T>*>(field.get_tlev_ptr(t_dst)));
    field.set_transformed(t_dst, true);

    // Treat the boundary conditions
    // Real part of the Fourier transform of the left boundary value
    T bval_left_hat{field.get_bvals().get_bv_left() * static_cast<T>(field.get_my())};
    // Real part of the Fourier transform of the right boundary value
    T bval_right_hat{field.get_bvals().get_bv_right() * static_cast<T>(field.get_my())};

    // This is the value we later add to the ky=0 mode in the n=0 row
    T add_to_boundary_left{0.0};
    // This is the value we later add to the ky=0 mode in the n=Nx-1 row
    T add_to_boundary_right{0.0};
    switch(field.get_bvals().get_bc_left())
    {
        case twodads::bc_t::bc_dirichlet:
            add_to_boundary_left = bval_left_hat * get_rx() * 2.0;
            break;
        case twodads::bc_t::bc_neumann:
            add_to_boundary_left = -1.0 * bval_left_hat * get_rx() * field.get_geom().get_deltax();
            break;
        case twodads::bc_t::bc_periodic:
        default:
            throw not_implemented_error(std::string("Periodic boundary conditions not supported by this integrator"));
            //break; 
    }

    switch(field.get_bvals().get_bc_right())
    {
        case twodads::bc_t::bc_dirichlet:
            add_to_boundary_right = bval_right_hat * get_rx() * 2.0;
            break;
        case twodads::bc_t::bc_neumann:
            add_to_boundary_right = bval_right_hat * get_rx() * field.get_geom().get_deltax();
            break;
        case twodads::bc_t::bc_periodic:
        default:
            throw not_implemented_error(std::string("Periodic boundary conditions not supported by this integrator"));
            //break;         
    }
    // Add boundary terms, i.e. the real part of the fourier transform. (field is defined as T->twodads::real_T)
    field.apply([=] LAMBDACALLER (T input, const size_t n, const size_t m, twodads::slab_layout_t geom) -> T
    {
        if(n == 0  && m == 0)
            return(input + add_to_boundary_left);
        else if(n == geom.get_nx() - 1 && m == 0)
            return(input + add_to_boundary_right);
        else
            return(input);
    }, t_dst);

    detail :: impl_solve_tridiagonal(field, get_diag_u(), get_diag(), get_diag_l(), t_dst, get_ell_solver(), allocator<T>{});
}


// Karniadakis integration for bispectral layout
// The integrate member needs to be called with complex fields
template <typename T, template<typename> class allocator>
class integrator_karniadakis_bs_t : public integrator_base_t<T, allocator>
{
    public:
        integrator_karniadakis_bs_t(const twodads::slab_layout_t& _sl,
                                    const twodads::bvals_t<T>& _bv,
                                    const twodads::stiff_params_t& _sp) :
            geom{_sl}, bvals{_bv}, stiff_params{_sp},
            k2_map(get_geom(), twodads::bvals_t<T>(twodads::bc_t::bc_dirichlet, twodads::bc_t::bc_dirichlet, T(0.0), T(0.0)), 1)
        {
            k2_map.set_transformed(0, true);
            init_k2_map();
        }

        void integrate(cuda_array_bc_nogp<T, allocator>&,
                    const cuda_array_bc_nogp<T, allocator>&,
                    const size_t, const size_t, const size_t,
                    const size_t, const size_t);

        void init_k2_map();
        const cuda_array_bc_nogp<T, allocator>& get_k2_map() const {return(k2_map);};

        inline twodads::slab_layout_t get_geom() const {return(geom);};
        inline twodads::bvals_t<T> get_bvals() const {return(bvals);};
        inline twodads::stiff_params_t get_tint_params() const {return(stiff_params);};
    
    private:
        const twodads::slab_layout_t geom;
        const twodads::bvals_t<twodads::real_t> bvals;
        const twodads::stiff_params_t stiff_params; 
        cuda_array_bc_nogp<T, allocator> k2_map;
};


template<typename T, template<typename> class allocator>
void integrator_karniadakis_bs_t<T, allocator> :: init_k2_map()
{
    // Construct Ly (cut in half) outside of the function
    k2_map.apply([] LAMBDACALLER (T input, const size_t n, const size_t m, twodads::slab_layout_t geom) -> T
                  {
                    const T kx{twodads::TWOPI * ( (n < geom.get_nx() / 2 + 1) ? T(n) : (T(n) - T(geom.get_nx())) ) / geom.get_Lx()};
                    const T ky{twodads::TWOPI * 0.5 * T(m - (m % 2)) / geom.get_Ly()};
                    return(kx * kx + ky * ky);
                  }, 0);

    //utility :: print(get_k2_map(), 0, std::cout);
}


template<typename T, template<typename> class allocator>
void integrator_karniadakis_bs_t<T, allocator> :: integrate(cuda_array_bc_nogp<T, allocator>& field,
                                                            const cuda_array_bc_nogp<T, allocator>& explicit_part,
                                                            const size_t t_src1, const size_t t_src2, const size_t t_src3,
                                                            const size_t t_dst, const size_t order)
{
    // Set up the data for time integration:
    // Sum up the implicit and explicit terms into t_dst.
    // Assert that all previous implicit and explicit terms are transformed 
    // i.e. we have their fourier coefficients.
    // Mark the resulting field as transformed as well.

    assert(order < 4);
    assert(get_k2_map().is_transformed(0));

    const T diff{get_tint_params().get_diff()};
    const T dt{get_tint_params().get_deltat()};

    switch(order)
    {
        case 1:
            assert(field.is_transformed(t_src1));
            assert(explicit_part.is_transformed(t_src1 - 1));
            assert(get_k2_map().is_transformed(0));
    
            // u^{0} = alpha_1 u^{-1}
            field.elementwise([=] LAMBDACALLER(T lhs, T rhs) -> T {return(rhs * twodads::alpha[order - 1][1]);}, 
                              t_dst, t_src1);

            // u^{0} += beta_1 N^{-1}    
            field.elementwise([=] LAMBDACALLER(T lhs, T rhs) -> T {return(lhs + rhs * twodads::beta[order - 1][0] * dt);},
                              explicit_part, t_dst, t_src1 - 1);
            
            // u^{0} /= (1.0 + dt * (diff * k^2 ))
            field.elementwise([=] LAMBDACALLER (T lhs, T k2) -> T 
                              {
                                  return(lhs / (twodads::alpha[order - 1][0] + dt * k2 * diff));
                              }, get_k2_map(), 0, t_dst);                       
            field.set_transformed(t_dst, true);
            break;

        case 2:
            assert(field.is_transformed(t_src1));
            assert(field.is_transformed(t_src2));

            assert(explicit_part.is_transformed(t_src2 - 1));
            assert(explicit_part.is_transformed(t_src1 - 1));

            assert(get_k2_map().is_transformed(0));
            
            // u^{0} = alpha_2 * u^{-2}
            field.elementwise([=] LAMBDACALLER(T lhs, T rhs) -> T {return(rhs * twodads::alpha[order - 1][2]);},
                              t_dst, t_src2);
            // u^{0} += alpha_1 * u^{-1}
            field.elementwise([=] LAMBDACALLER(T lhs, T rhs) -> T {return(lhs + rhs * twodads::alpha[order - 1][1]);},
                              t_dst, t_src1);


            // u^{0} += dt * beta_2 * N^{-2}
            field.elementwise([=] LAMBDACALLER(T lhs, T rhs) -> T {return(lhs + rhs * twodads::beta[order - 1][1] * dt);},
                              explicit_part, t_dst, t_src2 - 1);
            // u^{0} += dt * beta_1 * N^{-1}
            field.elementwise([=] LAMBDACALLER(T lhs, T rhs) -> T {return(lhs + rhs * twodads::beta[order - 1][0] * dt);},
                              explicit_part, t_dst, t_src1 - 1);

            // u^{0} /= (1.0 + dt * (diff * k^2 + hv * k^6))
            field.elementwise([=] LAMBDACALLER (T lhs, T k2) -> T
                            {
                                return(lhs / (twodads::alpha[order - 1][0] + k2 * dt * diff));
                            }, get_k2_map(), 0, t_dst);
            field.set_transformed(t_dst, true);
            break;

        case 3:    
            assert(field.is_transformed(t_src1));
            assert(field.is_transformed(t_src2));
            assert(field.is_transformed(t_src3));

            assert(explicit_part.is_transformed(t_src1 - 1));
            assert(explicit_part.is_transformed(t_src2 - 1));
            assert(explicit_part.is_transformed(t_src3 - 1));

            // u^{0} = alpha_3 * u^{-3}
            field.elementwise([=] LAMBDACALLER(T lhs, T rhs) -> T {return(rhs * twodads::alpha[order - 1][3]);},
                            t_dst, t_src3);
            // u^{0} = alpha_2 * u^{-2}
            field.elementwise([=] LAMBDACALLER(T lhs, T rhs) -> T {return(lhs + rhs * twodads::alpha[order - 1][2]);},
                            t_dst, t_src2);
            // u^{0} += alpha_1 * u^{-1}
            field.elementwise([=] LAMBDACALLER(T lhs, T rhs) -> T {return(lhs + rhs * twodads::alpha[order - 1][1]);},
                            t_dst, t_src1);

            // u^{0} += dt * beta_3 * N^{-3}
            field.elementwise([=] LAMBDACALLER(T lhs, T rhs) -> T {return(lhs + rhs * twodads::beta[order - 1][2] * dt);},
                            explicit_part, t_dst, t_src3 - 1);
            // u^{0} += dt * beta_2 * N^{-2}
            field.elementwise([=] LAMBDACALLER(T lhs, T rhs) -> T {return(lhs + rhs * twodads::beta[order - 1][1] * dt);},
                            explicit_part, t_dst, t_src2 - 1);
            // u^{0} += dt * beta_1 * N^{-1}
            field.elementwise([=] LAMBDACALLER(T lhs, T rhs) -> T {return(lhs + rhs * twodads::beta[order - 1][0] * dt);},
                            explicit_part, t_dst, t_src1 - 1);

            // u^{0} /= (1.0 + dt * (diff * k^2 + hv * k^6))
            field.elementwise([=] LAMBDACALLER (T lhs, T k2) -> T
                            {
                                return(lhs / (twodads::alpha[2][0] + k2 * dt * diff));
                            }, get_k2_map(), 0, t_dst);
            field.set_transformed(t_dst, true);
            break;
        
        default:
            throw(not_implemented_error("Spectral time integration supports only orders 1-3"));
            break;
    }
}


#endif //INTEGRATORS_H
// End of file integrators.h
