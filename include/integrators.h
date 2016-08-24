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
                                allocator_device<T>)
    {
        /*
        solvers :: elliptic my_ell_solver(field.get_geom());
        my_ell_solver.solve(reinterpret_cast<cuDoubleComplex*>(src.get_tlev_ptr(t_src)), 
                            reinterpret_cast<cuDoubleComplex*>(dst.get_tlev_ptr(t_dst)),
                            reinterpret_cast<cuDoubleComplex*>(diag_l.get_tlev_ptr(0)), 
                            reinterpret_cast<cuDoubleComplex*>(diag.get_tlev_ptr(0)), 
                            reinterpret_cast<cuDoubleComplex*>(diag_u.get_tlev_ptr(0)));
        */
        std::cerr << "integrator_karniadakis :: detail :: impl_solve_tridiagonal: Not implemented yet" << std::endl;
        
    }

#endif //__CUDACC__


    template <typename T>
    void impl_solve_tridiagonal(cuda_array_bc_nogp<T, allocator_host>& field,
                                const cuda_array_bc_nogp<CuCmplx<T>, allocator_host>& diag_u,
                                const cuda_array_bc_nogp<CuCmplx<T>, allocator_host>& diag,
                                const cuda_array_bc_nogp<CuCmplx<T>, allocator_host>& diag_l,
                                const size_t t_dst, 
                                allocator_host<T>)
    {
        solvers :: elliptic my_ell_solver(field.get_geom());
#ifndef __CUDACC__
        my_ell_solver.solve(nullptr,
                            reinterpret_cast<lapack_complex_double*>(field.get_tlev_ptr(t_dst)),
                            reinterpret_cast<lapack_complex_double*>(diag_l.get_tlev_ptr(0)) + 1,
                            reinterpret_cast<lapack_complex_double*>(diag.get_tlev_ptr(0)),
                            reinterpret_cast<lapack_complex_double*>(diag_u.get_tlev_ptr(0)));  
#endif //__CUDACC__
        //std::cout << "impl_integrate1: do nothing" << std::endl;  
    }
}


// Karniadakis stiffly-stable time integrator. Sub-class of integrator
template <typename T, template<typename> class allocator>
class integrator_karniadakis
{
    public:
#ifdef DEVICE
    using dft_library_t = cufft_object_t<twodads::real_t>;
#endif // DEVICE

#ifdef HOST
    using dft_library_t = fftw_object_t<twodads::real_t>;
#endif // HOST
        integrator_karniadakis(const twodads::slab_layout_t& _sl, const twodads::bvals_t<T>& _bv, const twodads::stiff_params_t& _sp) :
            geom{_sl}, bvals{_bv}, stiff_params{_sp},  
            geom_transpose{get_geom().get_ylo(),
                           get_geom().get_deltay(),
                           get_geom().get_xleft(),
                           get_geom().get_deltax(),
                           (get_geom().get_my() + get_geom().get_pad_y()) / 2, 0,
                           get_geom().get_nx(), 0,
                           get_geom().get_grid()},
            myfft{new dft_library_t(get_geom(), twodads::dft_t::dft_1d)},   
            My_int{static_cast<int>(get_geom().get_my())},
            My21_int{static_cast<int>(get_geom().get_my() + get_geom().get_pad_y()) / 2},
            Nx_int{static_cast<int>(get_geom().get_nx())},
            diag_order{0},
            // Pass a complex bvals_t to these guys. They don't really need it though.
            diag(get_geom_transpose(), twodads::bvals_t<CuCmplx<T>>(twodads::bc_t::bc_dirichlet, twodads::bc_t::bc_dirichlet, twodads::bc_t::bc_periodic, twodads::bc_t::bc_periodic, CuCmplx<T>{0.0}, CuCmplx<T>{0.0}, CuCmplx<T>{0.0}, CuCmplx<T>{0.0}), 1),
            diag_l(get_geom_transpose(), twodads::bvals_t<CuCmplx<T>>(twodads::bc_t::bc_dirichlet, twodads::bc_t::bc_dirichlet, twodads::bc_t::bc_periodic, twodads::bc_t::bc_periodic, CuCmplx<T>{0.0}, CuCmplx<T>{0.0}, CuCmplx<T>{0.0}, CuCmplx<T>{0.0}), 1),
            diag_u(get_geom_transpose(), twodads::bvals_t<CuCmplx<T>>(twodads::bc_t::bc_dirichlet, twodads::bc_t::bc_dirichlet, twodads::bc_t::bc_periodic, twodads::bc_t::bc_periodic, CuCmplx<T>{0.0}, CuCmplx<T>{0.0}, CuCmplx<T>{0.0}, CuCmplx<T>{0.0}), 1)
        {
            init_diagonal(1);
            init_diagonals_ul();
        }
        ~integrator_karniadakis();
        
        void integrate(cuda_array_bc_nogp<T, allocator>&, const size_t, const size_t, const size_t, const size_t, const size_t);

        // init_diagonals() initializes the diagonal elements used for elliptic solver.
        // The main diagonal depends on the order of the integrator and is called in constructor for first level
        // and subsequently the first time when a higher level integrator routine is called
        void init_diagonal(size_t);
        // It should really be private, but then nvcc complains
        // An explicit __device__ lambda cannot be defined in a member function that has private or protected access within its class
        // The upper and lower diagonals are the same for all time levels and are initialized once by the constructor
        void init_diagonals_ul();


        inline twodads::slab_layout_t get_geom() const {return(geom);};
        inline twodads::bvals_t<T> get_bvals() const {return(bvals);};
        inline twodads::stiff_params_t get_params() const {return(stiff_params);};
        inline twodads::slab_layout_t get_geom_transpose() const {return(geom_transpose);};

        inline cuda_array_bc_nogp<CuCmplx<T>, allocator> get_diag() const {return(diag);};
        inline cuda_array_bc_nogp<CuCmplx<T>, allocator> get_diag_u() const {return(diag_u);};
        inline cuda_array_bc_nogp<CuCmplx<T>, allocator> get_diag_l() const {return(diag_l);};

        inline int get_my_int() const {return(My_int);};
        inline int get_my21_int() const {return(My21_int);};
        inline int get_nx_int() const {return(Nx_int);};

        inline T get_rx() const {return(get_params().get_diff() * get_params().get_deltat() / (get_geom().get_deltax() * get_geom().get_deltax()));};

    private:
        // Diagonal elements for elliptic solver
        const twodads::slab_layout_t geom;
        const twodads::bvals_t<twodads::real_t> bvals;
        const twodads::stiff_params_t stiff_params;

        // Transposed complex layout for the diagonals. See derivatives.h
        const twodads::slab_layout_t geom_transpose;

        // Fourier transformation happens in the time integration where we solve
        // in each fourier mode
        dft_object_t<twodads::real_t>* myfft;

        // Array boundaries etc.
        const int My_int;
        const int My21_int;
        const int Nx_int;

        size_t diag_order;
        void set_diag_order(const size_t o) {diag_order = o;};
        size_t get_diag_order() const {return(diag_order);};

        cuda_array_bc_nogp<CuCmplx<T>, allocator> diag;
        cuda_array_bc_nogp<CuCmplx<T>, allocator> diag_l;
        cuda_array_bc_nogp<CuCmplx<T>, allocator> diag_u;
};


template <typename T, template<typename> class allocator>
void integrator_karniadakis<T, allocator> :: init_diagonal(const size_t tlev)
{
    const T Lx{get_geom().get_deltax() * 2 * (get_geom().get_nx() - 1)};
    //const CuCmplx<T> inv_dx2{1.0 / (get_geom().get_deltay() * get_geom().get_deltay())};
    //const CuCmplx<T> rx{get_params().get_diff() * get_params().get_deltat() / (get_geom().get_deltax() * get_geom().get_deltax())};
    //const CuCmplx<T> dx{get_geom().get_deltax()};
    //const CuCmplx<T> bv_left{get_bvals().get_bv_left()};
    //const CuCmplx<T> bv_right{get_bvals().get_bv_right()};

    std::cout << "Initializing diagonal for level " << tlev << ": rx = " << get_rx() << ", alpha0 = " << twodads::alpha[tlev - 1][0] << std::endl;
    CuCmplx<T> first_element{0.0, 0.0};
    CuCmplx<T> last_element{0.0, 0.0};

    switch(get_bvals().get_bc_left())
    {
        case twodads::bc_t::bc_dirichlet:
            first_element = CuCmplx<T>(twodads::alpha[tlev - 1][0] + 3.0 * get_rx(), 0.0);
            break;
        case twodads::bc_t::bc_neumann:
            first_element = CuCmplx<T>(twodads::alpha[tlev - 1][0] + get_rx(), 0.0);
            break;
        case twodads::bc_t::bc_periodic:
        default:
            throw not_implemented_error(std::string("Periodic boundary conditions not supported by this integrator"));
            //break;
    }

    switch(get_bvals().get_bc_left())
    {
        case twodads::bc_t::bc_dirichlet:
            last_element = CuCmplx<T>(twodads::alpha[tlev - 1][0] + 3.0 * get_rx(), 0.0);
            break;
        case twodads::bc_t::bc_neumann:
            last_element = CuCmplx<T>(twodads::alpha[tlev - 1][0] + get_rx(), 0.0);
            break;
        case twodads::bc_t::bc_periodic:
        default:
            throw not_implemented_error(std::string("Periodic boundary conditions not supported by this integrator"));
            //break;
    }

    const T rx{get_rx()};
    diag.apply([=] LAMBDACALLER (CuCmplx<T> input, const size_t n, const size_t m, twodads::slab_layout_t geom) -> CuCmplx<T>
    {
        // ky runs with index n (the kernel addressing function, see cuda::thread_idx)
        const T ky2{twodads::TWOPI * twodads::TWOPI * static_cast<T>(n * n) / (Lx * Lx)};
        if(n == 0 || n == geom.get_nx() - 1)
        {
            return(CuCmplx<T>{2.0 * rx + ky2 * rx * geom.get_deltax() * geom.get_deltax() + twodads::alpha[tlev - 1][0], 0.0});    
        }
        return(CuCmplx<T>{2.0 * rx + ky2 * rx * geom.get_deltax() * geom.get_deltax() + twodads::alpha[tlev - 1][0], 0.0});
    }, 0);

    set_diag_order(tlev);
}


template <typename T, template<typename> class allocator>
void integrator_karniadakis<T, allocator> :: init_diagonals_ul()
{
    // diag_[lu] are transposed. Use m = 0..Nx-1, n = 0..My/2 and interchange x and y
    // ->  Lx = dx * (2 * nx - 1) as we have cut nx roughly in half
    //const CuCmplx<T> rx{get_params().get_diff() * get_params().get_deltat() / (get_geom().get_deltax() * get_geom().get_deltax())};

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


// Integrate the fields in time. Returns the 1d-dft into out
// Given u^{k-1}, u^{k-2}, ... u^{k-K} compute u^{0}
// field: array where time data is stored. 
// t_src1: data from previous time step u^{-1}
// t_src2: data from next previous time step u^{t-2}
//
//
// Field must not be transformed at any time level
// Routine returns data in real space

template <typename T, template<typename> class allocator>
void integrator_karniadakis<T, allocator> :: integrate(cuda_array_bc_nogp<T, allocator>& field, 
                                                       const size_t t_src1, const size_t t_src2, const size_t t_src3, 
                                                       const size_t t_dst, const size_t tlev) 
{
    std::cout << "Integrating, tlev = " << tlev << std::endl;

    assert(field.is_transformed(t_src1) == false);
    assert(field.is_transformed(t_src2) == false);
    assert(field.is_transformed(t_src3) == false);

    // Set up the data for time integrations:
    // Sum up the fields into t_dst 
    // Fourier transform
    // Add the boundary terms to ky=0 mode for n=0, Nx-1
    // Call tridiagonal solver



    if(tlev == 1)
    {
        //std::cout << "tlev = 1. t_src1 = " << t_src1 << ", t_dst = " << t_dst << std::endl;
        // The field needs to be transformed
        if(get_diag_order() != 1)
        {
            init_diagonal(1);
            //std::cout << "======= Integrating level 1: diagonal" << std::endl;
            //utility :: print(diag, 0, std::cout);
        }
        //std::cout << "======= Integrating level 1: start" << std::endl;
        //utility :: print(field, t_dst, std::cout);
        field.elementwise([=] LAMBDACALLER(T lhs, T rhs) -> T {return(rhs * twodads::alpha[0][1]);}, t_dst, t_src1);
        //std::cout << "======= Integrating level 1: added t_src1" << std::endl;
        //utility :: print(field, t_dst, std::cout);
    }
    else if(tlev == 2)
    {
        //std::cout << "tlev = 2. t_src1 = " << t_src1 << ", t_src2 = " << t_src2 << ", t_dst = " << t_dst << std::endl;
        // Initialize main diagonal for second order time step
        if(get_diag_order() != 2)
        {
            init_diagonal(2);
            //std::cout << "======= Integrating level 2: diagonal" << std::endl;
            //utility :: print(diag, 0, std::cout);
        }
        // Sum up previous time steps in t_dst:
        // t_dst = alpha_2 * u_2
        //std::cout << "======= Integrating level 2: start" << std::endl;
        //utility :: print(field, t_dst, std::cout);

        field.elementwise([=] LAMBDACALLER(T lhs, T rhs) -> T { return(rhs * twodads::alpha[1][2]);}, t_dst, t_src2);
        //std::cout << "======= Integrating level 2: added t_src2" << std::endl;
        //utility :: print(field, t_dst, std::cout);

        // t_dst += alpha_1 * u_1
        field.elementwise([=] LAMBDACALLER(T lhs, T rhs) -> T { return(lhs + rhs * twodads::alpha[1][1]);}, t_dst, t_src1);
        //std::cout << "======= Integrating level 2: added t_src1" << std::endl;
        //utility :: print(field, t_dst, std::cout);
    }

    else if (tlev == 3)
    {   
        //std::cout << "tlev = 3. t_src1 = " << t_src1 << ", t_src2 = " << t_src2 << ", t_src3 = " << t_src3 << ", t_dst = " << t_dst << std::endl;
        if(get_diag_order() != 3)
        {
            init_diagonal(3);
            //std::cout << "======= Integrating level 3: diagonal" << std::endl;
            //utility :: print(diag, 0, std::cout);
        }
        // Sum up previous time steps in t_dst:
        // t_dst = alpha_3 * u^{-3}
        //std::cout << "======= Integrating level 3: start" << std::endl;
        //utility :: print(field, t_dst, std::cout);
        field.elementwise([=] LAMBDACALLER(T lhs, T rhs) -> T { return(rhs * twodads::alpha[2][3]);}, t_dst, t_src3);
        //std::cout << "======= Integrating level 3:  add t_src3" << std::endl;
        //utility :: print(field, t_dst, std::cout);
        // t_dst = alpha_2 * u^{-2}
        field.elementwise([=] LAMBDACALLER(T lhs, T rhs) -> T { return(lhs + rhs * twodads::alpha[2][2]);}, t_dst, t_src2);
        //std::cout << "======= Integrating level 3:  add t_src2" << std::endl;
        //utility :: print(field, t_dst, std::cout);
        // t_dst += alpha_1 * u^{-1}
        field.elementwise([=] LAMBDACALLER(T lhs, T rhs) -> T { return(lhs + rhs * twodads::alpha[2][1]);}, t_dst, t_src1);
        //std::cout << "======= Integrating level 3:  add t_src1" << std::endl;
        //utility :: print(field, t_dst, std::cout);
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
            add_to_boundary_left = bval_left_hat * get_rx() * field.get_geom().get_deltax() * -1.0;
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
            add_to_boundary_right = bval_right_hat * get_rx() * field.get_geom().get_deltax() * -1.0;
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


    detail :: impl_solve_tridiagonal(field, get_diag_u(), get_diag(), get_diag_l(), t_dst, allocator<T>{});

    // Field is overwritten with the solution, don't subtract boundary terms before inverse transformation
    // Inverse DFT of the result
    (*myfft).dft_c2r(reinterpret_cast<CuCmplx<T>*>(field.get_tlev_ptr(t_dst)), field.get_tlev_ptr(t_dst));
    utility :: normalize(field, t_dst);
    field.set_transformed(t_dst, false);    

    //std::cout << "======= Integrating level " << tlev << ":  result" << std::endl;
    //utility :: print(field, t_dst, std::cout);
}


template <typename T, template<typename> class allocator>
integrator_karniadakis<T, allocator> :: ~integrator_karniadakis()
{

}

#endif //INTEGRATORS_H
// End of file integrators.h