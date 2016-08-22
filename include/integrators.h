/*
 * Provide time integrators to be used by slab data types
 *
 */


#ifndef INTEGRATORS_H
#define INTEGRATORS_H

#include <iostream>
#include "cuda_array_bc_nogp.h"
#include "error.h"


namespace detail
{
#ifdef __CUDACC__
#endif //__CUDACC__

    template <typename T>
    void impl_integrate(cuda_array_bc_nogp<T, allocator_host>& in,
                        cuda_array_bc_nogp<T, allocator_host>& rhs,
                        const size_t t_src,
                        cuda_array_bc_nogp<twodads::cmplx_t, allocator_host>& diag,
                        cuda_array_bc_nogp<twodads::cmplx_t, allocator_host>& diag_l,
                        cuda_array_bc_nogp<twodads::cmplx_t, allocator_host>& diag_u)
    {


    }

}


// Karniadakis stiffly-stable time integrator. Sub-class of integrator
template <typename T, template<typename> class allocator>
class integrator_karniadakis
{
    public:
        //using real_t = twodads::real_t;
        using cmplx_t = twodads::cmplx_t;
        using cmplx_arr = cuda_array_bc_nogp<cmplx_t, allocator>;

        //integrator_karniadakis(const cuda_array_bc_nogp<T, allocator>&,
        //                       const twodads::stiff_params_t);
        integrator_karniadakis(const twodads::slab_layout_t _sl, const twodads::bvals_t<T> _bv, const twodads::stiff_params_t _sp) :
            geom{_sl}, bvals{_bv}, stiff_params{_sp},  
            geom_transpose{get_geom().get_ylo(),
                           get_geom().get_deltay(),
                           get_geom().get_xleft(),
                           get_geom().get_deltax(),
                           (get_geom().get_my() + get_geom().get_pad_y()) / 2, 0,
                           get_geom().get_nx(), 0,
                           get_geom().get_grid()},        
            My_int{static_cast<int>(get_geom().get_my())},
            My21_int{static_cast<int>(get_geom().get_my() + get_geom().get_pad_y()) / 2},
            Nx_int{static_cast<int>(get_geom().get_nx())},
            // Pass a complex bvals_t to these guys. They don't really need it though.
            diag(get_geom_transpose(), twodads::bvals_t<CuCmplx<T>>(twodads::bc_t::bc_dirichlet, twodads::bc_t::bc_dirichlet, twodads::bc_t::bc_periodic, twodads::bc_t::bc_periodic, cmplx_t{0.0}, cmplx_t{0.0}, cmplx_t{0.0}, cmplx_t{0.0}), 1),
            diag_l(get_geom_transpose(), twodads::bvals_t<CuCmplx<T>>(twodads::bc_t::bc_dirichlet, twodads::bc_t::bc_dirichlet, twodads::bc_t::bc_periodic, twodads::bc_t::bc_periodic, cmplx_t{0.0}, cmplx_t{0.0}, cmplx_t{0.0}, cmplx_t{0.0}), 1),
            diag_u(get_geom_transpose(), twodads::bvals_t<CuCmplx<T>>(twodads::bc_t::bc_dirichlet, twodads::bc_t::bc_dirichlet, twodads::bc_t::bc_periodic, twodads::bc_t::bc_periodic, cmplx_t{0.0}, cmplx_t{0.0}, cmplx_t{0.0}, cmplx_t{0.0}), 1)
        {
            init_diagonals();
        }
        ~integrator_karniadakis();
        
        void integrate(cuda_array_bc_nogp<T, allocator>&, const size_t);

        // init_diagonals() initializes the diagonal elements used for elliptic solver.
        // It should really be private, but then nvcc complains
        // An explicit __device__ lambda cannot be defined in a member function that has private or protected access within its class
        void init_diagonals();


        inline twodads::slab_layout_t get_geom() const {return(geom);};
        inline twodads::bvals_t<T> get_bvals() const {return(bvals);};
        inline twodads::stiff_params_t get_params() const {return(stiff_params);};
        inline twodads::slab_layout_t get_geom_transpose() const {return(geom_transpose);};

        inline int get_my_int() const {return(My_int);};
        inline int get_my21_int() const {return(My21_int);};
        inline int get_nx_int() const {return(Nx_int);};
    private:
        // Diagonal elements for elliptic solver
        const twodads::slab_layout_t geom;
        const twodads::bvals_t<twodads::real_t> bvals;
        const twodads::stiff_params_t stiff_params;

        // Transposed complex layout for the diagonals. See derivatives.h
        const twodads::slab_layout_t geom_transpose;

        // Array boundaries etc.
        const int My_int;
        const int My21_int;
        const int Nx_int;

        cmplx_arr diag;
        cmplx_arr diag_l;
        cmplx_arr diag_u;


        //solvers::elliptic elliptic_solver;
};


template <typename T, template<typename> class allocator>
void integrator_karniadakis<T, allocator> :: init_diagonals()
{
    // diag[_l, _u] are transposed. Use m = 0..Nx-1, n = 0..My/2 and interchange x and y
    const T Lx{geom.get_deltax() * 2 * (get_geom().get_nx() - 1)};
    const CuCmplx<T> inv_dx2{1.0 / (get_geom().get_deltay() * get_geom().get_deltay())};
    const CuCmplx<T> rx{get_params().get_diff() * get_params().get_deltat() / (get_geom().get_deltax() * get_geom().get_deltax())};
    const CuCmplx<T> dx{get_geom().get_deltax()};

    const CuCmplx<T> bv_left{get_bvals().get_bv_left()};
    const CuCmplx<T> bv_right{get_bvals().get_bv_right()};

    std::cout << "rx = " << rx << ", dx = " << dx << ", main = " << rx * (2.0) + 1.0 << std::endl;
    diag.apply([=] LAMBDACALLER (CuCmplx<T> input, const size_t n, const size_t m, twodads::slab_layout_t geom) -> CuCmplx<T>
    {
        // ky runs with index n (the kernel addressing function, see cuda::thread_idx
        // We are transposed, Lx = dx * (2 * nx - 1) as we have cut nx roughly in half
        const CuCmplx<T> ky2 = twodads::TWOPI * twodads::TWOPI * static_cast<T>(n * n) / (Lx * Lx);
        return(rx * (2.0) + ky2 * rx * dx * dx + 1.0);
    }, 0);

    switch(get_bvals().get_bc_left())
    {
        std::cout << "Left dirichlet: " << bv_left << std::endl;
        case twodads::bc_t::bc_dirichlet:
            diag.apply([=] LAMBDACALLER (CuCmplx<T> input, const size_t n, const size_t m, twodads::slab_layout_t geom) -> CuCmplx<T>
            {
                const CuCmplx<T> ky2 = twodads::TWOPI * twodads::TWOPI * static_cast<T>(n * n) / (Lx * Lx);
                if(n == 0 && m == 0)
                    return(rx * (3.0) + ky2 * rx * dx * dx - rx * (-2.0) * bv_left + 1.0);
                return(input);
            }, 0);
            break;

        case twodads::bc_t::bc_neumann:
            std::cout << "Left Neumann: " << bv_left << std::endl;
            diag.apply([=] LAMBDACALLER (CuCmplx<T> input, const size_t n, const size_t m, twodads::slab_layout_t geom) -> CuCmplx<T>
            {
                const CuCmplx<T> ky2 = twodads::TWOPI * twodads::TWOPI * static_cast<T>(n * n) / (Lx * Lx);
                if(n == 0 && m == 0)
                    return(rx + ky2 * rx * dx * dx - rx * dx * bv_left + 1.0);
                return(input);
            }, 0);
            break;

        case twodads::bc_t::bc_periodic:
            std::cerr << "intgrator_karniadakis :: init_diagonals(): periodic boundary conditions not supported" << std::endl; 
            break;
    }

    switch(get_bvals().get_bc_right())
    {
        case twodads::bc_t::bc_dirichlet:
            std::cout << "Right dirichlet: " << bv_right << std::endl;
            diag.apply([=] LAMBDACALLER (CuCmplx<T> input, const size_t n, const size_t m, twodads::slab_layout_t geom) -> CuCmplx<T>
            {
                const CuCmplx<T> ky2 = twodads::TWOPI * twodads::TWOPI * static_cast<T>(n * n) / (Lx * Lx);
                if(n == 0 && m == geom.get_my() - 1)
                    return(rx * (3.0) + ky2 * rx * dx * dx - rx * (-2.0) * bv_right + 1.0);
                return(input);
            }, 0);
            break;

        case twodads::bc_t::bc_neumann:
            std::cout << "Right Neumann: " << bv_right << std::endl;
            diag.apply([=] LAMBDACALLER (CuCmplx<T> input, const size_t n, const size_t m, twodads::slab_layout_t geom) -> CuCmplx<T>
            {
                const CuCmplx<T> ky2 = twodads::TWOPI * twodads::TWOPI * static_cast<T>(n * n) / (Lx * Lx);
                if(n == 0 && m == geom.get_my() - 1)
                    return(rx + ky2 * rx * dx * dx - rx * dx * bv_right + 1.0);
                return(input);
            }, 0);
            break;

        case twodads::bc_t::bc_periodic:
            break; 
    }

    diag_l.apply([=] LAMBDACALLER (CuCmplx<T> dummy, const size_t n, const size_t m, twodads::slab_layout_t geom) -> CuCmplx<T>
    {
        // CUBLAS requires the first element in the lower diagonal to be zero.
        // Remember to shift the pointer in the MKL implementation when passing to the
        // MKL caller routine in solver
        if(m == 0)
            return(0.0);
        return(inv_dx2);
    }, 0);

    diag_u.apply([=] LAMBDACALLER (CuCmplx<T> dummy, const size_t n, const size_t m, twodads::slab_layout_t geom) -> CuCmplx<T>
    {
        // CUBLAS requires the last element in the upper diagonal to be zero.
        // Remember to shift the pointer in the MKL implementation when passing to the
        // MKL caller routine in solver
        if(m == geom.get_my() - 1)
            return(0.0);  
        return(inv_dx2);  
    }, 0);

    std::cout << "Main diagonal:" << std::endl;
    utility :: print(diag, 0, std::cout);

}


// Integrate the fields in time. Returns the 1d-dft into out
// see also derivative :: invert_laplace
template <typename T, template<typename> class allocator>
void integrator_karniadakis<T, allocator> :: integrate(cuda_array_bc_nogp<T, allocator>& field, const size_t tlev) 
{
    std::cout << "Integrating, tlev = " << tlev << std::endl;
    // The field needs to be transformed
    assert(field.is_transformed());

    // Update the diagonals

    //elliptic_solver.solve((cuDoubleComplex*) out, (cuDoubleComplex*) get_array().get_array_d(get_params().get_tlevs() - 1),
    //                      (cuDoubleComplex*)d_diag_l, (cuDoubleComplex*) d_diag, nullptr);
}


template <typename T, template<typename> class allocator>
integrator_karniadakis<T, allocator> :: ~integrator_karniadakis()
{

}

#endif //INTEGRATORS_H
// End of file integrators.h