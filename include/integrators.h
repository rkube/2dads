/*
 * Provide time integrators to be used by slab data types
 *
 */

#include <iostream>
#include "cuda_array_bc_nogp.h"
#include "error.h"


// Generic integrator class.
template <typename allocator>
class integrator 
{
    public:
        using value_t = typename my_allocator_traits<allocator> :: value_type;
        using real_t = cuda::real_t;
        using cmplx_t = cuda::cmplx_t;

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


// Karniadakis stiffly-stable time integrator. Sub-class of integrator
template <typename allocator>
class integrator_karniadakis : public integrator<allocator>
{
    public:
        using value_t = typename integrator<allocator>::value_t;
        using real_t = cuda::real_t;
        using cmplx_t = cuda::cmplx_t;

        using integrator<allocator>::get_array;
        using integrator<allocator>::get_geom;
        using integrator<allocator>::get_params;
        using integrator<allocator>::get_bvals;

        integrator_karniadakis(const cuda::slab_layout_t,
                               const cuda::bvals_t<cuda::real_t>,
                               const cuda::stiff_params_t);
        ~integrator_karniadakis();
        
        virtual void initialize_field(const cuda_array_bc_nogp<allocator>*, const size_t);
        virtual void integrate(cuda_array_bc_nogp<allocator>*);

        inline int get_my_int() const {return(My_int);};
        inline int get_my21_int() const {return(My21_int);};
        inline int get_nx_int() const {return(Nx_int);};
        inline value_t get_inv_dx2() const {return(inv_dx2);};
    private:
        // Diagonal elements for elliptic solver
        cmplx_t* d_diag;
        cmplx_t* d_diag_u;
        cmplx_t* d_diag_l;
        cmplx_t* h_diag;

        // Array boundaries etc.
        const int My_int;
        const int My21_int;
        const int Nx_int;
        const value_t inv_dx2;

        solvers::elliptic elliptic_solver;
};


template <typename allocator>
integrator_karniadakis<allocator> :: integrator_karniadakis(const cuda::slab_layout_t _geom, 
                                                            const cuda::bvals_t<cuda::real_t> _bvals_theta, 
                                                            const cuda::stiff_params_t _stiff_params) : 
    d_diag{nullptr},
    d_diag_u{nullptr},
    d_diag_l{nullptr},
    h_diag{nullptr},
    My_int{static_cast<int>(_geom.get_my())},
    My21_int{static_cast<int>((_geom.get_my() + _geom.get_pad_y()) / 2)},
    Nx_int{static_cast<int>(_geom.get_nx())},
    inv_dx2{1.0 / (_geom.get_deltax() * _geom.get_deltax())}, 
    elliptic_solver(_geom),
    integrator<allocator>(_geom, _bvals_theta, _stiff_params) 
{
    const size_t My21{(get_geom().get_my() + get_geom().get_pad_y()) / 2};

    // Allocate diagonal elements
    gpuErrchk(cudaMalloc((void**) &d_diag, get_geom().get_nx() * My21 * sizeof(cmplx_t)));
    gpuErrchk(cudaMalloc((void**) &d_diag_u, get_geom().get_nx() * My21 * sizeof(cmplx_t)));
    gpuErrchk(cudaMalloc((void**) &d_diag_l, get_geom().get_nx() * My21 * sizeof(cmplx_t)));

    // Allocate host vectors
    h_diag = new cmplx_t[get_geom().get_nx()];
    cmplx_t* h_diag_u = new cmplx_t[get_geom().get_nx()];
    cmplx_t* h_diag_l = new cmplx_t[get_geom().get_nx()];

    value_t ky2{0.0};
    value_t rx{0.0};
    for(size_t m = 0; m < static_cast<size_t>(get_my21_int()); m++)
    {
        ky2 = cuda::TWOPI * cuda::TWOPI * static_cast<value_t>(m * m) / (get_geom().get_Ly() * get_geom().get_Ly());
        rx = get_params().get_deltat() * (get_params().get_diff() - ky2) * inv_dx2;
        for(size_t n = 0; n < get_geom().get_nx(); n++)
        {
            h_diag[n] = 1.0 + 2.0 * rx;
        }
        
        // Update the first and last row for correct treatment of the boundary conditions
        if(m == 0)
        {
            switch(get_bvals().get_bc_left())
            {
                case cuda::bc_t::bc_dirichlet:
                    h_diag[0] = h_diag[0] + rx - 2.0 * rx * get_bvals().get_bv_left() * cuda::TWOPI * static_cast<value_t>(get_geom().get_my()) / get_geom().get_Ly();
                    break;
                case cuda::bc_t::bc_neumann:
                    h_diag[0] = h_diag[0] - rx - rx * get_geom().get_deltax() * get_bvals().get_bv_left() * cuda::TWOPI * static_cast<value_t>(get_geom().get_my()) / get_geom().get_Ly();
                    break;
                case cuda::bc_t::bc_periodic:
                    break;
            }
            switch(get_bvals().get_bc_right())
            {
                case cuda::bc_t::bc_dirichlet:
                    h_diag[get_geom().get_nx() - 1] = h_diag[get_geom().get_nx() - 1] + rx - 2.0 * rx * get_bvals().get_bv_right() * cuda::TWOPI * static_cast<value_t>(get_geom().get_my()) / get_geom().get_Ly();
                    break;
                case cuda::bc_t::bc_neumann:
                    h_diag[get_geom().get_nx() - 1] = h_diag[get_geom().get_nx() - 1] - rx - rx * get_bvals().get_bv_right() * cuda::TWOPI * static_cast<value_t>(get_geom().get_my()) / get_geom().get_Ly();
                    break;
                case cuda::bc_t::bc_periodic:
                    break;
            }
        }
        gpuErrchk(cudaMemcpy(d_diag + m * get_geom().get_nx(), h_diag, get_geom().get_nx() * sizeof(cmplx_t), cudaMemcpyHostToDevice));
    }

    // Initialize upper and lower diagonal elements
    for(size_t n = 0; n < get_geom().get_nx(); n++)
    {
        h_diag_u[n] = -1.0 * rx;
        h_diag_l[n] = -1.0 * rx;
    }
    h_diag_u[get_geom().get_nx() - 1] = 0.0;
    h_diag_l[0] = 0.0;

    for(size_t m = 0; m < static_cast<size_t>(get_my21_int()); m++)
    {
        gpuErrchk(cudaMemcpy(d_diag_l + m * get_geom().get_nx(), h_diag_l, get_geom().get_nx() * sizeof(cmplx_t), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_diag_u + m * get_geom().get_nx(), h_diag_u, get_geom().get_nx() * sizeof(cmplx_t), cudaMemcpyHostToDevice));
    }
    delete [] h_diag_l;
    delete [] h_diag_u;
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
    elliptic_solver.solve((cuDoubleComplex*) out, (cuDoubleComplex*) get_array().get_array_d(get_params().get_tlevs() - 1),
                          (cuDoubleComplex*)d_diag_l, (cuDoubleComplex*) d_diag, nullptr);
}


template <typename allocator>
integrator_karniadakis<allocator> :: ~integrator_karniadakis()
{
    delete [] h_diag;
    cudaFree(d_diag_l);
    cudaFree(d_diag_u);
    cudaFree(d_diag); 
}

// End of file integrators.h
