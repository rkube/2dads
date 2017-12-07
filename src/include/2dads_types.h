#ifndef TWODADS_TYPES
#define TWODADS_TYPES

#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>

#if defined(__clang__) && defined(__CUDA__) && defined(__CUDA_ARCH__)
#define CUDAMEMBER __host__ __device__
#else
#define CUDAMEMBER
#endif

#include "cucmplx.h"
#ifdef DEVICE
#include "cufft.h"
#endif //DEVICE

#ifdef HOST
#include <fftw3.h>
#endif //HOST


namespace twodads {
    /**
    .. cpp:namespace:: ::twodads
    
    */

    /**
     .. cpp:type:: real_t=double

     This can be changed to float for faster GPU performance but lower precision.

    */
    using real_t = double;

    /*
     .. cpp:type:: cmplx_t=CuCmplx<real_t>

    */
    using cmplx_t = CuCmplx<real_t>;

    #ifdef DEVICE
    using fft_handle_t = cufftHandle;
    #endif //DEVICE

    #ifdef HOST
    using fft_handle_t = fftw_plan;
    #endif //HOST

    /**
     .. cpp:member:: constexpr real_t PI = 3.141592653589793

    */
    constexpr real_t PI{3.141592653589793};

    /**
     .. cpp:member:: constexpr real_t twodads::TWOPI = 6.2831853071795864769

    */
    constexpr real_t TWOPI{6.2831853071795864769};  ///< $2.0 \pi$

    /**
     .. cpp:member:: constexpr real_t FOURPIS = 39.47841760435743

    */
    constexpr real_t FOURPIS{39.47841760435743};    ///< $4.0 * \pi^2$

    /**
     .. cpp:member:: constexpr real_t epsilon = 10^-6

    */
    constexpr real_t epsilon{0.000001};             ///< $\epsilon = 10^-6$

    /**
     .. cpp:member:: constexpr real_t Sigma = -3.185349

    */
    constexpr real_t Sigma{-3.185349};

    /**
     .. cpp:member:: constexpr size_t max_initc

     Maximal number of initial condition parameter that are parsed from input.json

    */
    constexpr size_t max_initc{6};                  ///< Maximal number of initial conditions

    /**
     .. cpp:member:: constexpr int io_w = 12

     Width of output fields for ascii output (file, terminal)

    */
    constexpr int io_w{12};                         ///< width of text fields used in cout

    /**
     .. cpp:member:: constexpr int io_p = 8

     Precision of ascii output fields

    */
    constexpr int io_p{8};                          ///< precision when printing with cout


    /**
     .. cpp:member constexpr real_t alpha

     Coefficients for stiffly stable time integration, implicit part

    */
    constexpr real_t alpha[3][4] = {{1.0, 1.0, 0.0, 0.0}, {1.5, 2.0, -0.5, 0}, {11.0/6.0, 3.0, -1.5, 1.0/3.0}};

    /**
     .. cpp:member constexpr real_t beta 

     Coefficients for stiffly stable time integration, explicit part

    */
    constexpr real_t beta[3][3] = {{1.0, 0.0, 0.0}, {2.0, -1.0, 0.0}, {3.0, -3.0, 1.0}};
    enum class tint_order_t {o1_t, o2_t, o3_t};


    // To access the alpha and beta arrays in cuda, we need a wrapper crutch.
    // Accessing alpha and beta in vec.apply in device code fails without error :/
    // http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#constexpr-variables
    //constexpr __device__ real_t get_alpha(size_t t, size_t k) {return(alpha[t][k]);};
    //constexpr __device__ real_t get_beta(size_t t, size_t k) {return(beta[t][k]);};

    // Boundary conditions: Dirichlet, Neumann, periodic, null
    /**
     .. cpp:enum-class:: bc_t

     Describes boundary condition.

     ============  =========================================================
     Value         Description
     ============  =========================================================
     bc_dirichlet  Dirichlet boundary condition
     bc_neumann    Neumann boundary condition
     bc_periodic   Periodic boundary condition
     bc_null       No boundary condition. Interpolation will throw an error.
     ============  =========================================================

    */
    enum class bc_t {bc_dirichlet, bc_neumann, bc_periodic, bc_null};

    /**
     .. cpp:enum-class:: grid_t

     ===============  =====================================================================
     Value            Description
     ===============  =====================================================================
     vertex_centered  Vertex centered grid :math:`x_n = x_\mathrm{L} + n \triangle_x`
     cell_centered    Cell centered grid :math:`x_n = x_\mathrm{L} + (n + 1/2) \triangle_x`
     ===============  =====================================================================

     Defines whether to use cell-centered or vertex-centered discretization points.

     */
    enum class grid_t {vertex_centered, cell_centered};
    // 1d DFT (in last dimension), 2d DFT
    /**
     .. cpp:enum-class:: dft_t

     Defines 1d and 2d dft type

     ====== ==========================
     Value  Description
     ====== ==========================
     dft_1d 1d DFT along y-axis
     dft_2d 2d DFT along x- and y-axis
     ====== ==========================

    */
    enum class dft_t {dft_1d, dft_2d};


    struct slab_layout_t
    {
        /**
         .. cpp:namespace-push:: slab_layout_t
        */

        /**
         .. cpp:class:: slab_layout_t

         Defines the layout and geometry of arrays.

        */

        slab_layout_t(real_t _xl, real_t _dx, real_t _yl, real_t _dy, 
                size_t _nx, size_t _pad_x, size_t _my, size_t _pad_y, 
                twodads::grid_t _grid) :
            x_left(_xl), delta_x(_dx), y_lo(_yl), delta_y(_dy), 
            Nx(_nx), pad_x(_pad_x), My(_my), pad_y(_pad_y),
            grid(_grid), 
            cellshift(grid == twodads::grid_t::vertex_centered ? 0.0 : 0.5) 
            {};


        friend std::ostream& operator<< (std::ostream& os, const slab_layout_t& sl)
        {
            os << "x_left = " << sl.get_xleft();
            os << "\tdelta_x = " << sl.get_deltax();
            os << "\ty_lo = " << sl.get_ylo();
            os << "\tdelta_y = " << sl.get_deltay();
            os << "\tNx = " << sl.get_nx();
            os << "\tpad_x = " << sl.get_pad_x();
            os << "\tMy = " << sl.get_my();
            os << "\tpad_y = " << sl.get_pad_y();
            os << "\tcellshift = " << sl.get_cellshift();
            os << std::endl;
            return(os);
        }


        /**
         .. cpp:function:: public inline bool operator==(const twodads::slab_layout_t& rhs)

         Compare whether two slab_layout_t's are equal. Currently compares
         number of discretization points and number of padding elements.

        */
        CUDAMEMBER inline bool operator==(const slab_layout_t& rhs)
        {
            return((get_nx() == rhs.get_nx()) &&
                   (get_my() == rhs.get_my()) &&
                   (get_pad_x() == rhs.get_pad_x()) &&
                   (get_pad_y() == rhs.get_pad_y())); 
        }

        /**
         .. cpp:function:: public inline real_t get_xleft() const

         Returns left domain boundary.

        */
        CUDAMEMBER inline real_t get_xleft() const {return(x_left);};

        /**
         .. cpp:function:: public inline real_t get_deltax() const

         Returns distance between discretization points in x-direction.
        */

        CUDAMEMBER inline real_t get_deltax() const {return(delta_x);};

        /**
         .. cpp:function:: public inline real_t get_xright() const

         Calculates the right domain boundary.

        */
        CUDAMEMBER inline real_t get_xright() const {return(x_left + static_cast<real_t>(Nx) * get_deltax());};

        /**
         .. cpp:function:: inline real_t get_Lx() const

         Calculates the distance between right and left domain boundary.

         */
        CUDAMEMBER inline real_t get_Lx() const {return(static_cast<real_t>(Nx) * get_deltax());};

        /**
         .. cpp:function:: inline real_t get_x(const size_t n) const
          
          :param const size_t n: Discretization point.

         Calculates the x-coordinate of a discretization point.
        */
        CUDAMEMBER inline real_t get_x(const size_t n) const {return(get_xleft() + (static_cast<real_t>(n) + get_cellshift()) * get_deltax());};

        /**
         .. cpp:function:: inline real_t get_y(const size_t m) const
          
          :param const size_t m: Discretization point.

         Calculates the y-coordinate of a discretization point.

        */
        CUDAMEMBER inline real_t get_y(const size_t m) const {return(get_ylo() + (static_cast<real_t>(m) + get_cellshift()) * get_deltay());};

        /**
         .. cpp:function:: inline real_t get_ylo() const

         Returns lower domain boundary.

        */
        CUDAMEMBER inline real_t get_ylo() const {return(y_lo);};

        /**
         .. cpp:function:: inline real_t get_deltay() const

         Returns the distance between discretization points in y-direction.
        
        */
        CUDAMEMBER inline real_t get_deltay() const {return(delta_y);};
        
        /**
         .. cpp:function:: inline real_t get_yup() const

         Calculates the upper domain boundary.

        */
        CUDAMEMBER inline real_t get_yup() const {return(y_lo + static_cast<real_t>(My) * delta_y);};

        /**
         .. cpp:function:: inline real_t get_Ly() const

         Calculates the distance between upper and lower domain boundary.

        */
        CUDAMEMBER inline real_t get_Ly() const {return(static_cast<real_t>(My) * delta_y);};

        /**
         .. cpp:function:: inline size_t get_nx() const

         Returns Nx.
   
        */
        CUDAMEMBER inline size_t get_nx() const {return(Nx);};
        /**
         .. cpp:function:: inline size_t get_pad_x() const

         Returns pad_x.

        */
        CUDAMEMBER inline size_t get_pad_x() const {return(pad_x);};

        /**
         .. cpp:function:: inline size_t get_my() const

         Returns My.

        */
        CUDAMEMBER inline size_t get_my() const {return(My);};

        /**
         .. cpp:function:: inline size_t get_pad_y() const

         Returns pad_y.

        */
        CUDAMEMBER inline size_t get_pad_y() const {return(pad_y);};

        /**
         .. cpp:function:: inline real_t get_nelem_per_t() const

         Calculates total number of array elements per time step:

           ..math:: (N_x + \mathrm{pad}_x) (M_y + \mathrm{pad}_y) 

        */
        CUDAMEMBER inline size_t get_nelem_per_t() const {return((get_nx() + get_pad_x()) * (get_my() + get_pad_y()));};

        /**
         .. cpp:function twodads::grid_t get_grid() const

         Returns grid type.

        */
        CUDAMEMBER inline twodads::grid_t get_grid() const {return(grid);};
        /**
         .. cpp:function:: inline twodads::real_t get_cellshift() const

         Returns the cell-shift. - 0 for vertex-centered grid, 1/2 for cell-centered grid.

        */
        CUDAMEMBER inline real_t get_cellshift() const {return(cellshift);};

        /**
         .. cpp:member:: const real_t x_left

         Left domain boundary

        */
        const real_t x_left;
        /**
         .. cpp:member:: const real_t delta_x

         Distance of discretization points along x-direction

        */
        const real_t delta_x;
        /**
         .. cpp:member:: const real_t y_lo

         Lower domain boundary

        */
        const real_t y_lo;
        /**
         .. cpp:member:: const real_t delta_y

         Distance of discretization points along y-direction
         
        */
        const real_t delta_y;

        /**
         .. cpp:member:: const size_t Nx

         Number of discretization points along x-direction

        */
        const size_t Nx;

        /**
         .. cpp:member:: const size_t pad_x

         Number of additional points along x-direction. Can usually be set to zero.

        */
        const size_t pad_x;

        /**
         .. cpp:member:: const size_t My

         Number of discretization points along y-direction

        */
        const size_t My;

        /**
         .. cpp:member:: const size_t pad_y

         Number of additional points along y-direction. Set to zero to store Nyquist
         Frequency of DFTs algorithms.

        */
        const size_t pad_y;

        /**
         .. cpp:member:: const twodads::grid_t grid

         Define the location of discretization points.

        */
        const twodads::grid_t grid;

        /**
         .. cpp:member:: const twodads::real_t cellshift

         Define the cell shift used in the grid. 0 for vertex-centered, 1/2 for cell-centered.

        */
        const real_t cellshift;

        // Align slab_layout_t at 8 byte boundaries(as for real_t)
        // Do this, otherwise you get differently aligned structures when
        // compiling in g++ and nvcc. Also, don't forget the -maligned-double flag
        // in g++

        /**
         .. cpp:namespace-pop::

        */
    } __attribute__ ((aligned (8)));





    template <typename T>
    class bvals_t
    {

    /**
     .. cpp:namespace-push:: template<typename T> bvals_t

    */

    /**
     .. cpp:class:: template <typename T> twodads::bvals_t

     Representation of left and right domain bonudaries

    */
        public:
            /**
             .. cpp:function:: bval_t(bc_t _bc_left, bc_t bc_right, T _bv_l, T _bv_r)
           
              :param bc_t _bc_left: Boundary condition on left domain boundary
              :param bc_t _bc_right: Boundary condition on right domain boundary
              :param T _bv_l: Boundary value on left domain boundary
              :param T _bv_r: Boundary value on right domain boundary

            Constructs a boundary value object. Pass the boundary conditions and values explicitly.

            */
            CUDAMEMBER bvals_t(bc_t _bc_left, bc_t _bc_right, T _bv_l, T _bv_r)
                               : bc_left(_bc_left), bc_right(_bc_right), 
                                 bval_left(_bv_l), bval_right(_bv_r) {}

            CUDAMEMBER bvals_t() : bc_left(twodads::bc_t::bc_null), 
                                   bc_right(twodads::bc_t::bc_null),
                                   bval_left(T(0.0)), bval_right(T(0.0)) {}
             
            /**
             .. cpp:function:: public inline bc_t get_bc_left() const

             Returns left boundary condition

            */            
            CUDAMEMBER inline bc_t get_bc_left() const {return(bc_left);};

            /**
             .. cpp:function:: public inline bc_t get_bc_right() const
             
             Returns right boundary condition

            */
            CUDAMEMBER inline bc_t get_bc_right() const {return(bc_right);};

            /**
             .. cpp:function:: public inline T get_bv_left() const

             Returns the left boundary value
            */
            CUDAMEMBER inline T get_bv_left() const {return(bval_left);};

            /**
             .. cpp:function:: public inline T get_bv_right() const

             Returns the right boundary value
            */
            CUDAMEMBER inline T get_bv_right() const {return(bval_right);};

            /**
             .. cpp:function:: public inline bool operator==(const bvals_t& rhs) const

             Returns true if both boundary conditions and values are identical. Returns false otherwise.
            */
            CUDAMEMBER inline bool operator==(const bvals_t& rhs) const
            {
                if((bc_left  == rhs.bc_left)  && (bc_right == rhs.bc_right)
                   && (std::fabs(bval_left - rhs.bval_left) < twodads::epsilon)
                   && (std::fabs(bval_right - rhs.bval_right) < twodads::epsilon))
                {
                    return (true);
                }
                return (false);
            }

            CUDAMEMBER inline bool operator!=(const bvals_t& rhs) const
            {
                return(!(*this == rhs));
            }

        private: 
            // The boundary conditions on the domain border
            const bc_t bc_left;
            const bc_t bc_right;

            // The boundary values on the domain border
            const T bval_left;
            const T bval_right;

    /**
     .. cpp:namespace-pop::

    */
    };


    class stiff_params_t
    {
    /**
     .. cpp:namespace-push stiff_params_t

    */

    /**
     .. cpp:class:: stiff_params_t

     Stores data for the Karniadakis time integrator (stiffly-stable time integration scheme).

    */
        public:
            CUDAMEMBER stiff_params_t(real_t dt, real_t lx, real_t ly, real_t d, real_t h, size_t my, size_t nx21, size_t l) :
                delta_t(dt), length_x(lx), length_y(ly), diff(d), hv(h), My(my), Nx21(nx21), level(l) {};
            CUDAMEMBER real_t get_deltat() const {return(delta_t);};
            CUDAMEMBER real_t get_lengthx() const {return(length_x);};
            CUDAMEMBER real_t get_lengthy() const {return(length_y);};
            CUDAMEMBER real_t get_diff() const {return(diff);};
            CUDAMEMBER real_t get_hv() const {return(hv);};
            CUDAMEMBER size_t get_my() const {return(My);};
            CUDAMEMBER size_t get_nx21() const {return(Nx21);};
            CUDAMEMBER size_t get_tlevs() const {return(level);};
        private:
            const real_t delta_t;
            const real_t length_x;
            const real_t length_y;
            const real_t diff;
            const real_t hv;
            const size_t My;
            const size_t Nx21;
            const size_t level;
    } __attribute__ ((aligned (8)));
    /**
     .. cpp:namespace-pop

    */

    /**
      .. cpp:enum-class:: dyn_field_t

      =======  ===========
      Value    Description
      =======  ===========
      f_theta  theta      
      f_tau    tau        
      f_omega  omega      
      =======  ===========

      Enumerates available fields for time integration.

    */
    enum class dyn_field_t 
    {
        f_theta, ///< theta, thermodynamic variable
        f_tau, ///<tau, electron temperature
    	f_omega  ///< omega, vorticity
    };

    /**
     .. cpp:enum-class:: twodads::field_t

     =========== ==========================
     Value       Description
     =========== ==========================
     f_theta     Theta
     f_theta_x   theta_x
     f_theta_y   theta_y
     f_omega     omega
     f_omega_x   omega_x
     f_omega_y   omega_y
     f_tau       tau
     f_tau_x     tau_x
     f_tau_y     tau_y
     f_strmf     strmf
     f_strmf_x   strmf_x
     f_strmf_y   strmf_y
     f_tmp       tmp
     f_tmp_x     tmp_x
     f_tmp_y     tmp_y 
     f_theta_rhs theta_rhs
     f_tau_rhs   tau_rhs
     f_omega_rhs omega_rhs
     =========== ==========================

    */
    enum class field_t 
    {
        f_theta,    ///< \f$\theta\f$
        f_theta_x,  ///< \f$\theta_x  \f$, radial derivative of theta
        f_theta_y,  ///< \f$ \theta_y \f$, poloidal derivative of theta
        f_omega,    ///< \f$ \Omega   \f$
        f_omega_x,  ///< \f$ \Omega_x \f$
        f_omega_y,  ///< \f$ \Omega_x \f$ 
        f_tau,      ///< \f$ \tau \f$
        f_tau_x,    ///< \f$ \tau_x \f$
        f_tau_y,    ///< \f$ \tau_x \f$ 
        f_strmf,    ///< \f$ \phi     \f$  
        f_strmf_x,  ///< \f$ \phi_x   \f$ 
        f_strmf_y,  ///< \f$ \phi_y   \f$   
        f_tmp,      ///< tmp field
        f_tmp_x,    ///< tmp field
        f_tmp_y,    ///< tmp field
        f_theta_rhs,///< rhs for time integration
        f_tau_rhs,  ///< rhs for time integration
        f_omega_rhs ///< rhs for time integration
    };

               
    /**
     .. cpp:enum-class:: twodads::init_fun_t

     ===================  ========================================
     Value                Description
     ===================  ========================================
     init_NA              Fallback value in case of parsing errors
     init_gaussian        Initializes a 2d Gaussian function
     init_constant        Initializes a constant value
     init_sine            Initializes a sine function
     init_turbulent_bath  Initializes a turbulent bath
     init_lamb_dipole     Initializes a Lamb dipole
     ===================  ========================================

     Enumerates available initialization functions.

    */
    enum class init_fun_t 
    {
        init_NA,              ///< Not available, throws an error
        init_gaussian,        ///< Initializes gaussian profile 
        init_constant,        ///< Initialize field with constant value
        init_sine,            ///< Initializes sinusoidal profile for theta
        init_mode,            ///< Initializes single modes for theta_hat
        init_turbulent_bath,  ///< Initialize all modes randomly
        init_lamb_dipole      ///<Lamb Dipole
    };

    /**
     .. cpp:enum-class:: twodads::rhs_t

      =====================  =========================================================
      Value                  Description
      =====================  =========================================================
      rhs_theta_null         No explicit terms. Defaults to diffusion / hyperviscosity
      rhs_theta_lin          Advection with electric drift, no logarithmic density
      rhs_theta_log          Advection with electric drift, logarithmic density formulation
      rhs_omega_null         No explicit terms. Defaults to diffusion / hyperviscosity
      rhs_omega_ic           Interchange model
      rhs_tau_null           No explicit terms. Defaults to diffusion / hyperviscosity
      rhs_tau_log            Advection with electric drift, logarithmic temperature
      =====================  =========================================================

      Available right-hand side models.
     */
    enum class rhs_t 
    {
        rhs_theta_lin, 
        rhs_theta_log,
        rhs_theta_null, 
        rhs_omega_ic,
        rhs_omega_null, 
        rhs_tau_null,
        rhs_tau_log
    };

    /**
      ..cpp:enum-class twodads::diagnostic_t

       ==============  ==========================================
       Value           Description
       ==============  ==========================================
       diag_com_theta  Center-of-mass diagnostics for theta field
       diag_com_tau    Center-of-mass diagnostics for tau field
       diag_max_theta  Maximum diagnostics for theta field
       diag_max_tau    Maximum diagnostics for tau field
       diag_energy     Energy integrals
       diag_probes     Probe diagnostics
       ==============  ==========================================

      Available diagnostic routines.

     */
    enum class diagnostic_t 
    {
        //diag_blobs,         ///< Information on blob dynamics
        diag_com_theta,     ///< COM dynamics on theta field
        diag_com_tau,       ///< COM dynamics of tau field
        diag_max_theta,     ///< max dynamics on theta
        diag_max_tau,       ///< max dynamics on tau
        //diag_max_omega,     ///< max dynamics on omega
        //diag_max_strmf,     ///< max dynamics on strmf
        diag_energy,        ///< Energetics for turbulence simulations
        diag_probes,        ///< Time series from probes
    };
    
    /**
     .. cpp:enum-class:: twoadads::output_t

      =========  =====================
      Value      Description
      =========  =====================
      o_theta    Theta
      o_theta_x  x-derivative of theta
      o_theta_y  y-derivative of theta
      o_omega    omega
      o_omega_x  x-derivative of omega
      o_omega_y  y-derivative of omega
      o_tau      tau
      o_tau_x    x-derivative of tau
      o_tau_y    y-derivative of tau
      o_strmf    stream function phi
      o_strmf_x  x-derivative of phi
      o_strmf_y  y-derivative of phi
      =========  =====================

     Enumerates fields available for binary output.
     
     */
    enum class output_t 
    {
        o_theta,  
        o_theta_x,
        o_theta_y,
        o_omega, 
        o_omega_x, 
        o_omega_y, 
        o_tau, 
        o_tau_x, 
        o_tau_y,
        o_strmf,
        o_strmf_x, 
        o_strmf_y
    };

    /**
     .. cpp:namespace-pop

    */
};


#endif //2dads_types
//End of file 2dads_types.h
