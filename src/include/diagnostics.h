/* 
 *   To write diagnostics for turbulence simulations
 */ 


#ifndef DIAGNOSTICS_H
#define DIAGNOSTICS_H

#include "2dads_types.h"
#include "slab_config.h"
#include "cuda_array_bc_nogp.h"



using namespace std;

class diagnostic_t {
    /**
     .. cpp:namspace-push:: diagnostic_c

    */

    /** 
     .. cpp:class:: diagnostic_t

    */
	public:
        using value_t = twodads::real_t;
        // Pointer type to diagnostic member functions
        // All diagnostic functions are required to have the same signature
        using dfun_ptr_t = void (diagnostic_t::*)(const twodads::real_t) const;

#ifdef DEVICE
        using arr_real = cuda_array_bc_nogp<value_t, allocator_device>;
#endif

#ifdef HOST
        using arr_real = cuda_array_bc_nogp<value_t, allocator_host>;
#endif

        /**
         .. cpp:function diagnostic_t(const slab_config_js& sc)

         :param const slab_config_js& sc: Slab configuration

        */
		diagnostic_t(const slab_config_js&);
    
		~diagnostic_t();

        /**
         .. cpp:function init_field_ptr(const twodads::field_t fname, cuda_array_bc_nogp<value_t, allocator>* fptr)

         :param const twodads::field_t fname: Name of the field
         :param cuda_array_bc_nogp<value_t, allocator>*: Pointer to the data field
   
        Initializes member pointers to the data fields used to compute the diagnostics.

        */
        void init_field_ptr(twodads::field_t, arr_real*);

        /**
         .. cpp:function write_diagnostics (const twodads::real_t time, const slab_config_js& sc)

         :param const twodads::real_t time: Simulation time
         :param const slab_config_js& sc: Slab configuration

        Execute the diagnostic routines speicified in the configuration for the specified time.

        */
        void write_diagnostics(const twodads::real_t, const slab_config_js&);

    private:
        const twodads::slab_layout_t slab_layout;

        void diag_com(const twodads::field_t fieldname, const std::string filename, const twodads::real_t time) const;

        /**
         .. cpp:function inline void diag_com_theta(const twodads::real_t time) const

         :param const twodads::real_t time: Time for simulation output.

         Writes center-of-mass diagnostics of theta field. Forwards to diag_com.
        
        */
        inline void diag_com_theta(const twodads::real_t time) const
        { diag_com(twodads::field_t::f_theta, filename_str_map.at(twodads::diagnostic_t::diag_com_theta), time); }

        /**
         .. cpp:function inline void diag_com_tau(const twodads::real_t time) const

         :param const twodads::real_t time: Time for simulation output.

         Writes center-of-mass diagnostics of tau field. Forwards to diag_com.
        
        */
        inline void diag_com_tau(const twodads::real_t time) const
        { diag_com(twodads::field_t::f_tau, filename_str_map.at(twodads::diagnostic_t::diag_com_tau), time); }


        void diag_max(const twodads::field_t fieldname, const std::string filename, const twodads::real_t time) const;
        /**
         .. cpp:function inline void diag_max_theta(const twodads::real_t time) const

         :param const twodads::real_t time: Time for simulation output.

         Writes maximum diagnostics of theta field. Forwards to diag_max.
        
        */
        inline void diag_max_theta(const twodads::real_t time) const
        { diag_max(twodads::field_t::f_theta, filename_str_map.at(twodads::diagnostic_t::diag_max_theta), time); }

        /**
         .. cpp:function inline void diag_max_tau(const twodads::real_t time) const

         :param const twodads::real_t time: Time for simulation output.

         Writes maximum diagnostics of tau field. Forwards to diag_max.
        
        */
        inline void diag_max_tau(const twodads::real_t time) const
        {diag_max(twodads::field_t::f_tau, filename_str_map.at(twodads::diagnostic_t::diag_max_tau), time); }

        static const std::map<twodads::diagnostic_t, std::string> header_str_map;
        static const std::map<twodads::diagnostic_t, std::string> filename_str_map;

        std::map<twodads::diagnostic_t, dfun_ptr_t> diag_func_map;

        // We really want the diagnostics to be read only. So mark everything as const.
        // This leads to some weird const-casting, as in init_field_ptr. But it seems
        // to work out. The data pointers are read-only and the diagnostic output is usable.
		
        /*
         .. cpp:function:
        void diag_com(const twodads::field_t, const std::string, const twodads::real_t) const;
        ///@brief Write out max diagnostics
        void diag_max(const twodads::field_t, const std::string, const twodads::real_t) const;

        ///@brief Compute energy integrals for various turbulence models
        ///@param time Time of output
        ///@detailed energy.dat: t E K T U W D1 D2 D3 D4 D5 D6 D7 D8 D9 D10 D11 D12 D13 <br>
        ///@detailed \f$E = \frac{1}{2} \int \mathrm{d}A\, \left( \widetilde{n} + \phi_x^2  + \phi_y^ \right)^2 \f$ <br>
        ///@detailed \f$D_{1} = \frac{1}{2A} \int \mathrm{d}A\, n^2 \f$  <br>
        ///@detailed \f$D_{2} = \frac{1}{2A} \int \mathrm{d}A\, \left( \nabla_\perp \phi \right)^2 \f$  <br>
        ///@detailed \f$D_{3} = \frac{1}{2A} \int \mathrm{d}A\, \Omega^2\f$ <br>
        ///@detailed \f$D_{4} = -\frac{1}{A} \int \mathrm{d}A\, \widetilde{n}\widetilde{\phi_y}\f$ <br>
        ///@detailed \f$D_{5} = \frac{1}{A} \int \mathrm{d}A\, \left( \overline{n} - \overline{\phi} \right)^2\f$ <br>
        ///@detailed \f$D_{6} = \frac{1}{A} \int \mathrm{d}A\, \left( \widetilde{n} - \widetilde{\phi} \right)^2\f$ <br>
        ///@detailed \f$D_{7} = \frac{1}{A} \int \mathrm{d}A\, \bar{\phi} \left( \bar{\phi} - \bar{n} \right)\f$ <br>
        ///@detailed \f$D_{8} = \frac{1}{A} \int \mathrm{d}A\, \widetilde{\phi} \left( \widetilde{\phi} - \widetilde{n} \right) \f$ <br>
        ///@detailed \f$D_{9} = \frac{1}{A} \int \mathrm{d}A\, 0\f$ <br>
        ///@detailed \f$D_{10} = \frac{1}{A} \int \mathrm{d}A\, \nabla_\perp^2 n\f$ <br>
        ///@detailed \f$D_{11} = \frac{1}{A} \int \mathrm{d}A\, \nabla_\perp \phi \nabla_\perp \Omega\f$ <br>
        ///@detailed \f$D_{12} = \frac{1}{A} \int \mathrm{d}A\, n_{xxx}^2 + n_{yyy}^2\f$ <br>
        ///@detailed \f$D_{13} = \frac{1}{A} \int \mathrm{d}A\, \phi_{xxx} \Omega_{xxx} + \phi_{yyy} \Omega_{yyy}\f$ <br>
        //void diag_energy(twodads::real_t const);
        /// @brief write output for probes
        /// @detailed Probe layout is in a square grid. Specifying num_probes = N_pr gives
        /// @detailed N_pr * N_pr probes in a equidistant grid, starting at n=m=0
        /// @detailed probe write n_tilde, n, phi, phi_tilde, omega, omega_tilde, phi_y_tilde, phi_x, phy_x_tilde
        //void diag_probes(twodads::real_t const);
        /// @brief Write simulation parameters to log file
        //void diag_rhs(twodads::real_t const);
        /// @brief Creates a log file
		
        void write_logfile();		
	   
        // const arr_real* makes the fields themself read-only.
        const arr_real *theta_ptr, *theta_x_ptr, *theta_y_ptr;
        const arr_real *omega_ptr, *omega_x_ptr, *omega_y_ptr;
        const arr_real *tau_ptr, *tau_x_ptr, *tau_y_ptr;
        const arr_real *strmf_ptr, *strmf_x_ptr, *strmf_y_ptr;
        // assign data pointers to names of the fields for internal use.
        const std::map<twodads::field_t, const arr_real**> data_ptr_map;

        //unsigned int n_probes; /// Number of probes
		//bool use_log_theta; /// Is the density field logarithmic? If true, use exp(theta) in all routines
        //twodads::real_t theta_bg; /// Subtract uniform background on theta
        */

    /**
     .. cpp:namespace-pop::

    */
};

#endif //DIAGNOSTICS_H	
