#include <iostream>
#include <vector>
#include <iomanip>
#include "slab_cuda.h"
#include "diagnostics_cu.h"
#include "output.h"

using namespace std;


class mode_struct{
    public:
        mode_struct(uint kx_, uint ky_) : kx(kx_), ky(ky_) {};
        uint kx;
        uint ky;
};


int main(void)
{
    slab_config my_config;
    my_config.consistency();

    slab_cuda slab(my_config);
    output_h5 slab_output(my_config);
    diagnostics_cu slab_diag_cu(my_config);

    // Create and populate list of modes specified in input.ini
    //vector<mode_struct> mode_list;
    //vector<double> initc = my_config.get_initc();
    //const unsigned int num_modes = my_config.get_initc().size() / 2;
    //cout << "Parsing " << num_modes << " modes in main():\n";
    //for(uint n = 0; n < num_modes; n++)
    //{
    //    cout << "mode " << n << ": kx=" << initc[2 * n + 1] << ", ky=" << initc[2 * n + 1] << "\n";
    //    mode_list.push_back(mode_struct(initc[2 * n], initc[2 * n + 1]));
    //}

    twodads::real_t time{0.0};
    twodads::real_t delta_t{my_config.get_deltat()};

    const unsigned int tout_full(ceil(my_config.get_tout() / my_config.get_deltat()));
    const unsigned int tout_diag{my_config.get_tdiag() / my_config.get_deltat()};
    const unsigned int num_tsteps(ceil(my_config.get_tend() / my_config.get_deltat()));
    const unsigned int tlevs = my_config.get_tlevs();
    unsigned int t = 1;

    slab.init_dft();
    slab.initialize();
    slab.rhs_fun(my_config.get_tlevs() - 1);

    slab_diag_cu.update_arrays(slab);
    slab_diag_cu.write_diagnostics(time, my_config);

    cout << "Output every " << tout_full << " steps\n";

    slab_output.write_output(slab, time);

    // Integrate the first two steps with a lower order scheme
    for(t = 1; t < tlevs - 1; t++)
    {
        //for(auto mode : mode_list)
        //{
        //    cout << "mode with kx = " << mode.kx << " ky = " << mode.ky << "\n";
        //    slab.integrate_stiff_debug(twodads::field_k_t::f_theta_hat, t + 1, mode.kx, mode.ky);
        //} 
        slab.integrate_stiff(twodads::field_k_t::f_theta_hat, t + 1);

        time += delta_t;
        //slab.inv_laplace(twodads::field_k_t::f_omega_hat, twodads::field_k_t::f_strmf_hat, 0);
        //slab.dft_c2r(twodads::field_k_t::f_theta_hat, twodads::field_t::f_theta, my_config.get_tlevs() - t - 1);
        //slab.dft_c2r(twodads::field_k_t::f_strmf_hat, twodads::field_t::f_strmf, my_config.get_tlevs() - t);

        //slab.write_output(time);
        //slab.rhs_fun();
        //slab.move_t(twodads::field_k_t::f_theta_rhs_hat, my_config.get_tlevs() - t - 1, 0);
        //slab.move_t(twodads::field_k_t::f_omega_rhs_hat, my_config.get_tlevs() - t - 1, 0);
        //outfile.str("");
        //outfile << "theta_t" << setfill('0') << setw(5) << t << ".dat";
        //slab.print_field(twodads::f_theta_hat, outfile.str());
    }

    for(; t < num_tsteps + 1; t++)
    {
        //for(auto mode : mode_list)
        //{
        //    cout << "mode with kx = " << mode.kx << " ky = " << mode.ky << "\n";
        //    slab.integrate_stiff_debug(twodads::field_k_t::f_theta_hat, tlevs, mode.kx, mode.ky);
        //} 
        slab.integrate_stiff(twodads::field_k_t::f_theta_hat, tlevs);
        //slab.integrate_stiff(twodads::dyn_field_t::d_omega, tlevs);
        //slab.dft_c2r(twodads::field_k_t::f_theta_hat, twodads::field_t::f_theta, 0);
        //slab.dft_c2r(twodads::field_k_t::f_omega_hat, twodads::field_t::f_omega, 0);
        slab.advance();
        slab.update_real_fields(1);
        time += delta_t;

        if(t % tout_full == 0)
            slab_output.write_output(slab, time);

        if(t % tout_diag == 0)
        {
            slab_diag_cu.update_arrays(slab);
            slab_diag_cu.write_diagnostics(time, my_config);
        }
    }
    return(0);
}

