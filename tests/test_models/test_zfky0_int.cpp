// Test integration of ky=0 modes with different diffusion coefficient

#include <iostream>
#include <include/slab_cuda.h>

using namespace std;

int main(void)
{
    slab_config my_config;
    slab_cuda slab(my_config);

    slab.print_grids();

    slab.init_dft();

    cout << "Initializing\n";
    slab.initialize();

    cout << "t=0\n";
    slab.print_field(twodads::field_k_t::f_omega_hat);
    slab.integrate_stiff_ky0(twodads::field_k_t::f_omega_hat, 2);
    cout << "t=1\n";
    slab.print_field(twodads::field_k_t::f_omega_hat);

}

