/*
 * Test derivative kernels
 *
 */


#include <iostream>
#include <include/slab_cuda.h>

using namespace std;

int main(void)
{
    slab_config my_config;
    my_config.consistency();

    slab_cuda slab(my_config);
    slab.initialize();

    cout << "Enumerating kernels for d/dx\n";
    slab.d_dx_debug(twodads::field_k_t::f_theta_hat, twodads::field_k_t::f_theta_x_hat, 3);
    slab.dump_field(twodads::field_k_t::f_theta_x_hat);

    cout << "Enumerating kernels for d/dy\n";
    slab.d_dy_debug(twodads::field_k_t::f_theta_hat, twodads::field_k_t::f_theta_y_hat, 3);
    slab.dump_field(twodads::field_k_t::f_theta_y_hat);

    cout << "Enumerating kernels for inv_laplace\n";
    slab.inv_laplace_debug(twodads::field_k_t::f_theta_hat, twodads::field_k_t::f_tmp_hat, 3);
    slab.dump_field(twodads::field_k_t::f_tmp_hat);

}
