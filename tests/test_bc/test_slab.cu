/*
 * Test methods requiring two arrays with a dummy slab.
 *
 *
 */

#include <iostream>
#include <slab_test.h>


int main(void)
{
    constexpr int Nx{16};
    constexpr int My{16};
    constexpr double y_lo{0.0};
    constexpr double y_up{1.0};
    constexpr double x_l{0.0};
    constexpr double x_r{1.0};
    
    cuda::bvals<double> my_bvals{cuda::bc_t::bc_neumann, cuda::bc_t::bc_neumann, cuda::bc_t::bc_periodic, cuda::bc_t::bc_periodic, 0.0, 0.0, 0.0, 0.0};
    slab_layout_t my_slab_layout{0.0, (x_r - x_l) / double(Nx), 0.0, (y_up - y_lo) / double(My), 0.001, My, Nx};
    
    slab_bc<double> my_slab(my_slab_layout, my_bvals);
    
    my_slab.print_field(field_t::f1);
    my_slab.test_d_dx(0);
    my_slab.print_field(field_t::f2);


    return(0);
}
