// Test access to ky=0 modes
//

#include <iostream>
#include <include/slab_cuda.h>

using namespace std;

int main(void)
{
    slab_config my_config;
    slab_cuda slab(my_config);

    slab.print_grids();
}
