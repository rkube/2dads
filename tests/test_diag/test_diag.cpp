#include <iostream>
#include "cuda_runtime_api.h"
#include "slab_cuda.h"
#include "diagnostics.h"
#include "diagnostics_cu.h"

extern template class diag_array<double>;

using namespace std;

int main(void)
{
    slab_config my_config;
    my_config.consistency();

    slab_cuda slab(my_config);
    diagnostics slab_diag(my_config);
    diagnostics_cu slab_diag_cu(my_config);

    twodads::real_t time(0.0);

    slab.init_dft();
    slab.initialize();

    slab_diag.update_arrays(slab);
    slab_diag.write_diagnostics(time, my_config);

    slab_diag_cu.update_arrays(slab);
    slab_diag_cu.write_diagnostics(time, my_config);

}
