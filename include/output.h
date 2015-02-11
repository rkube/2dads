/*
 *  output.h
 *  2dads-oo
 *
 *  Created by Ralph Kube on 01.12.10.
 *  Copyright 2010 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef OUTPUT_H
#define OUTPUT_H

#include <vector>
#include <map>
#include <H5Cpp.h>
#include "2dads_types.h"
#include "cuda_types.h"
#include "slab_cuda.h"
#include "slab_config.h"

#ifndef H5_NO_NAMESPACE
    using namespace H5;
#endif

using namespace std;

// Base class for output
// Virtual only

class output {
public:
    output(slab_config);
    //~output() {};

    // Interface to write output in given output resource
    // write_output is purely virtual and will only be defined in the derived class
    //virtual void surface(twodads::output_t, cuda_array<cuda::real_t, cuda::real_t>*, const cuda::real_t) = 0;
    virtual void write_output(slab_cuda&, twodads::real_t) = 0;

    //void update_array(slab_cuda&);
    // Output counter and array dimensions
    inline uint get_output_counter() const {return(output_counter);};
    inline void increment_output_counter() {output_counter++;};
    inline uint get_nx() const {return(Nx);};
    inline uint get_my() const {return(My);};
private:
    uint output_counter;
    const uint My, Nx;
};


// Derived class, handles output to HDF5
class output_h5 : public output {
public:
    /// Setup output for fields specified in configuration file of passed slab reference.
    //output_h5(vector<twodads::output_t>, uint, uint);
    output_h5(slab_config);
    /// Cleanup.
    ~output_h5();
    
    /// @brief Call this routine to write output
    /// @detailed Note that get_field_by_name, when called with a twodads::output_t
    /// @detailed also calls cuda_array<%>.copy_device_to_host
    void write_output(slab_cuda&, twodads::real_t);
    /// @brief Write output field
    /// @detailed This assumes that the host data src points to is up-to-date. 
    void surface(twodads::output_t, cuda_array<cuda::real_t>*, const cuda::real_t);

private:
    string filename;
    H5File* output_file;
    //const size_t fdim[2];
    
    Group* group_theta;
    Group* group_theta_x;
    Group* group_theta_y;
    Group* group_omega;
    Group* group_omega_x;
    Group* group_omega_y;
    Group* group_strmf;
    Group* group_strmf_x;
    Group* group_strmf_y;
    Group* group_omega_rhs;
    Group* group_theta_rhs;

    DataSpace* dspace_file;
    // Vector of output functions
    vector<twodads::output_t> o_list;
    
    DataSpace dspace_theta;
    DataSpace dspace_theta_x;
    DataSpace dspace_theta_y;
    DataSpace dspace_omega;
    DataSpace dspace_omega_x;
    DataSpace dspace_omega_y;
    DataSpace dspace_strmf;
    DataSpace dspace_strmf_x;
    DataSpace dspace_strmf_y;
    DataSpace dspace_omega_rhs;
    DataSpace dspace_theta_rhs;
    DSetCreatPropList ds_creatplist;
    // Mapping from field types to dataspace
    map<twodads::output_t, DataSpace*> dspace_map;
    // Mapping from field types to dataspace names
    map<twodads::output_t, string> fname_map;
};

#endif //OUTPUT_H
