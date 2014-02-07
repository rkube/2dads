/*
 *  output.h
 *  2dads-oo
 *
 *  Created by Ralph Kube on 01.12.10.
 *  Copyright 2010 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __OUTPUT_CLASS
#define __OUTPUT_CLASS

#include "include/2dads_types.h"
#include "include/cuda_types.h"
//#include "array_base.h"
#include "include/cuda_array2.h"
#include <string>
#include <vector>
#include <H5Cpp.h>

#ifndef H5_NO_NAMESPACE
    using namespace H5;
#endif

using namespace std;

// Base class for output
// Virtual only

class output {
public:
    output(uint, uint);
    //~output() {};

    // Interface to write output in given output resource
    // write_output is purely virtual and will only be defined in the derived class
    virtual void surface(twodads::output_t, cuda_array<cuda::real_t>*, const cuda::real_t) = 0;

    // Output counter and array dimensions
    uint output_counter;
    const uint Nx, My;
};


// Derived class, handles output to HDF5
class output_h5 : public output {
public:
    /// Setup output for fields specified in configuration file of passed slab reference.
    output_h5(vector<twodads::output_t>, uint, uint);
    /// Cleanup.
    ~output_h5();
    
    /// Write variable type var_type from passed slab_array to output file.
    void surface(twodads::output_t, cuda_array<cuda::real_t>*, const cuda::real_t);

private:
    string filename;
    H5File* output_file;
    
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
    //vector<ofun_ptr> o_funs_vec;
    
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
    DataSpace* get_dspace_from_field(twodads::output_t);
    // Mapping from field types to dataspace names
    string get_fname_str_from_field(twodads::output_t);
};

#endif //__OUTPUT_H	
