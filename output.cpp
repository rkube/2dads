/*
 *  output.cpp
 *  2dads-oo
 *
 *  Created by Ralph Kube on 01.12.10.
 *  Copyright 2010 __MyCompanyName__. All rights reserved.
 *
 */

#include "include/output.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <cmath>


using namespace std;
using namespace H5;

// Constructor of the base class
output :: output(slab_config config) :
    output_counter(0),
    Nx(config.get_nx()),
    My(config.get_my())
{
}


//output_h5 :: output_h5(vector<twodads::output_t> o_list, uint _nx, uint _my) :
output_h5 :: output_h5(slab_config config) :
    output(config),
    filename("output.h5"),
    output_file(new H5File(H5std_string(filename.data()), H5F_ACC_TRUNC)),
	group_theta(new Group(output_file -> createGroup("/T"))),
    group_theta_x(new Group(output_file -> createGroup("/Tx"))),
    group_theta_y(new Group(output_file -> createGroup("/Ty"))),
	group_omega(new Group(output_file -> createGroup("/O"))),
    group_omega_x(new Group(output_file -> createGroup("/Ox"))),
    group_omega_y(new Group(output_file -> createGroup("/Oy"))),
    group_strmf(new Group(output_file -> createGroup("/S"))),
    group_strmf_x(new Group(output_file -> createGroup("/Sx"))),
    group_strmf_y(new Group(output_file -> createGroup("/Sy"))),
    group_omega_rhs(new Group(output_file -> createGroup("/ORHS"))),
    group_theta_rhs(new Group(output_file -> createGroup("/TRHS"))),
	dspace_file(NULL),
    o_list(config.get_output())
{
	// DataSpace dimension 
	const hsize_t fdim[] = {Nx, My};
	// Hyperslab parameter for ghost point array output
	const hsize_t offset[] = {0,0};
	const hsize_t count[] = {Nx, My};
	DSetCreatPropList ds_creatplist;  
	dspace_file = new DataSpace(2, count); 

	// Iterate over defined output functions and add them to container
	#ifdef DEBUG
		cout << "Initializing HDF5 output\n";
		cout << "Output file " << output_file -> getFileName() << " created. Id: " << output_file -> getId() << "\n";
	#endif //DEBUG

    DataSpace* dspace;
    for(auto it: o_list)
    {
        dspace = get_dspace_from_field(it);
        (*dspace) = DataSpace(2, fdim);
        (*dspace).selectHyperslab(H5S_SELECT_SET, count, offset, NULL, NULL);
    }

    output_file -> flush(H5F_SCOPE_LOCAL);
    output_file -> close();
    delete output_file; 
}



output_h5 :: ~output_h5()
{
	delete group_theta;
	delete group_theta_x;
	delete group_theta_y;
	delete group_omega;
    delete group_omega_x;
    delete group_omega_y;
	delete group_strmf;
	delete group_strmf_x;
	delete group_strmf_y;
    delete group_theta_rhs;
    delete group_omega_rhs;
}


/// @brief Call this routine to write output
/// @detailed Note that get_field_by_name, when called with a twodads::output_t
/// @detailed also calls cuda_array<%>.copy_device_to_host
void output_h5 :: write_output(slab_cuda& slab, twodads::real_t time)
{
    cuda_array<cuda::real_t, cuda::real_t>* arr;
    // Iterate over list of fields we need to write output for
    for(auto it : o_list)
    {
        arr = slab.get_field_by_name(it);
        surface(it, arr, time);
    }
    output_counter++;
}



/// @brief Write output field
/// @detailed This assumes that the host data src points to is up-to-date. 
void output_h5 :: surface(twodads::output_t field_name, cuda_array<cuda::real_t, cuda::real_t>* src, const cuda::real_t time)
{
    // Dataset name is /[OST]/[0-9]*
    //string dataset_name = get_fname_str_from_field(field_name) + "/" + to_string(output_counter); 
    stringstream foo;
    foo << get_fname_str_from_field(field_name) << "/" << to_string(output_counter);
    string dataset_name(foo.str());
    output_file = new H5File(filename, H5F_ACC_RDWR);
	DataSpace* dspace_ptr = get_dspace_from_field(field_name);

    cout << "Dataset name: " << dataset_name << "\n";
        FloatType float_type(PredType::NATIVE_DOUBLE);
    DataSpace att_space(H5S_SCALAR);
	
	// Create dataset and write data
	DataSet* dataset = new DataSet( 
		output_file->createDataSet(dataset_name, PredType::NATIVE_DOUBLE, *dspace_file, ds_creatplist) );

    // Create time attribute for the Dataset
    Attribute att = dataset -> createAttribute("time", float_type, att_space);
    att.write(float_type, &time);
	dataset -> write( src -> get_array_h(), PredType::NATIVE_DOUBLE, *dspace_ptr );
	delete dataset;
    delete output_file;
}	

DataSpace* output_h5 :: get_dspace_from_field(twodads::output_t field_name)
{
    switch(field_name)
    {
        case twodads::output_t::o_theta:
            return &dspace_theta;
            break;
        case twodads::output_t::o_theta_x:
            return &dspace_theta_x;
            break;
        case twodads::output_t::o_theta_y:
            return &dspace_theta_y;
            break;
        case twodads::output_t::o_omega:
            return &dspace_omega;
            break;
        case twodads::output_t::o_omega_x:
            return &dspace_omega_x;
            break;
        case twodads::output_t::o_omega_y:
            return &dspace_omega_y;
            break;
        case twodads::output_t::o_strmf:
            return &dspace_strmf;
            break;
        case twodads::output_t::o_strmf_x:
            return &dspace_strmf_x;
            break;
        case twodads::output_t::o_strmf_y:
            return &dspace_strmf_y;
            break;
        case twodads::output_t::o_omega_rhs:
            return &dspace_omega_rhs;
            break;
        case twodads::output_t::o_theta_rhs:
            return &dspace_theta_rhs;
            break;
    }
    // This should never happen
    string err_msg("get_space_from_field: could not find field for twodads::output_t field_name");
    throw name_error(err_msg);
}


// Return a string containing the field name for the dataset field_name
string output_h5 :: get_fname_str_from_field(twodads::output_t field_name)
{
    switch (field_name)
    {
        case twodads::output_t::o_omega:
            return string("O/");
            break;
            
        case twodads::output_t::o_omega_x:
            return string("Ox/");
            break;

        case twodads::output_t::o_omega_y:
            return string("Oy/");
            break;
            
        case twodads::output_t::o_theta:
            return string("T/");
            break;

        case twodads::output_t::o_strmf:
            return string("S/");
            break;
            
        case twodads::output_t::o_strmf_x:
            return string("Sx/");
            break;
            
        case twodads::output_t::o_strmf_y:
            return string("Sy/");
            break;
            
        case twodads::output_t::o_theta_x:
            return string("Tx/");
            break;
            
        case twodads::output_t::o_theta_y:
            return string("Ty/");
            break;

        case twodads::output_t::o_theta_rhs:
            return string("TRHS/");
            break;

        case twodads::output_t::o_omega_rhs:
            return string("ORHS/");
            break;
    }
    // This should never happen
    string err_msg("get_fname_str_from_field: could not find field for twodads::output_t field_name");
    throw name_error(err_msg);
}

// End of file output.cpp
