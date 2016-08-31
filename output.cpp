/*
 *  output.cpp
 *  2dads
 *
 *  Created by Ralph Kube on 01.12.10.
 *
 */
#include "output.h"


using namespace std;
using namespace H5;

// Constructor of the base class
output_t :: output_t(slab_config_js config) :
    output_counter(0),
    dtout(config.get_tout()),
    geom(config.get_geom())
{
}


const std::map<twodads::output_t, std::string> output_h5_t :: fname_map
{
        {twodads::output_t::o_theta, "/N/"},
        {twodads::output_t::o_theta_x, "/Nx/"},
        {twodads::output_t::o_theta_y, "/Ny/"},
        {twodads::output_t::o_tau, "/T/"},
        {twodads::output_t::o_tau_x, "/Tx/"},
        {twodads::output_t::o_tau_y, "/Ty/"},
        {twodads::output_t::o_omega, "/O/"},
        {twodads::output_t::o_omega_x, "/Ox/"},
        {twodads::output_t::o_omega_y, "/Oy/"},
        {twodads::output_t::o_strmf, "/S/"},
        {twodads::output_t::o_strmf_x, "/Sx/"},
        {twodads::output_t::o_strmf_y, "/Sy/"}
};


// Initialize DataSpace objects in a later for-loop But populate dspace_map with pointers.
output_h5_t :: output_h5_t(const slab_config_js& config) :
    output_t(config),
    filename("output.h5"),
    output_file{new H5File(H5std_string(filename.data()), H5F_ACC_TRUNC)},
	group_theta{new Group(output_file ->   createGroup("/N"))},
    group_theta_x{new Group(output_file -> createGroup("/Nx"))},
    group_theta_y{new Group(output_file -> createGroup("/Ny"))},
	group_tau{new Group(output_file ->     createGroup("/T"))},
    group_tau_x{new Group(output_file ->   createGroup("/Tx"))},
    group_tau_y{new Group(output_file ->   createGroup("/Ty"))},
	group_omega{new Group(output_file ->   createGroup("/O"))},
    group_omega_x{new Group(output_file -> createGroup("/Ox"))},
    group_omega_y{new Group(output_file -> createGroup("/Oy"))},
    group_strmf{new Group(output_file ->   createGroup("/S"))},
    group_strmf_x{new Group(output_file -> createGroup("/Sx"))},
    group_strmf_y{new Group(output_file -> createGroup("/Sy"))},
	dspace_file{nullptr}
{
    // ds_memsize is the dimension of the memory allocaed by cuda_array_bc_npgp.
    // This includes the padding
    const hsize_t ds_memsize[] = {get_geom().get_nx() + get_geom().get_pad_x(), get_geom().get_my() + get_geom().get_pad_y()};
	const hsize_t offset[] = {0,0};
    // count is the number of elements we are selecting in the dataspaces
    // associated with the output fields. This ignores all padding.
    const hsize_t ds_fsize[] = {get_geom().get_nx(), get_geom().get_my()};

    DSetCreatPropList ds_creatplist;  
    dspace_file = new DataSpace(2, ds_fsize); 

    dspace_map[twodads::output_t::o_theta] = &dspace_theta;
    dspace_map[twodads::output_t::o_theta_x] = &dspace_theta_x;
    dspace_map[twodads::output_t::o_theta_y] = &dspace_theta_y;
    dspace_map[twodads::output_t::o_tau] = &dspace_tau;
    dspace_map[twodads::output_t::o_tau_x] = &dspace_tau_x;
    dspace_map[twodads::output_t::o_tau_y] = &dspace_tau_y;
    dspace_map[twodads::output_t::o_omega] = &dspace_omega;
    dspace_map[twodads::output_t::o_omega_x] = &dspace_omega_x;
    dspace_map[twodads::output_t::o_omega_y] = &dspace_omega_y;
    dspace_map[twodads::output_t::o_strmf] = &dspace_strmf;
    dspace_map[twodads::output_t::o_strmf_x] = &dspace_strmf_x;
    dspace_map[twodads::output_t::o_strmf_y] = &dspace_strmf_y;

    DataSpace* dspace;

#ifdef DEBUG
		std::cout << "Initializing HDF5 output\n";
		std::cout << "Output file " << output_file -> getFileName() << " created. Id: " << output_file -> getId() << "\n";
        std::cout << "ds_memsize = [" << ds_memsize[0] << ", " << ds_memsize[1] << "]" << std::endl;
        std::cout << "ds_fsize = [" << ds_fsize[0] << ", " << ds_fsize[1] << "]" << std::endl;
#endif //DEBUG

    output_file -> flush(H5F_SCOPE_LOCAL);
    output_file -> close();

    delete output_file;


    for(auto it : config.get_output())
    {
        dspace = dspace_map.at(it);
        (*dspace) = DataSpace(2, ds_memsize);
        // https://www.hdfgroup.org/HDF5/doc/cpplus_RM/class_h5_1_1_data_space.html#a92bd510d1c06ebef292faeff73f40c12
        // Define memory dataspace associated with array.
        // All arrays use the same padding (f.ex. pad_y = 2 for in-place FFTs) in the
        // last dimension. See cuda_array_bc_nogp.h
        
        (*dspace).selectHyperslab(H5S_SELECT_SET, ds_fsize, offset, NULL, NULL);
    }
}


output_h5_t :: ~output_h5_t()
{
    delete dspace_file;
	delete group_theta;
	delete group_theta_x;
	delete group_theta_y;
	delete group_tau;
	delete group_tau_x;
	delete group_tau_y;
	delete group_omega;
    delete group_omega_x;
    delete group_omega_y;
	delete group_strmf;
	delete group_strmf_x;
	delete group_strmf_y;

}


void output_h5_t :: surface(twodads::output_t field_name, 
                            const cuda_array_bc_nogp<twodads::real_t, allocator_host>& src,
                            const size_t tidx)
{
    // Dataset name is /[NOST]/[0-9]*
    const twodads::real_t time{twodads::real_t(get_output_counter()) * get_dtout()};

    stringstream foo;
    
    foo << fname_map.at(field_name) << "/" << to_string(get_output_counter());
    string dataset_name(foo.str());

    output_file = new H5File(filename, H5F_ACC_RDWR);
    DataSpace* dspace_ptr = dspace_map[field_name];

    FloatType float_type(PredType::NATIVE_DOUBLE);
    DataSpace att_space(H5S_SCALAR);
	
	// Create dataset in the file 
	DataSet* dataset = new DataSet(output_file->createDataSet(dataset_name, 
                                                              PredType::NATIVE_DOUBLE, 
                                                              *dspace_file, 
                                                              ds_creatplist));
    // Write to the data set we just created in the file.
    // Source pointed to by dspace_ptr, created in the constructor

/*
    const size_t nelem{src -> get_geom().get_nx() * (src -> get_geom().get_my() + src -> get_geom().get_pad_y())};
    twodads::real_t dummy[nelem];
    for(size_t n = 0; n < nelem; n++)
    {
        dummy[n] = twodads::real_t(n);
    }
*/
    std::cout << "writing data at tidx" << tidx << std::endl;
	dataset -> write(src.get_tlev_ptr(tidx), PredType::NATIVE_DOUBLE, *dspace_ptr);
    //dataset -> write(dummy, PredType::NATIVE_DOUBLE, *dspace_ptr);
    
    // Create time attribute for the Dataset
    Attribute att = dataset -> createAttribute("time", float_type, att_space);
    att.write(float_type, &time);

	delete dataset;
    delete output_file;
}	

// End of file output.cpp