/*
 * Read input.ini in directory argv[1], parse file, and print 
 * parameters to stdout
 *
 */


#include <iostream>
#include <string>
#include <fstream>
#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include "slab_config.h"

using namespace std;

int main(int argc, char* argv[])
{

    // Parse command line options to get base directory
    // See http://www.boost.org/doc/libs/1_57_0/libs/program_options/example/first.cpp
    po::variables_map vm;
    string simdir;
    try
    {
        po::options_description desc("Options");
        desc.add_options() ("simdir", po::value<string>(), "Base directory in which input.ini lies");

        po::store(po::parse_command_line(argc, argv, desc), vm);

        if(vm.count("simdir"))
        {
            simdir = vm["simdir"].as<string>();
            cout << "Simulation directory: " << simdir << endl;
        } else
        {
            simdir = ".";
        }

    }
    catch (exception& e)
    {
        cerr << "Error: " << e.what() << endl;
        return(1);
    }

    slab_config my_config(simdir);

    my_config.print_config();


    return(0);
}


