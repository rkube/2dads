/*
 * test_array.cpp
 *
 *  Created on: Oct 22, 2013
 *      Author: rku000
 */

#include <iostream>
#include "include/cuda_array2.h"

using namespace std;


int main(void)
{
    const int Nx = 16;
    const int My = 16;
    const int tlevs = 3;
	cuda_array<double> arr1(tlevs, Nx, My);
	//cuda_array<double> arr2(1, 16, 16);

	cout << "arr1: Nx = " << arr1.get_nx() << ", My = " << arr1.get_my() << ", tlevs = " << arr1.get_tlevs() << "\n";
	arr1.set_all(59.4);
	arr1.copy_device_to_host();
	cout << "arr1=\n" << arr1 << "\n";

	arr1.enumerate_array_t(0);
	arr1.copy_device_to_host();
	cout << "arr1=\n" << arr1 << "\n";

	arr1.advance();

	arr1.enumerate_array_t(0);
	arr1.copy_device_to_host();
	cout << "arr1 =\n" << arr1 << "\n";
	//arr1 = 16.0;
	//arr2 = 32.0;
}


