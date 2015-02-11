/*
 * test_constructors.cu
 *
 *  Created on: Feb 5, 2015
 *      Author: rku000
 */

#include <iostream>
#include <cuda_array4.h>


using namespace std;

int main(void)
{
	unsigned int Nx{16};
	unsigned int My{16};
	cuda_array<double> arr1(Nx, My);
	arr1 = 1.0;
	cout << "arr1: \n" << arr1 << endl;
	cuda_array<double>arr2(arr1);
	cout << "arr2: \n" << arr2 << endl;



	cuda_array<double>arr3(std::move(arr2));
	cout << "arr3: \n" << arr3 << endl;

}
