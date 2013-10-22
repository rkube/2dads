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
	cuda_array<double> arr1(16, 16, 1);
	cuda_array<double> arr2(16, 16, 1);

	arr1 = 16.0;
	arr2 = 32.0;

}


