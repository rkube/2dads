///*
//
// * test_operators.cu
// *
// *  Created on: Feb 6, 2015
// *      Author: rku000
// *
// * Test how to correctly template with cuda_operators
// *
// */
//
#include <iostream>
#include "cuda_operators.h"

using namespace std;


template <typename T>
class arr
{
public:
	arr(int n_) : N(n_), d_ptr(new T[N]) {};

	T* get_d_ptr() const {return d_ptr;};
	T get_d_ptr(int n) const {return d_ptr[n % N];};

	void op_all(const T& rhs);
	template <typename O>
	void op_all2(const T& rhs)
	{
		O op;
		for(int n = 0; n < N; n++)
			d_ptr[n] = op(d_ptr[n], rhs);
	}

	void print() const{
		for(int n = 0; n < N; n++)
			cout << d_ptr[n] << "\t";
		cout << endl;
	}
private:
	const unsigned int N;
	T* d_ptr;
};


template <typename T>
void arr<T> :: op_all(const T& rhs)
{
	for(int n = 0; n < N; n++)
		d_op1_assign<T>()(d_ptr[n], rhs);
}



int main(void)
{
	constexpr unsigned int N{8};
	arr<double> a(N);

	// Print initial
	//a.print();

	//a.op_all(1.0);
	//a.print();
	a.op_all2<d_op2_add<double> >(2.0);
	a.print();
}



