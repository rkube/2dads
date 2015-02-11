///@detailed
/// operators to be used in cuda kernels


#ifndef CUDA_OPERATORS_H
#define CUDA_OPERATORS_H

#ifdef CUDACC
// Do not use __host__, this would break templating of operator calls!
// ->cudacc uses host instantiation of template parameter in cuda_array4.h
#define CUDAMEMBER __device__
#endif

#ifndef CUDACC
#define CUDAMEMBER
#endif

/// Device functors for arithmetic operations


/*****************************************************************************
 *
 * Expression  operators
 *
 *****************************************************************************/


template <typename T>
class d_op0_expassign{
	public:
		__device__ void operator() (T& a) {a = exp(a);};
};

template <typename T>
class d_op0_logassign{
	public:
		__device__ void operator() (T& a) {a = log(a);};
};

/*****************************************************************************
 *
 * Unary operators
 *
 *****************************************************************************/


template <typename T>
class d_op1_assign{
	public:
//		d_op1_assign() {};
//		d_op1_assign(T val_) val(val_) {};
		__device__ void operator() (T& a, const T& b) {a = b;};
//		__device__ void operator() (T& a) {a = val;};
	private:
//		const T val;
};


template <typename T>
class d_op1_addassign{
	public:
		__device__ void operator() (T& a, const T& b) {a += b;};
};


template <typename T>
class d_op1_subassign{
	public:
		__device__ void operator() (T& a, const T& b) {a -= b;};
};


template <typename T>
class d_op1_mulassign{
	public:
		__device__ void operator() (T& a, const T& b) {a *= b;};
};


template <typename T>
class d_op1_divassign{
	public:
		__device__ void operator() (T& a, const T& b) {a /= b;};
};

template <typename T>
class d_op1_maxassign{
	public:
		__device__ void operator() (T& a, const T& b) {a = max(a, b);};
};


template <typename T>
class d_op1_minassign{
	public:
		__device__ void operator() (T& a, const T& b) {a = min(a, b);}
};

/*****************************************************************************
 *
 * Binary operators
 *
 *****************************************************************************/
template <typename T>
class d_op2_add{
    public:
        __device__ T operator()(const T& a,const T& b) const {return(a + b);};
        __device__ T operator()(const T& a, const T& b, const T& scale) const {return((a + b) * scale);};
};


template <typename T>
class d_op2_addexp{
	public:
		__device__ T operator()(const T& a, const T& b) const {return(exp(a) + exp(b));};
};


template <typename T>
class d_op2_sub{
    public:
        __device__ T operator()(const T& a, const T& b)  {return(a - b);};
};


template <typename T>
class d_op2_mul{
    public:
        __device__ T operator()(const T& a, const T& b) const {return(a * b);};
};


template <typename T>
class d_op2_div{
    public:
        __device__ T operator()(const T& a, const T& b) const {return(a / b);};
};


template <typename T>
class d_op2_max{
    public:
        __device__ T operator()(const T& a, const T& b) const {return(max(a, b));};
};


template <typename T>
class d_op2_min{
    public:
        __device__ T operator()(const T& a, const T& b) const {return(min(a, b));};
};



#endif //CUDA_OPERATORS_H

