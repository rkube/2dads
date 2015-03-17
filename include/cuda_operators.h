///@detailed
/// operators to be used in cuda kernels


#ifndef CUDA_OPERATORS_H
#define CUDA_OPERATORS_H

#ifdef __CUDA_ARCH__
// Do not use __host__, this would break templating of operator calls!
// ->cudacc uses host instantiation of template parameter in cuda_array4.h
#define CUDAMEMBER __device__
#endif

#ifndef __CUDA_ARCH__
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
		CUDAMEMBER inline void operator() (T& a) {a = exp(a);};
};

template <typename T>
class d_op0_logassign{
	public:
		CUDAMEMBER inline void operator() (T& a) {a = log(a);};
};


template <typename T>
class d_op0_squareassign{
    public:
        CUDAMEMBER inline void operator() (T& a) {a = a * a;};
};


template <typename T>
class d_op0_sqrtassign{
    public:
        CUDAMEMBER inline void operator() (T& a) {a = sqrt(a);};
};


template <typename T>
class d_op0_absassign{
    public:
        CUDAMEMBER inline void operator() (T& a) {a = fabs(a);};
};

/*****************************************************************************
 *
 * Unary operators
 *
 *****************************************************************************/


template <typename T>
class d_op1_assign{
	public:
		CUDAMEMBER inline void operator() (T& a, const T& b) {a = b;};
};


template <typename T>
class d_op1_addassign{
	public:
		CUDAMEMBER inline void operator() (T& a, const T& b) {a += b;};
};


template <typename T>
class d_op1_subassign{
	public:
		CUDAMEMBER inline void operator() (T& a, const T& b) {a -= b;};
};


template <typename T>
class d_op1_mulassign{
	public:
		CUDAMEMBER inline void operator() (T& a, const T& b) {a *= b;};
};


template <typename T>
class d_op1_divassign{
	public:
		CUDAMEMBER inline void operator() (T& a, const T& b) {a /= b;};
};

template <typename T>
class d_op1_maxassign{
	public:
		CUDAMEMBER inline void operator() (T& a, const T& b) {a = max(a, b);};
};


template <typename T>
class d_op1_absmaxassign{
    public:
        CUDAMEMBER inline void operator() (T& a, const T& b) {a = max(fabs(a), fabs(b));};
};


template <typename T>
class d_op1_minassign{
	public:
		CUDAMEMBER inline void operator() (T& a, const T& b) {a = min(a, b);}
};

template <typename T>
class d_op1_absminassign{
	public:
		CUDAMEMBER inline void operator() (T& a, const T& b) {a = min(fabs(a), fabs(b));}
};

/*****************************************************************************
 *
 * Binary operators
 *
 *****************************************************************************/
template <typename T>
class d_op2_add{
    public:
        CUDAMEMBER inline T operator()(const T& a,const T& b) const {return(a + b);};
        CUDAMEMBER inline T operator()(const T& a, const T& b, const T& scale) const {return((a + b) * scale);};
};


template <typename T>
class d_op2_addexp{
	public:
		CUDAMEMBER inline T operator()(const T& a, const T& b) const {return(exp(a) + exp(b));};
};


template <typename T>
class d_op2_sub{
    public:
        CUDAMEMBER inline T operator()(const T& a, const T& b)  {return(a - b);};
};


template <typename T>
class d_op2_mul{
    public:
        CUDAMEMBER inline T operator()(const T& a, const T& b) const {return(a * b);};
};


template <typename T>
class d_op2_div{
    public:
        CUDAMEMBER inline T operator()(const T& a, const T& b) const {return(a / b);};
};


template <typename T>
class d_op2_max{
    public:
        CUDAMEMBER inline T operator()(const T& a, const T& b) const {return(max(a, b));};
};


template <typename T>
class d_op2_min{
    public:
        CUDAMEMBER inline T operator()(const T& a, const T& b) const {return(min(a, b));};
};



#endif //CUDA_OPERATORS_H

