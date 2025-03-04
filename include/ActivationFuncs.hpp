#pragma once
#include <Tensors.hpp>

//We implement activation functions as a numerical "mask" applied as an outer product to the output of the associated layer
//   for(int i=0;i<dim;i++)
//      for(int b=0;b<batch_size;b++)
//          out(i,b) *= mask(i,b)

template<typename FloatType>
class ReLU{
public: 
  Matrix<FloatType> operator()(const Matrix<FloatType> &x) const;
};

template<typename FloatType>
class noActivation{
public:
  inline Matrix<FloatType> operator()(const Matrix<FloatType> &x) const{
    return Matrix<FloatType>(x.size(0),x.size(1),1.0);
  }
};

#include "implementation/ActivationFuncs.tcc"

#ifndef ACTIVATIONFUNC_EXTERN_TEMPLATE_INST
#define SS extern
#else
#define SS
#endif
SS template class ReLU<float>;
SS template class ReLU<double>;
SS template class noActivation<float>;
SS template class noActivation<double>;
#undef SS

