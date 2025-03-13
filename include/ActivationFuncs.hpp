#pragma once
#include <Tensors.hpp>

template<typename FloatType>
class ReLU{
public: 
  void operator()(Matrix<FloatType> &x, Matrix<FloatType> *deriv = nullptr) const;
};

template<typename FloatType>
class noActivation{
public:
  //f(x) = x
  inline void operator()(Matrix<FloatType> &x, Matrix<FloatType> *deriv = nullptr) const{
    if(deriv) *deriv = Matrix<FloatType>(x.size(0),x.size(1),1.0);
  }
};

#include "implementation/ActivationFuncs.tcc"

// #ifndef ACTIVATIONFUNC_EXTERN_TEMPLATE_INST
// #define SS extern
// #else
// #define SS
// #endif
// SS template class ReLU<float>;
// SS template class ReLU<double>;
// SS template class noActivation<float>;
// SS template class noActivation<double>;
// #undef SS

