#pragma once
#include <Tensors.hpp>

template<typename FloatType>
class ReLU{
public:
  //f(x)_i = max(x_i, 0) = x_i  |  x_i > 0
  //                     = 0    |  x_i <= 0
  //if deriv != nullptr, it is populated with df(x)_i / dx_i
  void operator()(Matrix<FloatType> &x, Matrix<FloatType> *deriv = nullptr) const;
  template<int Dim>
  void operator()(Tensor<FloatType,Dim> &x, Tensor<FloatType,Dim> *deriv = nullptr) const;
};

template<typename FloatType>
class noActivation{
public:
  //f_i(x) = x_i
  inline void operator()(Matrix<FloatType> &x, Matrix<FloatType> *deriv = nullptr) const{
    if(deriv) *deriv = Matrix<FloatType>(x.size(0),x.size(1),1.0);
  }
  template<int Dim>
  inline void operator()(Tensor<FloatType,Dim> &x, Tensor<FloatType,Dim> *deriv = nullptr) const{
    if(deriv) *deriv = Tensor<FloatType,Dim>(x.sizeArray(),1.0);
  }
  
};

template<typename FloatType>
class GeLU{
public:
  //f(x)_i = x_i ( 1 + erf(x_i/sqrt2) )/2
  template<int Dim>
  void operator()(Tensor<FloatType,Dim> &x, Tensor<FloatType,Dim> *deriv = nullptr) const;
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

