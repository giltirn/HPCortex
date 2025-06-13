#pragma once
#include "LayerCommon.hpp"

//The input layer
//This is always the lowest layer in the model
template<typename _FloatType, typename _InputType = Matrix<_FloatType> >
class InputLayer{  
public:
  typedef _FloatType FloatType;
  typedef _InputType InputType;
  typedef LeafTag tag;
  
  inline InputLayer(){}
  inline InputLayer(InputLayer &&r) = default;
  inline InputLayer(const InputLayer &r) = delete;

  inline const InputType &value(const InputType &x){
    //Simply reflect the passed-down input value back up to commence forwards propagation
    return x;
  }

  //input_above_deriv_return is the derivative of the cost with respect to the inputs
  inline int deriv(Vector<FloatType> &cost_deriv, int off, InputType &&above_deriv, InputType* input_above_deriv_return = nullptr) const{
    //We don't have to do anything for backpropagation as this is the last layer
    if(input_above_deriv_return) *input_above_deriv_return = std::move(above_deriv); //copy back the input derivative if desired
    return off;
  }
  
  inline int update(int off, const Vector<FloatType> &new_params){ return off; }

  inline int step(int off, const Vector<FloatType> &derivs, FloatType eps){ return off; }
  
  inline int nparams() const{ return 0; }

  inline size_t FLOPS(int value_or_deriv) const{ return 0; }
  
  inline int getParams(Vector<FloatType> &into, int off) const{ return off; }

  //For pipelining
  inline void resizeInputBuffer(size_t to){}
};

template<typename FloatType, typename InputType = Matrix<FloatType> >
inline InputLayer<FloatType,InputType> input_layer(){ return InputLayer<FloatType,InputType>(); }
