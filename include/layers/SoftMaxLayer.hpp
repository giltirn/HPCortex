#pragma once
#include "LayerCommon.hpp"
#include <components/SoftMaxComponent.hpp>

//A layer implementing the softmax operation on a batch tensor (one for which the last dimension is the batch dimension) along an arbitrary dimension other than the batch dimension
template<typename Config, int TensDim, typename _InputType, typename Store >
class SoftMaxLayer{  
public:
  EXTRACT_CONFIG_TYPES;
  typedef _InputType InputType;
  typedef LeafTag tag;
private:
  Store leaf;
  SoftMaxComponent<Config,TensDim> cpt;
public:
  
  inline SoftMaxLayer(Store &&leaf, int softmax_dim, FloatType beta = 1.0): leaf(std::move(leaf)), cpt(softmax_dim,beta){}
  inline SoftMaxLayer(SoftMaxLayer &&r) = default;
  inline SoftMaxLayer(const SoftMaxLayer &r) = delete;

  Tensor<FloatType,TensDim> value(const InputType &x);
  
  int deriv(Vector<FloatType> &cost_deriv, int off, Tensor<FloatType,TensDim> &&above_deriv, InputType* input_above_deriv_return = nullptr) const;
  
  inline int update(int off, const Vector<FloatType> &new_params){ return leaf.v.update(off, new_params); }

  inline int step(int off, const Vector<FloatType> &derivs, FloatType eps){ return leaf.v.step(off,derivs,eps); }
  
  inline int nparams() const{ return leaf.v.nparams(); }

  inline size_t FLOPS(int value_or_deriv) const{ return cpt.FLOPS(value_or_deriv) + leaf.v.FLOPS(value_or_deriv); }
  
  inline int getParams(Vector<FloatType> &into, int off) const{ return leaf.v.getParams(into,off); }

  //For pipelining
  inline void resizeInputBuffer(size_t to){
    cpt.resizeInputBuffer(to);
    leaf.v.resizeInputBuffer(to);
  }

  //Set the inverse-temperature, beta
  inline void setBeta(FloatType beta){ cpt.setBeta(beta); }
};

#define LAYER_TYPE SoftMaxLayer<CONFIGTYPE(U),TensDim,INPUTTYPE(U),DDST(u)>
template<int TensDim, typename U, typename std::enable_if<ISLEAF(U), int>::type = 0>
auto softmax_layer(int softmax_dim, FLOATTYPE(U) beta, U &&u)->LAYER_TYPE{
  return LAYER_TYPE(std::forward<U>(u), softmax_dim, beta);
}

template<int TensDim, typename U, typename std::enable_if<ISLEAF(U), int>::type = 0>
auto softmax_layer(int softmax_dim, U &&u)->LAYER_TYPE{
  return LAYER_TYPE(std::forward<U>(u), softmax_dim, FLOATTYPE(U)(1.0) );
}


#undef LAYER_TYPE

#include "implementation/SoftMaxLayer.tcc"

