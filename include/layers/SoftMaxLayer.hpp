#pragma once
#include "LayerCommon.hpp"
#include <components/SoftMaxComponent.hpp>

//A layer implementing the softmax operation on a batch tensor (one for which the last dimension is the batch dimension) along an arbitrary dimension other than the batch dimension
template<typename _FloatType, int TensDim, typename _InputType, typename Store >
class SoftMaxLayer{  
public:
  typedef _FloatType FloatType;
  typedef _InputType InputType;
  typedef LeafTag tag;
private:
  Store leaf;
  SoftMaxComponent<FloatType,TensDim> cpt;
public:
  
  inline SoftMaxLayer(Store &&leaf, int softmax_dim, FloatType beta = 1.0): leaf(std::move(leaf)), cpt(softmax_dim,beta){}
  inline SoftMaxLayer(SoftMaxLayer &&r) = default;
  inline SoftMaxLayer(const SoftMaxLayer &r) = delete;

  Tensor<FloatType,TensDim> value(const InputType &x);
  
  void deriv(Vector<FloatType> &cost_deriv, int off, Tensor<FloatType,TensDim> &&above_deriv, InputType* input_above_deriv_return = nullptr) const;
  
  inline void update(int off, const Vector<FloatType> &new_params){ leaf.v.update(off, new_params); }

  inline void step(int off, const Vector<FloatType> &derivs, FloatType eps){ leaf.v.step(off,derivs,eps); }
  
  inline int nparams() const{ return leaf.v.nparams(); }

  inline void getParams(Vector<FloatType> &into, int off){ leaf.v.getParams(into,off); }

  //For pipelining
  inline void resizeInputBuffer(size_t to){
    cpt.resizeInputBuffer(to);
    leaf.v.resizeInputBuffer(to);
  }
};

#define LAYER_TYPE SoftMaxLayer<FLOATTYPE(U),TensDim,INPUTTYPE(U),DDST(u)>
template<int TensDim, typename U, typename std::enable_if<ISLEAF(U), int>::type = 0>
auto softmax_layer(U &&u, int softmax_dim, FLOATTYPE(U) beta=FLOATTYPE(U)(1.0) )->LAYER_TYPE{
  return LAYER_TYPE(std::forward<U>(u), softmax_dim, beta);
}
#undef LAYER_TYPE

#include "implementation/SoftMaxLayer.tcc"

