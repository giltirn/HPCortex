#pragma once
#include "LayerCommon.hpp"
#include <Embeddings.hpp>

//A layer than embeds positions into the input using the sinusoidal method. Expects a tensor of size CxExB where C is the context size, E the embedding size and B the batch size
template<typename _FloatType, typename _InputType, typename Store>
class EmbedPositionsSinusoidalLayer{
public:
  typedef _FloatType FloatType;
  typedef _InputType InputType;
private:
  Store leaf;
  mutable FLOPScounter value_FLOPS;
public:
  typedef LeafTag tag;
  
  EmbedPositionsSinusoidalLayer(Store &&leaf):
    leaf(std::move(leaf))
  {  }
  EmbedPositionsSinusoidalLayer(const EmbedPositionsSinusoidalLayer &r) = delete;
  EmbedPositionsSinusoidalLayer(EmbedPositionsSinusoidalLayer &&r) = default;
  
  //Forward pass
  Tensor<FloatType,3> value(const InputType &x){
    auto out = embedPositionsSinusoidal(leaf.v.value(x), &value_FLOPS);
    value_FLOPS.lock();
    return out;
  }

  inline int deriv(Vector<FloatType> &cost_deriv, int off, Tensor<FloatType,3> &&_above_deriv, InputType* input_above_deriv_return = nullptr) const{
    //This is a constant additive embedding so it doesn't touch the derivatives
    return leaf.v.deriv(cost_deriv,off,std::move(_above_deriv),input_above_deriv_return);
  }
  inline int update(int off, const Vector<FloatType> &new_params){
    return leaf.v.update(off,new_params);
  }  
  inline int step(int off, const Vector<FloatType> &derivs, FloatType eps){
    return leaf.v.step(off,derivs, eps);
  }

  //accumulated #params for layers here and below
  inline int nparams() const{ return leaf.v.nparams(); }
  
  size_t FLOPS(int value_or_deriv) const{ return (value_or_deriv == 0 ? value_FLOPS.value() : 0) + leaf.v.FLOPS(value_or_deriv); }

  inline int getParams(Vector<FloatType> &into, int off){
    return leaf.v.getParams(into,off);
  }
  inline void resizeInputBuffer(size_t to){
    leaf.v.resizeInputBuffer(to);
  }

};

template<typename U, typename std::enable_if<ISLEAF(U), int>::type = 0>
auto embed_positions_sinusoidal_layer(U &&u){
  return EmbedPositionsSinusoidalLayer<FLOATTYPE(U),INPUTTYPE(U),DDST(u)>(std::forward<U>(u));
}
