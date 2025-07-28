#pragma once
#include "LayerCommon.hpp"

template<typename Config, typename _InputType, typename ChainInternal, typename ChainBelow>
class SkipConnection{
public:
  EXTRACT_CONFIG_TYPES;
  typedef _InputType InputType;
  typedef typename ChainBelow::type ChainBelowInternalType;
  typedef LAYERTYPEOUTPUTTYPE(ChainBelowInternalType) LayerInputOutputType;
private:
  ChainBelow leaf_below;
  ChainInternal leaf_internal; //must terminate on an InputLayer (even though it's not really an input layer)
public:
  typedef LeafTag tag;
  
  SkipConnection(ChainInternal &&leaf_internal, ChainBelow &&leaf_below):
    leaf_below(std::move(leaf_below)), leaf_internal(std::move(leaf_internal)){  }
  SkipConnection(const SkipConnection &r) = delete;
  SkipConnection(SkipConnection &&r) = default;
  
  //Forward pass
  LayerInputOutputType value(const InputType &x, EnableDeriv enable_deriv = DerivNo);

  int deriv(Vector<FloatType> &cost_deriv, int off, LayerInputOutputType &&_above_deriv, InputType* input_above_deriv_return = nullptr) const;
  
  int update(int off, const Vector<FloatType> &new_params);
  
  int step(int off, const Vector<FloatType> &derivs, FloatType eps);

  //accumulated #params for layers here and below
  inline int nparams() const{ return leaf_internal.v.nparams() + leaf_below.v.nparams(); }

  size_t FLOPS(int value_or_deriv) const{ return leaf_internal.v.FLOPS(value_or_deriv) + leaf_below.v.FLOPS(value_or_deriv); }
  
  //off measured from *end*, return new off
  int getParams(Vector<FloatType> &into, int off) const;

  //For pipelining
  inline void resizeInputBuffer(size_t to){
    leaf_internal.v.resizeInputBuffer(to);
    leaf_below.v.resizeInputBuffer(to);
  }
};

#define LAYER_TYPE SkipConnection<CONFIGTYPE(Internal),INPUTTYPE(Below),DDST(internal),DDST(below)>

template<typename Internal, typename Below, typename std::enable_if<ISLEAF(Internal) && ISLEAF(Below), int>::type = 0>
auto skip_connection(Internal &&internal, Below &&below)-> LAYER_TYPE{
  return LAYER_TYPE(std::forward<Internal>(internal),std::forward<Below>(below));
}
#undef LAYER_TYPE

#include "implementation/SkipConnection.tcc"
