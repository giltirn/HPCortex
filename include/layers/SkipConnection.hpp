#pragma once
#include "LayerCommon.hpp"

template<typename _FloatType, typename _InputType, typename ChainInternal, typename ChainBelow>
class SkipConnection{
  public:
  typedef _FloatType FloatType;
  typedef _InputType InputType;
private:
  ChainBelow leaf_below;
  ChainInternal leaf_internal; //must terminate on an InputLayer (even though it's not really an input layer)
  size_t in_size;
  size_t batch_size;
  mutable RingBuffer<Matrix<FloatType> > in_buf;
public:
  typedef LeafTag tag;
  
  SkipConnection(ChainInternal &&leaf_internal, ChainBelow &&leaf_below):
    leaf_below(std::move(leaf_below)), leaf_internal(std::move(leaf_internal)),  in_buf(1), batch_size(0), in_size(0){  }
  SkipConnection(const SkipConnection &r) = delete;
  SkipConnection(SkipConnection &&r) = default;
  
  //Forward pass
  Matrix<FloatType> value(const InputType &x);

  void deriv(Vector<FloatType> &cost_deriv, int off, Matrix<FloatType> &&_above_deriv, InputType* input_above_deriv_return = nullptr) const;
  
  void update(int off, const Vector<FloatType> &new_params);
  
  void step(int off, const Vector<FloatType> &derivs, FloatType eps);

  //accumulated #params for layers here and below
  inline int nparams() const{ return leaf_internal.v.nparams() + leaf_below.v.nparams(); }

  //off measured from *end*, return new off
  void getParams(Vector<FloatType> &into, int off);

  //For pipelining
  inline void resizeInputBuffer(size_t to){
    in_buf.resize(to);
    leaf_internal.v.resizeInputBuffer(to);
    leaf_below.v.resizeInputBuffer(to);
  }
};

#define LAYER_TYPE SkipConnection<FLOATTYPE(Internal),INPUTTYPE(Below),DDST(internal),DDST(below)>

template<typename Internal, typename Below, typename std::enable_if<ISLEAF(Internal) && ISLEAF(Below), int>::type = 0>
auto skip_connection(Internal &&internal, Below &&below)-> LAYER_TYPE{
  return LAYER_TYPE(std::forward<Internal>(internal),std::forward<Below>(below));
}
#undef LAYER_TYPE

#include "implementation/SkipConnection.tcc"
