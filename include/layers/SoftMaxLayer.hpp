#pragma once
#include "LayerCommon.hpp"

template<typename _FloatType, typename _InputType, typename Store >
class SoftMaxLayer{  
public:
  typedef _FloatType FloatType;
  typedef _InputType InputType;
  typedef LeafTag tag;
private:
  Store leaf;
  FloatType beta;
  mutable RingBuffer<Matrix<FloatType> > out_buf;
  int nlogp; //rows of input/output
  int batch_size;
public:
  
  inline SoftMaxLayer(Store &&leaf, FloatType beta = 1.0): leaf(std::move(leaf)), beta(beta){}
  inline SoftMaxLayer(SoftMaxLayer &&r) = default;
  inline SoftMaxLayer(const SoftMaxLayer &r) = delete;

  Matrix<FloatType> value(const InputType &x);
  
  void deriv(Vector<FloatType> &cost_deriv, int off, Matrix<FloatType> &&above_deriv, InputType* input_above_deriv_return = nullptr) const;
  
  inline void update(int off, const Vector<FloatType> &new_params){ leaf.v.update(off, new_params); }

  inline void step(int off, const Vector<FloatType> &derivs, FloatType eps){ leaf.v.step(off,derivs,eps); }
  
  inline int nparams() const{ return leaf.v.nparams(); }

  inline void getParams(Vector<FloatType> &into, int off){ leaf.v.getParams(into,off); }

  //For pipelining
  inline void resizeInputBuffer(size_t to){
    out_buf.resize(to);
    leaf.v.resizeInputBuffer(to);
  }
};

template<typename U, typename std::enable_if<ISLEAF(U), int>::type = 0>
auto softmax_layer(U &&u, FLOATTYPE(U) beta)->SoftMaxLayer<FLOATTYPE(U),INPUTTYPE(U),DDST(u)>{
  return SoftMaxLayer<FLOATTYPE(U),INPUTTYPE(U),DDST(u)>(std::forward<U>(u), beta);
}

#include "implementation/SoftMaxLayer.tcc"

