#pragma once
#include <type_traits>
#include <sstream>
#include <Tensors.hpp>
#include <RingBuffer.hpp>
#include <Linalg.hpp>

//A component implementing the softmax operation on a batch tensor (one for which the last dimension is the batch dimension) along an arbitrary dimension other than the batch dimension
template<typename _FloatType, int TensDim>
class SoftMaxComponent{
public:
  typedef _FloatType FloatType;
private:
  int softmax_dim;
  FloatType beta;
  mutable RingBuffer<Tensor<FloatType,TensDim> > out_buf;
public:
  
  SoftMaxComponent(int softmax_dim, FloatType beta = 1.0): softmax_dim(softmax_dim), beta(beta){
    assert(softmax_dim >=0 && softmax_dim < TensDim-1);
  }
  SoftMaxComponent(const SoftMaxComponent &r) = delete;
  SoftMaxComponent(SoftMaxComponent &&r) = default;
  
  Tensor<FloatType,TensDim> value(const Tensor<FloatType,TensDim> &in) const;
  void deriv(Tensor<FloatType,TensDim> &&dcost_by_dOut, Tensor<FloatType,TensDim> &dcost_by_dIn) const;
    
  inline int nparams() const{ return 0; }

  inline void resizeInputBuffer(size_t to){
    out_buf.resize(to);
  }
};

#include "implementation/SoftMaxComponent.tcc"
