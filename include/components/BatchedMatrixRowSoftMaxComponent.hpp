#pragma once
#include <cmath>
#include <type_traits>
#include <sstream>
#include <Tensors.hpp>
#include <RingBuffer.hpp>
#include <Linalg.hpp>

//A component implementing the softmax operation over the rows of a batched matrix ( Tensor<FloatType,3> with the last dimension the batch dimension ) with optional masking
//softmax(In + M)  where M is zero (no masking) or is *strictly* upper triangular (diagonal elements also zero) with nonzero elements =-inf
//If masking, the input must be a square
template<typename _FloatType>
class BatchedMatrixRowSoftMaxComponent{
public:
  typedef _FloatType FloatType;
private:
  FloatType beta;
  mutable RingBuffer<Tensor<FloatType,3> > out_buf;
  bool use_mask;
public:
  
  BatchedMatrixRowSoftMaxComponent(bool use_mask = false, FloatType beta = 1.0): use_mask(use_mask), beta(beta){ }
  BatchedMatrixRowSoftMaxComponent(const BatchedMatrixRowSoftMaxComponent &r) = delete;
  BatchedMatrixRowSoftMaxComponent(BatchedMatrixRowSoftMaxComponent &&r) = default;
  
  Tensor<FloatType,3> value(const Tensor<FloatType,3> &in) const;
  void deriv(Tensor<FloatType,3> &&dcost_by_dOut, Tensor<FloatType,3> &dcost_by_dIn) const;
    
  inline int nparams() const{ return 0; }

  inline void resizeInputBuffer(size_t to){
    out_buf.resize(to);
  }
};

#include "implementation/BatchedMatrixRowSoftMaxComponent.tcc"
