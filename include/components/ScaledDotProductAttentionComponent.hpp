#pragma once
#include <cmath>
#include <type_traits>
#include <sstream>
#include <ModelConfig.hpp>
#include <Tensors.hpp>
#include <Linalg.hpp>

#include <components/BatchedMatrixRowSoftMaxComponent.hpp>
#include <components/Batch3tensorPairContractComponent.hpp>

//A component implementing scaled dot-product attention with optional masking. Expects input tensors Q(C,d_k,B) ,  K(C,d_k,B)  and V(C,d_v,B)   where C is the context window size, B the batch size and d_k, d_v arbitrary
template<typename Config>
class ScaledDotProductAttentionComponent{
public:
  EXTRACT_CONFIG_TYPES;
private:
  //Attention:
  // 1)  S_{c c' b} = \sum_k  Q_{c k b} K_{c' k b}
  // 2)  SS_{c c' b} = scaled_softmax_c' ( S )    such that the probability weights sum to unity along c'
  // 3)  Out_{c v b} = \sum_c' SS_{c c' b} V_{c' v b}

  int C;
  int B;
  int d_k;
  int d_v;
  bool setup;
  
  Batch3tensorPairContractComponent<Config> mulQKtoGetS; //contract on indices 1,1 and normalize 1/sqrt(d_k)
  BatchedMatrixRowSoftMaxComponent<Config> softmaxS_to_SS; //softmax on index 1
  Batch3tensorPairContractComponent<Config> mulSSVtoGetOut; //contract SS with V on indices 1,0 

public:
  
  ScaledDotProductAttentionComponent(int d_k, int d_v, int use_mask = false): 
    d_k(d_k), d_v(d_v),
    mulQKtoGetS(1,1, 1./sqrt(FloatType(d_k))),
    softmaxS_to_SS(use_mask),
    mulSSVtoGetOut(1,0),
    setup(false)
  { }
  ScaledDotProductAttentionComponent(const ScaledDotProductAttentionComponent &r) = delete;
  ScaledDotProductAttentionComponent(ScaledDotProductAttentionComponent &&r) = default;
  
  //Forward pass
  Tensor<FloatType,3> value(const Tensor<FloatType,3> &Q, const Tensor<FloatType,3> &K, Tensor<FloatType,3> &V);

  void deriv(Tensor<FloatType,3> &&dCost_by_dOut, Tensor<FloatType,3> &dCost_by_dQ, Tensor<FloatType,3> &dCost_by_dK, Tensor<FloatType,3> &dCost_by_dV) const;

  size_t FLOPS(int value_or_deriv) const{ return mulQKtoGetS.FLOPS(value_or_deriv) + softmaxS_to_SS.FLOPS(value_or_deriv) + mulSSVtoGetOut.FLOPS(value_or_deriv); }

  int nparams() const{ return 0; }
  
  //For pipelining
  inline void resizeInputBuffer(size_t to){
    mulQKtoGetS.resizeInputBuffer(to);
    mulSSVtoGetOut.resizeInputBuffer(to);
    softmaxS_to_SS.resizeInputBuffer(to);
  }

};

#include "implementation/ScaledDotProductAttentionComponent.tcc"
