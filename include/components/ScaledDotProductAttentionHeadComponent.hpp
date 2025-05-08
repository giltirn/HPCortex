#pragma once
#include <components/ScaledDotProductAttentionComponent.hpp>

//An attention head is scaled dot-product attention but with the inputs multiplied by learnable weight matrics
//Expects input tensors Q(C,E,B) ,  K(C,E,B)  and V(C,E,B)   where C is the context window size, B the batch size and E the embedding size (assumed equal)
template<typename _FloatType>
class ScaledDotProductAttentionHeadComponent{
public:
  typedef _FloatType FloatType;
private:
  // 1)  Q'_{ckb} = \sum_e (W_Q)_{k e} Q_{c e b}     k \in {1.. d_k}
  // 2)  K'_{ckb} = \sum_e (W_K)_{k e} K_{c e b}
  // 3)  S_{c c' b} = \sum_k  Q_{c k b} K_{c' k b}
  // 4)  V'_{cvb} = \sum_e (W_V)_{v e} V_{c e b}     v \in {1.. d_v}
  // 5)  SS_{c c' b} = scaled_softmax_c' ( S )    such that the probability weights sum to unity along c'
  // 6)  Out_{c v b} = \sum_c' SS_{c c' b} V'_{c' v b}
  
  int C;
  int B;
  int E;
  int d_k;
  int d_v;
  bool setup;

  typedef MatrixTensorContractComponent<FloatType,3> MatTensMulCptType;
  MatTensMulCptType multWQ; //d_k * E  matrix  W_Q   operates on Q   as  \sum_e (W_Q)_{d e} Q_{ c e b }   -> 1)
  MatTensMulCptType multWK; //d_k * E  matrix  W_K   operates on K   as  \sum_e (W_K)_{d e} K_{ c e b }   -> 2)
  MatTensMulCptType multWV; //d_v * E  matrix  W_V   operates on V   as  \sum_e (W_V)_{v e} V_{ c e b }   -> 4)

  ScaledDotProductAttentionComponent<FloatType> attention;

public:
  ScaledDotProductAttentionHeadComponent(const Matrix<FloatType> &W_Q, const Matrix<FloatType> &W_K, const Matrix<FloatType> &W_V):
    multWQ(W_Q), multWK(W_K), multWV(W_V),    
    d_k(W_Q.size(0)), d_v(W_V.size(0)), E(W_Q.size(1)),
    attention(d_k,d_v),
    setup(false)
  {
    assert(W_K.size(0) == d_k);
    assert(W_K.size(1) == E);
    assert(W_V.size(1) == E);
  }

  ScaledDotProductAttentionHeadComponent(const ScaledDotProductAttentionHeadComponent &r) = delete;
  ScaledDotProductAttentionHeadComponent(ScaledDotProductAttentionHeadComponent &&r) = default;
  
  //Forward pass
  Tensor<FloatType,3> value(const Tensor<FloatType,3> &Q, const Tensor<FloatType,3> &K, Tensor<FloatType,3> &V);

  void deriv(Vector<FloatType> &cost_deriv, int off, Tensor<FloatType,3> &&dCost_by_dOut, Tensor<FloatType,3> &dCost_by_dQ, Tensor<FloatType,3> &dCost_by_dK, Tensor<FloatType,3> &dCost_by_dV) const;

  void update(int off, const Vector<FloatType> &new_params);
  
  void step(int off, const Vector<FloatType> &derivs, FloatType eps);

  //accumulated #params for layers here and below
  inline int nparams() const{ return multWQ.nparams() + multWK.nparams() + multWV.nparams(); }

  //off measured from *end*, return new off
  void getParams(Vector<FloatType> &into, int off);

  
  //For pipelining
  inline void resizeInputBuffer(size_t to){
    multWQ.resizeInputBuffer(to);
    multWK.resizeInputBuffer(to);
    multWV.resizeInputBuffer(to);
    attention.resizeInputBuffer(to);
  }

};

#include "implementation/ScaledDotProductAttentionHeadComponent.tcc"
