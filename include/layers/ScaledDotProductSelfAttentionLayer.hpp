#pragma once
#include "LayerCommon.hpp"
#include <components/MatrixTensorContractComponent.hpp>
#include <components/ScaledDotProductAttentionComponent.hpp>

//A layer implementing scaled dot-product self-attention. The input 3-tensor X is expected to have dimension C * E * B  in this order, where C is the size of the context, E the size of the embedding and B the batch size
template<typename _FloatType, typename _InputType, typename Store>
class ScaledDotProductSelfAttentionLayer{
public:
  typedef _FloatType FloatType;
  typedef _InputType InputType;
private:
  typedef Tensor<FloatType,3> LayerInputType;
  typedef MatrixTensorContractComponent<FloatType,3> MatTensMulCptType;

  //Attention:
  // 1)  Q_{ckb} = \sum_e (W_Q)_{k e} X_{c e b}     k \in {1.. d_k}
  // 2)  K_{ckb} = \sum_e (W_K)_{k e} X_{c e b}
  // 3)  S_{c c' b} = \sum_k  Q_{c k b} K_{c' k b}
  // 4)  V_{cvb} = \sum_e (W_V)_{v e} X_{c e b}     v \in {1.. d_v}
  // 5)  SS_{c c' b} = scaled_softmax_c' ( S )    such that the probability weights sum to unity along c'
  // 6)  Out_{c v b} = \sum_c' SS_{c c' b} V_{c' v b}

  int C;
  int E;
  int B;
  int d_k;
  int d_v;
  bool setup;
  
  MatTensMulCptType multWQ; //D_k * E  matrix  W_Q   operates on X   as  \sum_e (W_Q)_{d e} X_{ c e b }   -> 1)
  MatTensMulCptType multWK; //D_k * E  matrix  W_K   operates on X   as  \sum_e (W_K)_{d e} X_{ c e b }   -> 2)
  MatTensMulCptType multWV; //D_v * E  matrix  W_V   operates on X   as  \sum_e (W_V)_{v e} X_{ c e b }   -> 4)
  ScaledDotProductAttentionComponent<FloatType> attentionQKV;
  
  Store leaf;

public:
  typedef LeafTag tag;
  
  ScaledDotProductSelfAttentionLayer(Store &&leaf, const Matrix<FloatType> &W_Q, const Matrix<FloatType> &W_K, const Matrix<FloatType> &W_V):
    leaf(std::move(leaf)),
    multWQ(W_Q), multWK(W_K), multWV(W_V),    
    d_k(W_Q.size(0)), d_v(W_V.size(0)), E(W_Q.size(1)),
    attentionQKV(d_k,d_v),
    setup(false)
  {
    assert(W_K.size(0) == d_k);
    assert(W_K.size(1) == E);
    assert(W_V.size(1) == E);
  }
  ScaledDotProductSelfAttentionLayer(const ScaledDotProductSelfAttentionLayer &r) = delete;
  ScaledDotProductSelfAttentionLayer(ScaledDotProductSelfAttentionLayer &&r) = default;
  
  //Forward pass
  Tensor<FloatType,3> value(const InputType &x);

  void deriv(Vector<FloatType> &cost_deriv, int off, Tensor<FloatType,3> &&_above_deriv, InputType* input_above_deriv_return = nullptr) const;

  void update(int off, const Vector<FloatType> &new_params);
  
  void step(int off, const Vector<FloatType> &derivs, FloatType eps);

  //accumulated #params for layers here and below
  inline int nparams() const{ return multWQ.nparams() + multWK.nparams() + multWV.nparams() + leaf.v.nparams(); }

  //off measured from *end*, return new off
  void getParams(Vector<FloatType> &into, int off);

  //For pipelining
  inline void resizeInputBuffer(size_t to){
    multWQ.resizeInputBuffer(to);
    multWK.resizeInputBuffer(to);
    multWV.resizeInputBuffer(to);
    attentionQKV.resizeInputBuffer(to);
    leaf.v.resizeInputBuffer(to);
  }

};

#define LAYER_TYPE ScaledDotProductSelfAttentionLayer<FLOATTYPE(U),INPUTTYPE(U),DDST(u)>
template<typename U, typename std::enable_if<ISLEAF(U), int>::type = 0>
auto scaled_dotproduct_self_attention_layer(U &&u,
				       const Matrix<FLOATTYPE(U)> &W_Q,
				       const Matrix<FLOATTYPE(U)> &W_K,
				       const Matrix<FLOATTYPE(U)> &W_V)-> LAYER_TYPE{
  return LAYER_TYPE(std::forward<U>(u), W_Q, W_K, W_V);
}
#undef LAYER_TYPE

#include "implementation/ScaledDotProductSelfAttentionLayer.tcc"
