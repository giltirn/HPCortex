#pragma once
#include "LayerCommon.hpp"
#include <components/ScaledDotProductAttentionHeadComponent.hpp>

//A layer implementing scaled dot-product self-attention with optional masking. The input 3-tensor X is expected to have dimension C * E * B  in this order, where C is the size of the context, E the size of the embedding and B the batch size
template<typename Config, typename _InputType, typename Store>
class ScaledDotProductSelfAttentionLayer{
public:
  EXTRACT_CONFIG_TYPES;
  typedef _InputType InputType;
private:
  typedef Tensor<FloatType,3> LayerInputType;

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
  
  ScaledDotProductAttentionHeadComponent<Config> attentionQKV;
  
  Store leaf;

public:
  typedef LeafTag tag;
  
  ScaledDotProductSelfAttentionLayer(Store &&leaf, const Matrix<FloatType> &W_Q, const Matrix<FloatType> &W_K, const Matrix<FloatType> &W_V, bool use_mask = false):
    leaf(std::move(leaf)),
    d_k(W_Q.size(0)), d_v(W_V.size(0)), E(W_Q.size(1)),
    attentionQKV(W_Q,W_K,W_V,use_mask),
    setup(false)
  {
    assert(W_K.size(0) == d_k);
    assert(W_K.size(1) == E);
    assert(W_V.size(1) == E);
  }
  ScaledDotProductSelfAttentionLayer(const ScaledDotProductSelfAttentionLayer &r) = delete;
  ScaledDotProductSelfAttentionLayer(ScaledDotProductSelfAttentionLayer &&r) = default;
  
  //Forward pass
  Tensor<FloatType,3> value(const InputType &x, EnableDeriv enable_deriv = DerivNo);

  int deriv(Vector<FloatType> &cost_deriv, int off, Tensor<FloatType,3> &&_above_deriv, InputType* input_above_deriv_return = nullptr) const;

  int update(int off, const Vector<FloatType> &new_params);
  
  int step(int off, const Vector<FloatType> &derivs, FloatType eps);

  //accumulated #params for layers here and below
  inline int nparams() const{ return attentionQKV.nparams() + leaf.v.nparams(); }

  //off measured from *end*, return new off
  int getParams(Vector<FloatType> &into, int off) const;

  size_t FLOPS(int value_or_deriv) const{ return attentionQKV.FLOPS(value_or_deriv) + (value_or_deriv == 1 ? B*C*E*2 : 0) + leaf.v.FLOPS(value_or_deriv); }
  
  //For pipelining
  inline void resizeInputBuffer(size_t to){
    attentionQKV.resizeInputBuffer(to);
    leaf.v.resizeInputBuffer(to);
  }
};

#define LAYER_TYPE ScaledDotProductSelfAttentionLayer<CONFIGTYPE(U),INPUTTYPE(U),DDST(u)>
template<typename U, typename std::enable_if<ISLEAF(U), int>::type = 0>
auto scaled_dotproduct_self_attention_layer(const Matrix<FLOATTYPE(U)> &W_Q,
					    const Matrix<FLOATTYPE(U)> &W_K,
					    const Matrix<FLOATTYPE(U)> &W_V,
					    bool use_mask,
					    U &&u)-> LAYER_TYPE{
  return LAYER_TYPE(std::forward<U>(u), W_Q, W_K, W_V, use_mask);
}
template<typename U, typename std::enable_if<ISLEAF(U), int>::type = 0>
auto scaled_dotproduct_self_attention_layer(const Matrix<FLOATTYPE(U)> &W_Q,
					    const Matrix<FLOATTYPE(U)> &W_K,
					    const Matrix<FLOATTYPE(U)> &W_V,
					    U &&u)-> LAYER_TYPE{
  return LAYER_TYPE(std::forward<U>(u), W_Q, W_K, W_V, false);
}
#undef LAYER_TYPE

#include "implementation/ScaledDotProductSelfAttentionLayer.tcc"
