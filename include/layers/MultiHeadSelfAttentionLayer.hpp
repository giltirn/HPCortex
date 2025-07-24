#pragma once
#include "LayerCommon.hpp"
#include <components/MultiHeadAttentionComponent.hpp>

//A layer implementing multi-head scaled dot-product self-attention. The input 3-tensor X is expected to have dimension C * E * B  in this order, where C is the size of the context, E the size of the embedding and B the batch size
//Require W_Q[i], W_K[i] :  d_qk^(i) x E,     W_V[i] : d_v^(i) x E      W_O :  E x sum_i d_v^(i)
template<typename Config, typename _InputType, typename Store>
class MultiHeadSelfAttentionLayer{
public:
  EXTRACT_CONFIG_TYPES;
  typedef _InputType InputType;
private:
  typedef Tensor<FloatType,3> LayerInputType;

  MultiHeadAttentionComponent<Config> mha;
  Store leaf;

public:
  typedef LeafTag tag;
  
  MultiHeadSelfAttentionLayer(Store &&leaf, int Nheads, Matrix<FloatType> const* const* W_Q, Matrix<FloatType> const* const* W_K, Matrix<FloatType> const* const* W_V, const Matrix<FloatType> &W_O, bool use_mask = false);
  MultiHeadSelfAttentionLayer(Store &&leaf, int Nheads, const std::vector<Matrix<FloatType> > &W_Q, const std::vector<Matrix<FloatType> > &W_K, const std::vector<Matrix<FloatType> > &W_V, const Matrix<FloatType> &W_O, bool use_mask = false);
  
  MultiHeadSelfAttentionLayer(const MultiHeadSelfAttentionLayer &r) = delete;
  MultiHeadSelfAttentionLayer(MultiHeadSelfAttentionLayer &&r) = default;
  
  //Forward pass
  Tensor<FloatType,3> value(const InputType &x);

  int deriv(Vector<FloatType> &cost_deriv, int off, Tensor<FloatType,3> &&_above_deriv, InputType* input_above_deriv_return = nullptr) const;

  int update(int off, const Vector<FloatType> &new_params);
  
  int step(int off, const Vector<FloatType> &derivs, FloatType eps);

  //accumulated #params for layers here and below
  inline int nparams() const{ return mha.nparams() + leaf.v.nparams(); }

  size_t FLOPS(int value_or_deriv) const{ return mha.FLOPS(value_or_deriv) + leaf.v.FLOPS(value_or_deriv); }
  
  //off measured from *end*, return new off
  int getParams(Vector<FloatType> &into, int off) const;

  //For pipelining
  void resizeInputBuffer(size_t to);

};

#define LAYER_TYPE MultiHeadSelfAttentionLayer<CONFIGTYPE(U),INPUTTYPE(U),DDST(u)>
template<typename U, typename std::enable_if<ISLEAF(U), int>::type = 0>
auto multihead_self_attention_layer(int Nheads,
				    Matrix<FLOATTYPE(U)> const* const* W_Q,
				    Matrix<FLOATTYPE(U)> const* const* W_K,
				    Matrix<FLOATTYPE(U)> const* const* W_V,
				    const Matrix<FLOATTYPE(U)> &W_O,
				    bool use_mask,
				    U &&u)-> LAYER_TYPE{
  return LAYER_TYPE(std::forward<U>(u), Nheads, W_Q, W_K, W_V, W_O, use_mask);
}
template<typename U, typename std::enable_if<ISLEAF(U), int>::type = 0>
auto multihead_self_attention_layer(int Nheads,
				    Matrix<FLOATTYPE(U)> const* const* W_Q,
				    Matrix<FLOATTYPE(U)> const* const* W_K,
				    Matrix<FLOATTYPE(U)> const* const* W_V,
				    const Matrix<FLOATTYPE(U)> &W_O,
				    U &&u)-> LAYER_TYPE{
  return LAYER_TYPE(std::forward<U>(u), Nheads, W_Q, W_K, W_V, W_O, false);
}




template<typename U, typename std::enable_if<ISLEAF(U), int>::type = 0>
auto multihead_self_attention_layer(int Nheads,
				    const std::vector<Matrix<FLOATTYPE(U)> > &W_Q,
				    const std::vector<Matrix<FLOATTYPE(U)> > &W_K,
				    const std::vector<Matrix<FLOATTYPE(U)> > &W_V,
				    const Matrix<FLOATTYPE(U)> &W_O,
				    bool use_mask,
				    U &&u)-> LAYER_TYPE{
  return LAYER_TYPE(std::forward<U>(u), Nheads, W_Q, W_K, W_V, W_O, use_mask);
}
template<typename U, typename std::enable_if<ISLEAF(U), int>::type = 0>
auto multihead_self_attention_layer(int Nheads,
				    const std::vector<Matrix<FLOATTYPE(U)> > &W_Q,
				    const std::vector<Matrix<FLOATTYPE(U)> > &W_K,
				    const std::vector<Matrix<FLOATTYPE(U)> > &W_V,
				    const Matrix<FLOATTYPE(U)> &W_O,
				    U &&u)-> LAYER_TYPE{
  return LAYER_TYPE(std::forward<U>(u), Nheads, W_Q, W_K, W_V, W_O, false);
}


//Default initialization has W_Q,W_K,W_V all of size E/Nheads x E  and W_O of size ExE
//each initialized using Glorot uniform
template<typename U, typename std::enable_if<ISLEAF(U), int>::type = 0>
auto multihead_self_attention_layer(int Nheads,
				    int E,
				    bool use_mask,
				    U &&u)-> LAYER_TYPE{
  typedef FLOATTYPE(U) FloatType;
  assert(E % Nheads == 0);
  int d_qkv = E/Nheads;
  std::vector< Matrix<FloatType> > W_Q(Nheads, Matrix<FloatType>(d_qkv,E));
  std::vector< Matrix<FloatType> > W_K(Nheads, Matrix<FloatType>(d_qkv,E));
  std::vector< Matrix<FloatType> > W_V(Nheads, Matrix<FloatType>(d_qkv,E));
  for(int h=0;h<Nheads;h++){
    glorotUniformRandom(W_Q[h]); glorotUniformRandom(W_K[h]); glorotUniformRandom(W_V[h]);
  }
  Matrix<FloatType> W_O(E,E);
  glorotUniformRandom(W_O);
  
  auto layer = LAYER_TYPE(std::forward<U>(u), Nheads, W_Q, W_K, W_V, W_O, use_mask);  
  return layer;
}
template<typename U, typename std::enable_if<ISLEAF(U), int>::type = 0>
auto multihead_self_attention_layer(int Nheads,
				    int E,
				    U &&u)-> LAYER_TYPE{
  return multihead_self_attention_layer(Nheads,E,false,std::forward<U>(u));
}

  
#undef LAYER_TYPE

#include "implementation/MultiHeadSelfAttentionLayer.tcc"
