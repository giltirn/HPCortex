#pragma once
#include "LayerCommon.hpp"
#include <components/ScaledDotProductAttentionHeadComponent.hpp>
#include <components/BatchTensorConcatenateComponent.hpp>

//A layer implementing multi-head scaled dot-product self-attention. The input 3-tensor X is expected to have dimension C * E * B  in this order, where C is the size of the context, E the size of the embedding and B the batch size
template<typename _FloatType, typename _InputType, typename Store>
class MultiHeadSelfAttentionLayer{
public:
  typedef _FloatType FloatType;
  typedef _InputType InputType;
private:
  typedef Tensor<FloatType,3> LayerInputType;

  int C;
  int E;
  int B;
  int Nparams_layer;
  bool setup;

  std::vector< std::unique_ptr<ScaledDotProductAttentionHeadComponent<FloatType> > > heads; //Y^h = attention(X,X,X)
  BatchTensorConcatenateComponent<FloatType,3> concatY; //Yconcat_{c,:,b} = concat_h( Y^h_{c,:,b} )
  
  MatrixTensorContractComponent<FloatType,3> multW_O; //W_O{oy} Yconcat_{c, y, b}
  Store leaf;

public:
  typedef LeafTag tag;
  
  MultiHeadSelfAttentionLayer(Store &&leaf, int Nheads, Matrix<FloatType> const* const* W_Q, Matrix<FloatType> const* const* W_K, Matrix<FloatType> const* const* W_V, const Matrix<FloatType> &W_O);
  MultiHeadSelfAttentionLayer(const MultiHeadSelfAttentionLayer &r) = delete;
  MultiHeadSelfAttentionLayer(MultiHeadSelfAttentionLayer &&r) = default;
  
  //Forward pass
  Tensor<FloatType,3> value(const InputType &x);

  void deriv(Vector<FloatType> &cost_deriv, int off, Tensor<FloatType,3> &&_above_deriv, InputType* input_above_deriv_return = nullptr) const;

  void update(int off, const Vector<FloatType> &new_params);
  
  void step(int off, const Vector<FloatType> &derivs, FloatType eps);

  //accumulated #params for layers here and below
  inline int nparams() const{ return Nparams_layer + leaf.v.nparams(); }

  //off measured from *end*, return new off
  void getParams(Vector<FloatType> &into, int off);

  //For pipelining
  void resizeInputBuffer(size_t to);

};

#define LAYER_TYPE MultiHeadSelfAttentionLayer<FLOATTYPE(U),INPUTTYPE(U),DDST(u)>
template<typename U, typename std::enable_if<ISLEAF(U), int>::type = 0>
auto multihead_self_attention_layer(U &&u,
				    int Nheads,
				    Matrix<FLOATTYPE(U)> const* const* W_Q,
				    Matrix<FLOATTYPE(U)> const* const* W_K,
				    Matrix<FLOATTYPE(U)> const* const* W_V,
				    const Matrix<FLOATTYPE(U)> &W_O)-> LAYER_TYPE{
  return LAYER_TYPE(std::forward<U>(u), Nheads, W_Q, W_K, W_V, W_O);
}
#undef LAYER_TYPE

#include "implementation/MultiHeadSelfAttentionLayer.tcc"
