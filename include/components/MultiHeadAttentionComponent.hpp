#pragma once
#include <components/ScaledDotProductAttentionHeadComponent.hpp>
#include <components/BatchTensorConcatenateComponent.hpp>

//A layer implementing multi-head scaled dot-product attention. The input 3-tensor X is expected to have dimension C * E * B  in this order, where C is the size of the context, E the size of the embedding and B the batch size
template<typename _FloatType>
class MultiHeadAttentionComponent{
public:
  typedef _FloatType FloatType;
private:
  typedef Tensor<FloatType,3> TensorType;

  int C;
  int E;
  int B;
  int Nparams_layer;
  bool setup;

  std::vector< std::unique_ptr<ScaledDotProductAttentionHeadComponent<FloatType> > > heads; //Y^h = attention(X,X,X)
  BatchTensorConcatenateComponent<FloatType,3> concatY; //Yconcat_{c,:,b} = concat_h( Y^h_{c,:,b} )
  
  MatrixTensorContractComponent<FloatType,3> multW_O; //W_O{oy} Yconcat_{c, y, b}

public:
  
  MultiHeadAttentionComponent(int Nheads, Matrix<FloatType> const* const* W_Q, Matrix<FloatType> const* const* W_K, Matrix<FloatType> const* const* W_V, const Matrix<FloatType> &W_O, bool use_mask=false);
  MultiHeadAttentionComponent(int Nheads, const std::vector<Matrix<FloatType> > &W_Q, const std::vector<Matrix<FloatType> > &W_K, const std::vector<Matrix<FloatType> > &W_V, const Matrix<FloatType> &W_O, bool use_mask=false);
  
  MultiHeadAttentionComponent(const MultiHeadAttentionComponent &r) = delete;
  MultiHeadAttentionComponent(MultiHeadAttentionComponent &&r) = default;
  
  //Forward pass
  TensorType value(const TensorType &Q, const TensorType &K, const TensorType &V);

  void deriv(Vector<FloatType> &cost_deriv, int off, TensorType &&dCost_by_dOut, TensorType &dCost_by_dQ, TensorType &dCost_by_dK, TensorType &dCost_by_dV) const;

  void update(int off, const Vector<FloatType> &new_params);
  
  void step(int off, const Vector<FloatType> &derivs, FloatType eps);

  //accumulated #params for layers here and below
  inline int nparams() const{ return Nparams_layer; }

  //off measured from *end*, return new off
  void getParams(Vector<FloatType> &into, int off);

  //For pipelining
  void resizeInputBuffer(size_t to);

};

#include "implementation/MultiHeadAttentionComponent.tcc"
