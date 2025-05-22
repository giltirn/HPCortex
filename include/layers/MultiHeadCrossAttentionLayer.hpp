#pragma once
#include "LayerCommon.hpp"
#include<components/MultiHeadAttentionComponent.hpp>

//Implementation of multi-head cross-attention where queries and keys are produced from one input chain and values from another
template<typename _FloatType, typename _InputTypeQK, typename _InputTypeV, typename ChainQK, typename ChainV>
class MultiHeadCrossAttentionLayer{
  public:
  typedef _FloatType FloatType;
  typedef std::pair<_InputTypeQK,_InputTypeV> InputType; //compound type for inputs two the two independent chains

  typedef Tensor<FloatType,3> TensorType;
  
private:
  ChainQK leaf_QK;
  ChainV leaf_V;
  MultiHeadAttentionComponent<FloatType> attention;
public:
  typedef LeafTag tag;
  
  MultiHeadCrossAttentionLayer(ChainQK &&leaf_QK, ChainV &&leaf_V,
		      int Nheads, Matrix<FloatType> const* const* W_Q, Matrix<FloatType> const* const* W_K, Matrix<FloatType> const* const* W_V, const Matrix<FloatType> &W_O, bool use_mask=false):
    leaf_QK(std::move(leaf_QK)), leaf_V(std::move(leaf_V)), attention(Nheads,W_Q,W_K,W_V,W_O,use_mask){  }
  MultiHeadCrossAttentionLayer(const MultiHeadCrossAttentionLayer &r) = delete;
  MultiHeadCrossAttentionLayer(MultiHeadCrossAttentionLayer &&r) = default;
  
  //Forward pass
  TensorType value(const InputType &x){
    //This layer breaks up value passed from above into the chain inputs
    TensorType in_QK = leaf_QK.v.value(x.first);
    TensorType in_V = leaf_V.v.value(x.second);
    return attention.value(in_QK,in_QK,in_V);
  }
  
  int deriv(Vector<FloatType> &cost_deriv, int off, TensorType &&_above_deriv, InputType* input_above_deriv_return = nullptr) const{
    int p = off;
    TensorType layer_deriv_Q, layer_deriv_K, layer_deriv_V;
    attention.deriv(cost_deriv, p, std::move(_above_deriv), layer_deriv_Q, layer_deriv_K, layer_deriv_V);
    p += attention.nparams();
      
    //Sum derivs for Q, K
    {
      {
	autoView(layer_deriv_Q_v, layer_deriv_Q, DeviceReadWrite);
	autoView(layer_deriv_K_v, layer_deriv_K, DeviceRead);
	accelerator_for3d(b, layer_deriv_Q.size(2), k, layer_deriv_Q.size(1), c, layer_deriv_Q.size(0), 1, {
	    layer_deriv_Q_v(c,k,b) += layer_deriv_K_v(c,k,b);
	  });
      }
      TensorType throwaway(std::move(layer_deriv_K));
    }
	
    p = leaf_QK.v.deriv(cost_deriv, p, std::move(layer_deriv_Q), input_above_deriv_return != nullptr ? &input_above_deriv_return->first : nullptr);
    return leaf_V.v.deriv(cost_deriv, p, std::move(layer_deriv_V), input_above_deriv_return != nullptr ? &input_above_deriv_return->second : nullptr);
  }

  int update(int off, const Vector<FloatType> &new_params){
    int p=off;
    attention.update(p,new_params);
    p+=attention.nparams();
    p = leaf_QK.v.update(p, new_params);
    return leaf_V.v.update(p,new_params);
  }
  int step(int off, const Vector<FloatType> &derivs, FloatType eps){
    int p=off;
    attention.step(p,derivs,eps);
    p+=attention.nparams();
    p = leaf_QK.v.step(p,derivs,eps);
    return leaf_V.v.step(p,derivs,eps);
  }


  //accumulated #params for layers here and below
  inline int nparams() const{ return attention.nparams() + leaf_QK.v.nparams() +  leaf_V.v.nparams() ; }

  int getParams(Vector<FloatType> &into, int off){
    int p=off;
    attention.getParams(into,p);
    p+=attention.nparams();
    p = leaf_QK.v.getParams(into,p);
    return leaf_V.v.getParams(into,p);
  }

  inline void resizeInputBuffer(size_t to){
    attention.resizeInputBuffer(to);
    leaf_QK.v.resizeInputBuffer(to);
    leaf_V.v.resizeInputBuffer(to);
  }
};

#define LAYER_TYPE MultiHeadCrossAttentionLayer<FLOATTYPE(ChainQK), \
						INPUTTYPE(ChainQK),INPUTTYPE(ChainV),\
						DDST(chain_QK),DDST(chain_V)>
template<typename ChainQK, typename ChainV,
	 typename std::enable_if<ISLEAF(ChainQK) && ISLEAF(ChainV) && std::is_same<FLOATTYPE(ChainQK),FLOATTYPE(ChainV)>::value  , int>::type = 0
	 >
auto multihead_cross_attention_layer(ChainQK &&chain_QK, ChainV &&chain_V,
				     int Nheads,
				     Matrix<FLOATTYPE(ChainQK)> const* const* W_Q,
				     Matrix<FLOATTYPE(ChainQK)> const* const* W_K,
				     Matrix<FLOATTYPE(ChainQK)> const* const* W_V,
				     const Matrix<FLOATTYPE(ChainQK)> &W_O,
				     bool use_mask = false)-> LAYER_TYPE{
  return LAYER_TYPE(std::forward<ChainQK>(chain_QK), std::forward<ChainV>(chain_V), Nheads, W_Q, W_K, W_V, W_O, use_mask);
}
#undef LAYER_TYPE
