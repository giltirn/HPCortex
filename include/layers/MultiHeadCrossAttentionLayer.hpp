#pragma once
#include "LayerCommon.hpp"
#include<components/MultiHeadAttentionComponent.hpp>

//Implementation of multi-head cross-attention where keys and values are produced from one input chain and queries from another
//Require W_Q[i], W_K[i] :  d_qkv^(i) x E,     W_V[i] : d_qkv^(i) x E      W_O :  E x sum_i d_qkv^(i)
template<typename _FloatType, typename _InputType, typename StoreKV, typename StoreQ>
class MultiHeadCrossAttentionLayer{
  public:
  typedef _FloatType FloatType;
  typedef _InputType InputType; //fundamental model input

  typedef Tensor<FloatType,3> TensorType;
  
private:
  StoreKV leaf_KV;
  StoreQ leaf_Q;
  MultiHeadAttentionComponent<FloatType> attention;
public:
  typedef LeafTag tag;
  
  MultiHeadCrossAttentionLayer(StoreKV &&leaf_KV, StoreQ &&leaf_Q,
		      int Nheads, Matrix<FloatType> const* const* W_Q, Matrix<FloatType> const* const* W_K, Matrix<FloatType> const* const* W_V, const Matrix<FloatType> &W_O, bool use_mask=false):
    leaf_KV(std::move(leaf_KV)), leaf_Q(std::move(leaf_Q)), attention(Nheads,W_Q,W_K,W_V,W_O,use_mask){  }

  MultiHeadCrossAttentionLayer(StoreKV &&leaf_KV, StoreQ &&leaf_Q,
			       int Nheads, const std::vector<Matrix<FloatType> > &W_Q, const std::vector<Matrix<FloatType> > &W_K, const std::vector<Matrix<FloatType> > &W_V, const Matrix<FloatType> &W_O, bool use_mask=false):
    leaf_KV(std::move(leaf_KV)), leaf_Q(std::move(leaf_Q)), attention(Nheads,W_Q,W_K,W_V,W_O,use_mask){  }
  
  MultiHeadCrossAttentionLayer(const MultiHeadCrossAttentionLayer &r) = delete;
  MultiHeadCrossAttentionLayer(MultiHeadCrossAttentionLayer &&r) = default;
  
  //Forward pass
  TensorType value(const InputType &x){
    TensorType in_KV = leaf_KV.v.value(x);
    TensorType in_Q = leaf_Q.v.value(x);
    return attention.value(in_Q,in_KV,in_KV);
  }
  
  int deriv(Vector<FloatType> &cost_deriv, int off, TensorType &&_above_deriv, InputType* input_above_deriv_return = nullptr) const{
    int p = off;
    TensorType layer_deriv_Q, layer_deriv_K, layer_deriv_V;
    attention.deriv(cost_deriv, p, std::move(_above_deriv), layer_deriv_Q, layer_deriv_K, layer_deriv_V);
    p += attention.nparams();
      
    //Sum derivs for K, V
    {
      {
	autoView(layer_deriv_K_v, layer_deriv_K, DeviceReadWrite);
	autoView(layer_deriv_V_v, layer_deriv_V, DeviceRead);
	accelerator_for3d(b, layer_deriv_K.size(2), k, layer_deriv_K.size(1), c, layer_deriv_K.size(0), 1, {
	    layer_deriv_K_v(c,k,b) += layer_deriv_V_v(c,k,b);
	  });
      }
      TensorType throwaway(std::move(layer_deriv_V));
    }
	
    p = leaf_KV.v.deriv(cost_deriv, p, std::move(layer_deriv_K), input_above_deriv_return);
    return leaf_Q.v.deriv(cost_deriv, p, std::move(layer_deriv_Q), input_above_deriv_return);
  }

  int update(int off, const Vector<FloatType> &new_params){
    int p=off;
    attention.update(p,new_params);
    p+=attention.nparams();
    p = leaf_KV.v.update(p, new_params);
    return leaf_Q.v.update(p,new_params);
  }
  int step(int off, const Vector<FloatType> &derivs, FloatType eps){
    int p=off;
    attention.step(p,derivs,eps);
    p+=attention.nparams();
    p = leaf_KV.v.step(p,derivs,eps);
    return leaf_Q.v.step(p,derivs,eps);
  }


  //accumulated #params for layers here and below
  inline int nparams() const{ return attention.nparams() + leaf_KV.v.nparams() +  leaf_Q.v.nparams() ; }

  size_t FLOPS(int value_or_deriv) const{ return attention.FLOPS(value_or_deriv) + leaf_KV.v.FLOPS(value_or_deriv) + leaf_Q.v.FLOPS(value_or_deriv); }
  
  int getParams(Vector<FloatType> &into, int off) const{
    int p=off;
    attention.getParams(into,p);
    p+=attention.nparams();
    p = leaf_KV.v.getParams(into,p);
    return leaf_Q.v.getParams(into,p);
  }

  inline void resizeInputBuffer(size_t to){
    attention.resizeInputBuffer(to);
    leaf_KV.v.resizeInputBuffer(to);
    leaf_Q.v.resizeInputBuffer(to);
  }
};

#define LAYER_TYPE MultiHeadCrossAttentionLayer<FLOATTYPE(ChainKV), \
						INPUTTYPE(ChainKV), \
						DDST(chain_KV),DDST(chain_Q)>
#define TEMPL \
template<typename ChainKV, typename ChainQ, \
	 typename std::enable_if<ISLEAF(ChainKV) && ISLEAF(ChainQ) && std::is_same<FLOATTYPE(ChainKV),FLOATTYPE(ChainQ)>::value && std::is_same<INPUTTYPE(ChainKV),INPUTTYPE(ChainQ)>::value , int>::type = 0 \
	 >

TEMPL
auto multihead_cross_attention_layer(int Nheads,
				     Matrix<FLOATTYPE(ChainKV)> const* const* W_Q,
				     Matrix<FLOATTYPE(ChainKV)> const* const* W_K,
				     Matrix<FLOATTYPE(ChainKV)> const* const* W_V,
				     const Matrix<FLOATTYPE(ChainKV)> &W_O,				     
				     bool use_mask,
				     ChainKV &&chain_KV, ChainQ &&chain_Q)-> LAYER_TYPE{
  return LAYER_TYPE(std::forward<ChainKV>(chain_KV), std::forward<ChainQ>(chain_Q), Nheads, W_Q, W_K, W_V, W_O, use_mask);
}
TEMPL
auto multihead_cross_attention_layer(int Nheads,
				     Matrix<FLOATTYPE(ChainKV)> const* const* W_Q,
				     Matrix<FLOATTYPE(ChainKV)> const* const* W_K,
				     Matrix<FLOATTYPE(ChainKV)> const* const* W_V,
				     const Matrix<FLOATTYPE(ChainKV)> &W_O,
				     ChainKV &&chain_KV, ChainQ &&chain_Q)-> LAYER_TYPE{
  return LAYER_TYPE(std::forward<ChainKV>(chain_KV), std::forward<ChainQ>(chain_Q), Nheads, W_Q, W_K, W_V, W_O, false);
}



TEMPL
auto multihead_cross_attention_layer(int Nheads,
				     const std::vector<Matrix<FLOATTYPE(ChainKV)> > &W_Q,
				     const std::vector<Matrix<FLOATTYPE(ChainKV)> > &W_K,
				     const std::vector<Matrix<FLOATTYPE(ChainKV)> > &W_V,
				     const Matrix<FLOATTYPE(ChainKV)> &W_O,
				     bool use_mask,
				     ChainKV &&chain_KV, ChainQ &&chain_Q)-> LAYER_TYPE{
  return LAYER_TYPE(std::forward<ChainKV>(chain_KV), std::forward<ChainQ>(chain_Q), Nheads, W_Q, W_K, W_V, W_O, use_mask);
}
TEMPL
auto multihead_cross_attention_layer(int Nheads,
				     const std::vector<Matrix<FLOATTYPE(ChainKV)> > &W_Q,
				     const std::vector<Matrix<FLOATTYPE(ChainKV)> > &W_K,
				     const std::vector<Matrix<FLOATTYPE(ChainKV)> > &W_V,
				     const Matrix<FLOATTYPE(ChainKV)> &W_O,
				     ChainKV &&chain_KV, ChainQ &&chain_Q)-> LAYER_TYPE{
  return LAYER_TYPE(std::forward<ChainKV>(chain_KV), std::forward<ChainQ>(chain_Q), Nheads, W_Q, W_K, W_V, W_O, false);
}


//Default initialization has W_Q,W_K,W_V all of size E/Nheads x E  and W_O of size ExE
//each initialized using Glorot uniform
TEMPL
auto multihead_cross_attention_layer(int Nheads,
				     int E,
				     bool use_mask,
				     ChainKV &&chain_KV, ChainQ &&chain_Q)-> LAYER_TYPE{
  typedef FLOATTYPE(ChainKV) FloatType;
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
  
  auto layer = LAYER_TYPE(std::forward<ChainKV>(chain_KV), std::forward<ChainQ>(chain_Q), Nheads, W_Q, W_K, W_V, W_O, use_mask);  
  return layer;
}

TEMPL
auto multihead_cross_attention_layer(int Nheads,
				     int E,
				     ChainKV &&chain_KV, ChainQ &&chain_Q){
  return multihead_cross_attention_layer(Nheads,E,false,std::forward<ChainKV>(chain_KV),std::forward<ChainQ>(chain_Q));
}

#undef TEMPL
#undef LAYER_TYPE
