#pragma once
#include "LayerCommon.hpp"
#include <components/NormComponent.hpp>

template<typename _FloatType, int TensDim, typename _InputType, typename Store>
class NormLayer{
public:
  typedef _FloatType FloatType;
  typedef _InputType InputType;
private:
  typedef Tensor<FloatType,TensDim> LayerInputType;

  NormComponent<FloatType,TensDim> nrm;
  Store leaf;

public:
  typedef LeafTag tag;
  
  NormLayer(Store &&leaf,
	    int norm_dim, bool use_affine, bool use_bias, FloatType affine_init, FloatType bias_init, FloatType epsilon): nrm(norm_dim,use_affine,use_bias,affine_init,bias_init,epsilon), leaf(std::move(leaf)){}

  NormLayer(const NormLayer &r) = delete;
  NormLayer(NormLayer &&r) = default;
  
  //Forward pass
  inline Tensor<FloatType,TensDim> value(const InputType &x){ return nrm.value(leaf.v.value(x)); }

  inline void deriv(Vector<FloatType> &cost_deriv, int off, Tensor<FloatType,TensDim> &&_above_deriv, InputType* input_above_deriv_return = nullptr) const{
    Tensor<FloatType,TensDim> layer_deriv; nrm.deriv(cost_deriv, off, std::move(_above_deriv), layer_deriv);
    leaf.v.deriv(cost_deriv, off+nrm.nparams(), std::move(layer_deriv), input_above_deriv_return);
  }

  inline void update(int off, const Vector<FloatType> &new_params){
    nrm.update(off,new_params); leaf.v.update(off+nrm.nparams(), new_params);
  }    
  
  inline void step(int off, const Vector<FloatType> &derivs, FloatType eps){
    nrm.step(off,derivs,eps); leaf.v.step(off+nrm.nparams(), derivs,eps);
  }
  
  //accumulated #params for layers here and below
  inline int nparams() const{ return nrm.nparams() + leaf.v.nparams(); }

  //off measured from *end*, return new off
  void getParams(Vector<FloatType> &into, int off){
    nrm.getParams(into,off); leaf.v.getParams(into,off+nrm.nparams());
  }

  //For pipelining
  inline void resizeInputBuffer(size_t to){
    nrm.resizeInputBuffer(to);
    leaf.v.resizeInputBuffer(to);
  }

};

#define LAYER_TYPE NormLayer<FLOATTYPE(U),TensDim,INPUTTYPE(U),DDST(u)>
template<int TensDim, typename U, typename std::enable_if<ISLEAF(U), int>::type = 0>
auto norm_layer(U &&u,
		int norm_dim,
		bool use_affine, bool use_bias,
		FLOATTYPE(U) affine_init, FLOATTYPE(U) bias_init, FLOATTYPE(U) epsilon = 1e-5)-> LAYER_TYPE{
  return LAYER_TYPE(std::forward<U>(u), norm_dim, use_affine, use_bias, affine_init, bias_init, epsilon);
}
#undef LAYER_TYPE
