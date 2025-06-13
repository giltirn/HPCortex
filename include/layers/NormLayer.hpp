#pragma once
#include "LayerCommon.hpp"
#include <components/NormComponent.hpp>
#include <components/ScaleComponent.hpp>

template<typename _FloatType, int TensDim, typename _InputType, typename Store>
class NormLayer{
public:
  typedef _FloatType FloatType;
  typedef _InputType InputType;
private:
  typedef Tensor<FloatType,TensDim> LayerInputType;

  NormComponent<FloatType,TensDim> nrm;
  ScaleComponent<FloatType,TensDim> scale;
  Store leaf;

public:
  typedef LeafTag tag;
  
  NormLayer(Store &&leaf,
	    int norm_dim, int norm_dim_size, bool use_affine, bool use_bias,
	    const Vector<FloatType> &affine_init, const Vector<FloatType> &bias_init, FloatType epsilon): scale(norm_dim,norm_dim_size,use_affine,use_bias,affine_init,bias_init), nrm(norm_dim,epsilon), leaf(std::move(leaf)){}

  NormLayer(const NormLayer &r) = delete;
  NormLayer(NormLayer &&r) = default;
  
  //Forward pass
  inline Tensor<FloatType,TensDim> value(const InputType &x){ return scale.value(nrm.value(leaf.v.value(x))); }

  inline int deriv(Vector<FloatType> &cost_deriv, int off, Tensor<FloatType,TensDim> &&_above_deriv, InputType* input_above_deriv_return = nullptr) const{
    Tensor<FloatType,TensDim> layer_deriv_scale;
    scale.deriv(cost_deriv, off, std::move(_above_deriv), layer_deriv_scale);
    Tensor<FloatType,TensDim> layer_deriv;
    nrm.deriv(std::move(layer_deriv_scale), layer_deriv);
    return leaf.v.deriv(cost_deriv, off+scale.nparams(), std::move(layer_deriv), input_above_deriv_return);
  }

  inline int update(int off, const Vector<FloatType> &new_params){
    scale.update(off,new_params); return leaf.v.update(off+scale.nparams(), new_params);
  }    
  
  inline int step(int off, const Vector<FloatType> &derivs, FloatType eps){
    scale.step(off,derivs,eps); return leaf.v.step(off+scale.nparams(), derivs,eps);
  }
  
  //accumulated #params for layers here and below
  inline int nparams() const{ return scale.nparams() + leaf.v.nparams(); }

  size_t FLOPS(int value_or_deriv) const{ return nrm.FLOPS(value_or_deriv) + scale.FLOPS(value_or_deriv) + leaf.v.FLOPS(value_or_deriv); }
  
  //off measured from *end*, return new off
  int getParams(Vector<FloatType> &into, int off){
    scale.getParams(into,off); return leaf.v.getParams(into,off+scale.nparams());
  }

  //For pipelining
  inline void resizeInputBuffer(size_t to){
    scale.resizeInputBuffer(to);
    nrm.resizeInputBuffer(to);
    leaf.v.resizeInputBuffer(to);
  }

};

#define LAYER_TYPE NormLayer<FLOATTYPE(U),TensDim,INPUTTYPE(U),DDST(u)>
template<int TensDim, typename U, typename std::enable_if<ISLEAF(U), int>::type = 0>
auto norm_layer(U &&u,
		int norm_dim, int norm_dim_size,
		bool use_affine, bool use_bias,
		const Vector<FLOATTYPE(U)> &affine_init, const Vector<FLOATTYPE(U)>  &bias_init, FLOATTYPE(U) epsilon = 1e-5)-> LAYER_TYPE{
  return LAYER_TYPE(std::forward<U>(u), norm_dim, norm_dim_size, use_affine, use_bias, affine_init, bias_init, epsilon);
}

//Default initialization affine = {1,1,1,1,...}  bias = {0,0,0,0....}
template<int TensDim, typename U, typename std::enable_if<ISLEAF(U), int>::type = 0>
auto norm_layer(U &&u,
		int norm_dim, int norm_dim_size,
		bool use_affine, bool use_bias,
		FLOATTYPE(U) epsilon = 1e-5)-> LAYER_TYPE{
  return LAYER_TYPE(std::forward<U>(u), norm_dim, norm_dim_size, use_affine, use_bias, Vector<FLOATTYPE(U)>(norm_dim_size,1.), Vector<FLOATTYPE(U)>(norm_dim_size,0.), epsilon);
}

#undef LAYER_TYPE
