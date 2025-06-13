#pragma once
#include <cmath>
#include <type_traits>
#include <sstream>
#include <Tensors.hpp>
#include <RingBuffer.hpp>
#include <Linalg.hpp>

//A component implementing the element-wise scale operation on a batch tensor (one for which the last dimension is the batch dimension) along an arbitrary dimension other than the batch dimension
//e.g. for dim 2:  out_{ijk} = gamma_j in_ijk + bias_j      (no sum)
//gamma_j and beta_j are learnable parameters
template<typename _FloatType, int TensDim>
class ScaleComponent{
public:
  typedef _FloatType FloatType;
private:
  int scale_dim;
  bool use_affine;
  bool use_bias;

  int nparams_val;

  int in_size[TensDim];
  size_t other_dim_vol;
  size_t stride;
  mutable FLOPScounter value_FLOPS;
  mutable FLOPScounter deriv_FLOPS;
  bool setup;

  Vector<FloatType> gamma;
  Vector<FloatType> beta;
  
  mutable RingBuffer<Tensor<FloatType,TensDim> > in_buf;
public:

  //if(use_affine),  affine_init.size(0) must equal dimension_size
  //if(use_bias),  bias_init.size(0) must equal dimension_size
  //the initializer inputs are ignored for those parameter vectors disabled by use_affine, use_bias
  ScaleComponent(int scale_dim, int dimension_size,
		bool use_affine, bool use_bias,
		const Vector<FloatType> &affine_init, const Vector<FloatType> &bias_init): scale_dim(scale_dim), use_affine(use_affine), use_bias(use_bias), 
											   gamma(affine_init), beta(bias_init), nparams_val( (int(use_bias)+int(use_affine))* dimension_size ), setup(false){
    assert(scale_dim >=0 && scale_dim < TensDim-1);
    if(!use_affine) gamma = Vector<FloatType>(dimension_size,1.0);
    if(!use_bias) beta = Vector<FloatType>(dimension_size,0.0);
    assert(gamma.size(0) == dimension_size && beta.size(0) == dimension_size);
  }
  ScaleComponent(const ScaleComponent &r) = delete;
  ScaleComponent(ScaleComponent &&r) = default;
  
  Tensor<FloatType,TensDim> value(const Tensor<FloatType,TensDim> &in);
  
  void deriv(Vector<FloatType> &cost_deriv, int off, Tensor<FloatType,TensDim> &&dcost_by_dOut, Tensor<FloatType,TensDim> &dcost_by_dIn) const;

  void update(int off, const Vector<FloatType> &new_params);
  
  void step(int off, const Vector<FloatType> &derivs, FloatType eps);

  //off measured from *end*, return new off
  void getParams(Vector<FloatType> &into, int off);
  
  size_t FLOPS(int value_or_deriv) const{ return value_or_deriv == 0 ? value_FLOPS.value() : deriv_FLOPS.value(); }

  inline int nparams() const{ return nparams_val; }

  inline void resizeInputBuffer(size_t to){
    in_buf.resize(to);
  }
};

#include "implementation/ScaleComponent.tcc"
