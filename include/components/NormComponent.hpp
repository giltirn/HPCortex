#pragma once
#include <cmath>
#include <type_traits>
#include <sstream>
#include <Tensors.hpp>
#include <RingBuffer.hpp>
#include <Linalg.hpp>

//A component implementing the norm operation on a batch tensor (one for which the last dimension is the batch dimension) along an arbitrary dimension other than the batch dimension
template<typename _FloatType, int TensDim>
class NormComponent{
public:
  typedef _FloatType FloatType;
private:
  int norm_dim;
  bool use_affine;
  bool use_bias;

  int nparams_val;
  FloatType epsilon;

  int in_size[TensDim];
  size_t other_dim_vol;
  size_t stride;
  bool setup;

  Vector<FloatType> gamma_beta;
  
  mutable RingBuffer<Tensor<FloatType,TensDim> > out_buf;
  mutable RingBuffer<Matrix<FloatType> > std_buf;
public:
  
  NormComponent(int norm_dim, bool use_affine, bool use_bias, FloatType affine_init, FloatType bias_init, FloatType epsilon = 1e-5): norm_dim(norm_dim), use_affine(use_affine), use_bias(use_bias), 
															      nparams_val((int)use_affine + (int)use_bias), epsilon(epsilon), gamma_beta(2), setup(false){
    assert(norm_dim >=0 && norm_dim < TensDim-1);
    if(!use_affine) affine_init = 1.;
    if(!use_bias) bias_init = 0.;
    {
      autoView(gamma_beta_v,gamma_beta,HostWrite);
      gamma_beta_v(0) = affine_init;
      gamma_beta_v(1) = bias_init;
    }
  }
  NormComponent(const NormComponent &r) = delete;
  NormComponent(NormComponent &&r) = default;
  
  Tensor<FloatType,TensDim> value(const Tensor<FloatType,TensDim> &in);
  
  void deriv(Vector<FloatType> &cost_deriv, int off, Tensor<FloatType,TensDim> &&dcost_by_dOut, Tensor<FloatType,TensDim> &dcost_by_dIn) const;

  void update(int off, const Vector<FloatType> &new_params);
  
  void step(int off, const Vector<FloatType> &derivs, FloatType eps);

  //off measured from *end*, return new off
  void getParams(Vector<FloatType> &into, int off);
  
  inline int nparams() const{ return nparams_val; }

  inline void resizeInputBuffer(size_t to){
    out_buf.resize(to);
    std_buf.resize(to);
  }
};

#include "implementation/NormComponent.tcc"
