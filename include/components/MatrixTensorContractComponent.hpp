#pragma once
#include <type_traits>
#include <sstream>
#include <Tensors.hpp>
#include <RingBuffer.hpp>
#include <Linalg.hpp>

//A component implementing    W_{ij} X_{..., j, b}   where W is a weight matrix and X is a tensor of at least dimension 2. The last dimension is always assumed to be the batch dimension
template<typename _FloatType, int TensDim>
class MatrixTensorContractComponent{
public:
  typedef _FloatType FloatType;
private:
  Matrix<FloatType> weights;
  int size0;
  int size1;
  int batch_size;
  
  int in_dims[TensDim];
  int out_dims[TensDim];
  bool setup;

  //Storage from last call to "value"
  //Buffer size > 1 depending on rank if doing pipelining
  mutable RingBuffer<Tensor<FloatType,TensDim> > in_buf;
public:

  
  MatrixTensorContractComponent(const Matrix<FloatType> &weights):
    weights(weights),
    size0(weights.size(0)), size1(weights.size(1)),
    batch_size(0), setup(false)
  {  }
  MatrixTensorContractComponent(const MatrixTensorContractComponent &r) = delete;
  MatrixTensorContractComponent(MatrixTensorContractComponent &&r) = default;
  
  //Forward pass
  Tensor<FloatType,TensDim> value(const Tensor<FloatType,TensDim> &x);

  void deriv(Vector<FloatType> &cost_deriv, int off, Tensor<FloatType,TensDim> &&dCost_by_dOut, Tensor<FloatType,TensDim> &dCost_by_dIn) const;

  void update(int off, const Vector<FloatType> &new_params);
  
  void step(int off, const Vector<FloatType> &derivs, FloatType eps);

  //accumulated #params for layers here and below
  inline int nparams() const{ return size0*size1; }

  //off measured from *end*, return new off
  void getParams(Vector<FloatType> &into, int off);

  //For pipelining
  inline void resizeInputBuffer(size_t to){
    in_buf.resize(to);
  }

};

#include "implementation/MatrixTensorContractComponent.tcc"
