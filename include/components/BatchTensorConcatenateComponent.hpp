#pragma once
#include <type_traits>
#include <sstream>
#include <array>
#include <ModelConfig.hpp>
#include <Tensors.hpp>
#include <Linalg.hpp>

//Concatenate Ntens tensors along a dimension concat_dim < Dim-1  (last dim is the batch index)
template<typename Config, int TensDim>
class BatchTensorConcatenateComponent{
public:
  EXTRACT_CONFIG_TYPES;
private:
  int concat_dim;
  int Ntens;
  std::vector< std::array<int,TensDim> > tens_dims;
  bool setup;
public:
  
  BatchTensorConcatenateComponent(int concat_dim, int Ntens): concat_dim(concat_dim), Ntens(Ntens), tens_dims(Ntens), setup(false){}
  BatchTensorConcatenateComponent(const BatchTensorConcatenateComponent &r) = delete;
  BatchTensorConcatenateComponent(BatchTensorConcatenateComponent &&r) = default;
  
  //Forward pass
  Tensor<FloatType,TensDim> value(Tensor<FloatType,TensDim> const* const* in){
    if(!setup){
      for(int i=0;i<Ntens;i++)
	for(int d=0;d<TensDim;d++)
	  tens_dims[i][d] = in[i]->size(d);
      setup = true;
    }
    return batchTensorConcatenate(in, Ntens, concat_dim);
  }
  //The output tensors array should point to tensor instances, but don't have to be appropriately sized
  void deriv(Tensor<FloatType,TensDim> &&_dcost_by_dOut, Tensor<FloatType,TensDim>* const* dcost_by_dIn) const{
    Tensor<FloatType,TensDim> dcost_by_dOut = std::move(_dcost_by_dOut);
    for(int t=0;t<Ntens;t++)
      *dcost_by_dIn[t] = Tensor<FloatType,TensDim>(tens_dims[t].data());
    batchTensorSplit(dcost_by_dIn, Ntens, dcost_by_dOut, concat_dim);
  }
  
  size_t FLOPS(int value_or_deriv) const{ return 0; }
  
  inline int nparams() const{ return 0; }
};
