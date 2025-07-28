#pragma once
#include<cmath>
#include<Tensors.hpp>
#include<InstanceStorage.hpp>
#include<Layers.hpp>

template<typename Store, typename CostFunc>
class CostFuncWrapper{
public:
  typedef typename Store::type ModelType;
  typedef typename ModelType::FloatType FloatType;
  typedef typename ModelType::InputType InputType;
  typedef typename CostFunc::PredictionType PredictionType; //the type of data output by the model
  static_assert( std::is_same<LAYEROUTPUTTYPE(ModelType),  PredictionType>::value );
  typedef typename CostFunc::ComparisonType ComparisonType; //the type of the data used to compare the predictions against (need not be the same as the PredictionType)
private:  
  Store leaf;
  
  PredictionType ypred; //dim * batch_size
  ComparisonType yval;
  CostFunc cost;
  int nparam;
public:
  CostFuncWrapper(Store &&leaf, const CostFunc &cost = CostFunc()): cost(cost), leaf(std::move(leaf)), nparam(leaf.v.nparams()){}

  /**
   * @brief Access the underlying model
   */
  ModelType &getModel(){ return leaf.v; }
  
  FloatType loss(const InputType &x, const ComparisonType &y, EnableDeriv enable_deriv = DerivNo){
    //std::cout << "Loss with tensor of dim " << x.dimension() << " and sizes " << x.sizeArrayString() << std::endl;
    
    ypred = leaf.v.value(x, enable_deriv);
    yval = y;
    return cost.loss(y,ypred);
  }
  Vector<FloatType> deriv() const{
    auto layer_deriv = cost.layer_deriv(yval, ypred);

    Vector<FloatType> cost_deriv(nparam,0.);    //zero initialize
    leaf.v.deriv(cost_deriv, 0, std::move(layer_deriv));
    return cost_deriv;
  }

  PredictionType predict(const InputType &x){
    return leaf.v.value(x);
  }
  
  template<typename _PredictionType=PredictionType, typename _InputType=InputType, int TensDimIn = _InputType::Dimension, int TensDimOut = _PredictionType::Dimension,
	   typename std::enable_if< std::is_same<_PredictionType,Tensor<FloatType,TensDimOut> >::value &&
				    std::is_same<_InputType,Tensor<FloatType,TensDimIn> >::value &&
				    std::is_same<_PredictionType,ComparisonType>::value, int>::type = 0>
  Tensor<FloatType,TensDimOut-1> predict(const Tensor<FloatType,TensDimIn-1> &x, int batch_size){
    int size_in[TensDimIn]; size_in[TensDimIn-1] = batch_size;
    memcpy(size_in, x.sizeArray(), (TensDimIn-1)*sizeof(int));
    Tensor<FloatType,TensDimIn> bx(size_in);
    bx.pokeLastDimension(x,0);    
    return predict(bx).peekLastDimension(0);
  }   
  
  void update(const Vector<FloatType> &new_params){
    leaf.v.update(0, new_params);
  }
  void step(const Vector<FloatType> &derivs, FloatType eps){
    leaf.v.step(0,derivs,eps);
  }
  int nparams() const{ return nparam; }

  size_t FLOPS(int value_or_deriv) const{ return leaf.v.FLOPS(value_or_deriv); }

  Vector<FloatType> getParams() const{
    Vector<FloatType> out(nparams());
    leaf.v.getParams(out,0);
    return out;
  }
};

template<typename OutputType>
class MSEcostFunc{};

template<typename FloatType, int Dim>
class MSEcostFunc<Tensor<FloatType,Dim> >{
public:
  typedef Tensor<FloatType,Dim> DataType;
  typedef DataType ComparisonType;
  typedef DataType PredictionType;
  
  static FloatType loss(const ComparisonType &y, const PredictionType &ypred);  
  static PredictionType layer_deriv(const ComparisonType &y, const PredictionType &ypred);  
};


#define CWRP CostFuncWrapper<DDST(u), MSEcostFunc<LAYEROUTPUTTYPE(U)> >
template<typename U, typename std::enable_if<ISLEAF(U), int>::type = 0>
auto mse_cost(U &&u)->CWRP{
  return CWRP(std::forward<U>(u));
}
#undef CWRP

template<typename CostFunc, typename U, typename std::enable_if<ISLEAF(U) && std::is_default_constructible<CostFunc>::value  , int>::type = 0>
auto cost_func_wrap(U &&u){
  return CostFuncWrapper<DDST(u), CostFunc>(std::forward<U>(u));
}
template<typename CostFunc, typename U, typename std::enable_if<ISLEAF(U), int>::type = 0>
auto cost_func_wrap(U &&u, const CostFunc &cf){
  return CostFuncWrapper<DDST(u), CostFunc>(std::forward<U>(u),cf);
}



#include "implementation/LossFunctions.tcc"

// #ifndef LOSSFUNC_EXTERN_TEMPLATE_INST
// #define SS extern
// #else
// #define SS
// #endif
// SS template class MSEcostFunc<float>;
// SS template class MSEcostFunc<double>;
// #undef SS
