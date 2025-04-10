#pragma once
#include<cmath>
#include<Tensors.hpp>
#include<InstanceStorage.hpp>
#include<Layers.hpp>

template<typename Store, typename CostFunc>
class CostFuncWrapper{
public:
  typedef typename Store::type::FloatType FloatType;
  typedef typename Store::type::InputType InputType;
  typedef LAYEROUTPUTTYPE(typename Store::type) OutputType; 
private:  
  Store leaf;
  
  OutputType ypred; //dim * batch_size
  OutputType yval;
  CostFunc cost;
  int nparam;
public:
  CostFuncWrapper(Store &&leaf, const CostFunc &cost = CostFunc()): cost(cost), leaf(std::move(leaf)), nparam(leaf.v.nparams()){}
  
  FloatType loss(const InputType &x, const OutputType &y){
    ypred = leaf.v.value(x);
    yval = y;
    return cost.loss(y,ypred);
  }
  Vector<FloatType> deriv() const{
    auto layer_deriv = cost.layer_deriv(yval, ypred);

    Vector<FloatType> cost_deriv(nparam,0.);    //zero initialize
    leaf.v.deriv(cost_deriv, 0, std::move(layer_deriv));
    return cost_deriv;
  }

  OutputType predict(const InputType &x){
    return leaf.v.value(x);
  }

  template<typename O=OutputType, typename std::enable_if< std::is_same<O,Matrix<FloatType> >::value && std::is_same<InputType,Matrix<FloatType> >::value, int>::type = 0>
  Vector<FloatType> predict(const Vector<FloatType> &x){
    Matrix<FloatType> b(x.size(0),1);
    pokeColumn(b,0,x);
    return peekColumn(predict(b),0);    
  }
  
  void update(const Vector<FloatType> &new_params){
    leaf.v.update(0, new_params);
  }
  void step(const Vector<FloatType> &derivs, FloatType eps){
    leaf.v.step(0,derivs,eps);
  }
  int nparams(){ return nparam; }

  Vector<FloatType> getParams(){
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
  static FloatType loss(const Tensor<FloatType,Dim> &y, const Tensor<FloatType,Dim> &ypred);  
  static Tensor<FloatType,Dim> layer_deriv(const Tensor<FloatType,Dim> &y, const Tensor<FloatType,Dim> &ypred);  
};


#define CWRP CostFuncWrapper<DDST(u), MSEcostFunc<LAYEROUTPUTTYPE(U)> >
template<typename U, typename std::enable_if<ISLEAF(U), int>::type = 0>
auto mse_cost(U &&u)->CWRP{
  return CWRP(std::forward<U>(u));
}
#undef CWRP

#include "implementation/LossFunctions.tcc"

// #ifndef LOSSFUNC_EXTERN_TEMPLATE_INST
// #define SS extern
// #else
// #define SS
// #endif
// SS template class MSEcostFunc<float>;
// SS template class MSEcostFunc<double>;
// #undef SS
