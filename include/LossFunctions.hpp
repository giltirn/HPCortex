#pragma once
#include<cmath>
#include<Tensors.hpp>
#include<InstanceStorage.hpp>
#include<Layers.hpp>

template<typename FloatType, typename Store, typename CostFunc>
class CostFuncWrapper{
  Store leaf;
  Matrix<FloatType> ypred; //dim * batch_size
  Matrix<FloatType> yval;
  CostFunc cost;
  int nparam;
public:
  CostFuncWrapper(Store &&leaf, const CostFunc &cost = CostFunc()): cost(cost), leaf(std::move(leaf)), nparam(leaf.v.nparams()){}
  
  FloatType loss(const Matrix<FloatType> &x, const Matrix<FloatType> &y){
    ypred = leaf.v.value(x);
    int dim = y.size(0);
    int batch_size = y.size(1);
    assert(ypred.size(0) == dim);
    assert(ypred.size(1) == batch_size);
    
    yval = y;
    return cost.loss(y,ypred);
  }
  Vector<FloatType> deriv() const{
    Matrix<FloatType> layer_deriv = cost.layer_deriv(yval, ypred);

    Vector<FloatType> cost_deriv(nparam,0.);    //zero initialize
    leaf.v.deriv(cost_deriv, 0, layer_deriv);
    return cost_deriv;
  }

  Matrix<FloatType> predict(const Matrix<FloatType> &x){
    return leaf.v.value(x);
  }

  Vector<FloatType> predict(const Vector<FloatType> &x){
    Matrix<FloatType> b(x.size(0),1);
    b.pokeColumn(0,x);
    return predict(b).peekColumn(0);    
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

template<typename FloatType>
class MSEcostFunc{
public:
  static FloatType loss(const Matrix<FloatType> &y, const Matrix<FloatType> &ypred);  
  static Matrix<FloatType> layer_deriv(const Matrix<FloatType> &y, const Matrix<FloatType> &ypred);  
};
   
template<typename U, typename std::enable_if<ISLEAF(U), int>::type = 0>
auto mse_cost(U &&u)->CostFuncWrapper<FLOATTYPE(U),DDST(u), MSEcostFunc<FLOATTYPE(U)> >{
  return CostFuncWrapper<FLOATTYPE(U),DDST(u), MSEcostFunc<FLOATTYPE(U)> >(std::forward<U>(u));
}

#include "implementation/LossFunctions.tcc"
