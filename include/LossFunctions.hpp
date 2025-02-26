#pragma once
#include<cmath>
#include<Tensors.hpp>
#include<InstanceStorage.hpp>
#include<Layers.hpp>

template<typename Store, typename CostFunc>
class CostFuncWrapper{
  Store leaf;
  Matrix ypred; //dim * batch_size
  Matrix yval;
  CostFunc cost;
  int nparam;
public:
  CostFuncWrapper(Store &&leaf, const CostFunc &cost = CostFunc()): cost(cost), leaf(std::move(leaf)), nparam(leaf.v.nparams()){}
  
  double loss(const Matrix &x, const Matrix &y){
    ypred = leaf.v.value(x);
    int dim = y.size(0);
    int batch_size = y.size(1);
    assert(ypred.size(0) == dim);
    assert(ypred.size(1) == batch_size);
    
    yval = y;
    return cost.loss(y,ypred);
  }
  Vector deriv() const{
    Matrix layer_deriv = cost.layer_deriv(yval, ypred);

    Vector cost_deriv(nparam,0.);    //zero initialize
    leaf.v.deriv(cost_deriv, 0, layer_deriv);
    return cost_deriv;
  }

  Matrix predict(const Matrix &x){
    return leaf.v.value(x);
  }

  Vector predict(const Vector &x){
    Matrix b(x.size(0),1);
    b.pokeColumn(0,x);
    return predict(b).peekColumn(0);    
  }
  
  void update(const Vector &new_params){
    leaf.v.update(0, new_params);
  }
  void step(const Vector &derivs, double eps){
    leaf.v.step(0,derivs,eps);
  }
  int nparams(){ return nparam; }

  Vector getParams(){
    Vector out(nparams());
    leaf.v.getParams(out,0);
    return out;
  }
};

class MSEcostFunc{
public:
  static double loss(const Matrix &y, const Matrix &ypred);  
  static Matrix layer_deriv(const Matrix &y, const Matrix &ypred);  
};
   
template<typename U, typename std::enable_if<ISLEAF(U), int>::type = 0>
auto mse_cost(U &&u)->CostFuncWrapper<DDST(u), MSEcostFunc>{
  return CostFuncWrapper<DDST(u), MSEcostFunc>(std::forward<U>(u));
}

