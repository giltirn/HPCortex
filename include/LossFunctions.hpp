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
  inline static double loss(const Matrix &y, const Matrix &ypred){
    int dim = y.size(0);
    int batch_size = y.size(1);
    
    double out = 0.;
    for(int i=0;i<dim;i++)
      for(int b=0;b<batch_size;b++)
	out += pow(ypred(i,b) - y(i,b),2);
    out /= (dim * batch_size);
    return out;
  }
  inline static Matrix layer_deriv(const Matrix &y, const Matrix &ypred){
    //for reverse differentiation, we pass down the derivatives with respect to our inputs
    //dout / dparam(i) = \sum_j 2*(ypred(j) - y(j)) * dypred(j)/dparam(i)

    //dout / dypred(i) = 2*(ypred(i) - y(i)) /dim
    int dim = y.size(0);
    int batch_size = y.size(1);
    
    Matrix layer_deriv(dim,batch_size);
    for(int i=0;i<dim;i++)
      for(int b=0;b<batch_size;b++)
	layer_deriv(i,b) = 2*(ypred(i,b) - y(i,b)) / (dim*batch_size);
    return layer_deriv;
  }
};
   
template<typename U, typename std::enable_if<ISLEAF(U), int>::type = 0>
auto mse_cost(U &&u)->CostFuncWrapper<DDST(u), MSEcostFunc>{
  return CostFuncWrapper<DDST(u), MSEcostFunc>(std::forward<U>(u));
}

