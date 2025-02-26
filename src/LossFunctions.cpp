#include <LossFunctions.hpp>

double MSEcostFunc::loss(const Matrix &y, const Matrix &ypred){
  int dim = y.size(0);
  int batch_size = y.size(1);
    
  double out = 0.;
  for(int i=0;i<dim;i++)
    for(int b=0;b<batch_size;b++)
      out += pow(ypred(i,b) - y(i,b),2);
  out /= (dim * batch_size);
  return out;
}

Matrix MSEcostFunc::layer_deriv(const Matrix &y, const Matrix &ypred){
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
