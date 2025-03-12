template<typename FloatType>
FloatType MSEcostFunc<FloatType>::loss(const Matrix<FloatType> &y, const Matrix<FloatType> &ypred){
  int dim = y.size(0);
  int batch_size = y.size(1);

  autoView(ypred_v,ypred,HostRead);
  autoView(y_v,y,HostRead);
  
  FloatType out = 0.;
  for(int i=0;i<dim;i++)
    for(int b=0;b<batch_size;b++)
      out += pow(ypred_v(i,b) - y_v(i,b),2);
  out /= (dim * batch_size);
  return out;
}

template<typename FloatType>
Matrix<FloatType> MSEcostFunc<FloatType>::layer_deriv(const Matrix<FloatType> &y, const Matrix<FloatType> &ypred){
  //for reverse differentiation, we pass down the derivatives with respect to our inputs
  //dout / dparam(i) = \sum_j 2*(ypred(j) - y(j)) * dypred(j)/dparam(i)

  //dout / dypred(i) = 2*(ypred(i) - y(i)) /dim
  int dim = y.size(0);
  int batch_size = y.size(1);
    
  Matrix<FloatType> layer_deriv_m(dim,batch_size);
  autoView(ypred_v,ypred,DeviceRead);
  autoView(y_v,y,DeviceRead);
  autoView(layer_deriv_v,layer_deriv_m,DeviceWrite);
  
  //Might be optimal to have more than just batch_size elements per block but this is a fair start
  accelerator_for2d(b,batch_size,i,dim,1,{
      layer_deriv_v(i,b) = 2*(ypred_v(i,b) - y_v(i,b)) / (dim*batch_size);
    });  
  
  return layer_deriv_m;
}
