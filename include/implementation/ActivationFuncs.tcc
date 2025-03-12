template<typename FloatType>
Matrix<FloatType> ReLU<FloatType>::operator()(const Matrix<FloatType> &x) const{
  int dim = x.size(0);
  int batch_size = x.size(1);
  Matrix<FloatType> out(dim,batch_size,1.0);
  autoView(out_v,out,DeviceReadWrite);
  autoView(x_v,x,DeviceRead);
  
  //f(x)_i = max(x_i, 0)
  accelerator_for2d(b,batch_size,i,dim,1,{
      if(x_v(i,b) <= 0.) out_v(i,b) = 0.;
    });
  return out;
}
