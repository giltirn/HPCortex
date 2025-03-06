template<typename FloatType>
Matrix<FloatType> ReLU<FloatType>::operator()(const Matrix<FloatType> &x) const{
  int dim = x.size(0);
  int batch_size = x.size(1);
  Matrix<FloatType> out(dim,batch_size,1.0);
  autoView(out_v,out,HostReadWrite);
  autoView(x_v,x,HostRead);
  
  //f(x)_i = max(x_i, 0)
  for(int i=0;i<dim;i++)
    for(int b=0;b<batch_size;b++)
      if(x_v(i,b) <= 0.) out_v(i,b) = 0.;
  return out;
}
