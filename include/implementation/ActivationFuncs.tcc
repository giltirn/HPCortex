template<typename FloatType>
void ReLU<FloatType>::operator()(Matrix<FloatType> &x, Matrix<FloatType> *deriv) const{
  int dim = x.size(0);
  int batch_size = x.size(1);

  //f(x)_i = max(x_i, 0) = x_i  |  x_i > 0
  //                     = 0    |  x_i <= 0
  if(deriv == nullptr){
    autoView(x_v,x,DeviceReadWrite);

    accelerator_for2d(b,batch_size,i,dim,1,{
	if(x_v(i,b) <= 0.) x_v(i,b) = 0.;
    });
  }else{
    *deriv = Matrix<FloatType>(dim,batch_size);

    autoView(deriv_v, (*deriv), DeviceWrite);
    autoView(x_v,x,DeviceReadWrite);
    accelerator_for2d(b,batch_size,i,dim,1,{
	if(x_v(i,b) <= 0.){
	  x_v(i,b) = 0.;
	  deriv_v(i,b) = 0.;
	}else{
	  deriv_v(i,b) = 1.;
	}
      });
  }
}
