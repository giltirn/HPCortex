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


template<typename FloatType>
template<int Dim>
void ReLU<FloatType>::operator()(Tensor<FloatType,Dim> &x, Tensor<FloatType,Dim> *deriv) const{
  int const* dims = x.sizeArray();
  int batch_size = dims[x.dimension()-1];
  size_t size_other = 1; for(int i=0; i<x.dimension()-1; i++) size_other *= dims[i];
  
  //f(x)_i = max(x_i, 0) = x_i  |  x_i > 0
  //                     = 0    |  x_i <= 0
  if(deriv == nullptr){
    autoView(x_v,x,DeviceReadWrite);

    accelerator_for2d(b,batch_size,i,size_other, 1,{
	FloatType &xx = *(x_v.data() + b+batch_size*i);	
	if(xx <= 0.) xx = 0.;
    });
  }else{
    *deriv = Tensor<FloatType,Dim>(dims);

    autoView(deriv_v, (*deriv), DeviceWrite);
    autoView(x_v,x,DeviceReadWrite);
    accelerator_for2d(b,batch_size,i,size_other, 1,{
	size_t off = b + batch_size*i;
	FloatType &xx = *(x_v.data() + off);
	FloatType &dd = *(deriv_v.data() + off);
	
	if(xx <= 0.){
	  xx = 0.;
	  dd = 0.;
	}else{
	  dd = 1.;
	}
      });
  }
}
