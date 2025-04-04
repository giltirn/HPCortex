template<typename FloatType, int Dim>
FloatType MSEcostFunc<Tensor<FloatType,Dim> >::loss(const Tensor<FloatType,Dim> &y, const Tensor<FloatType,Dim> &ypred){
  int const* dims = y.sizeArray();
  size_t other_sz = 1;
  for(int i=0;i<y.dimension()-1;i++)
    other_sz *= dims[i];
  
  int batch_size = dims[y.dimension()-1];

#ifdef USE_CUDA
  autoView(ypred_v,ypred,DeviceRead);
  autoView(y_v,y,DeviceRead);
  FloatType *out_d = (FloatType*)acceleratorAllocDevice(sizeof(FloatType));
  acceleratorMemSet(out_d,0,sizeof(FloatType));
  
  accelerator_for2d_shm(b,batch_size,i,other_sz, 1,(batch_size*sizeof(FloatType)),{
      extern __shared__ FloatType shared[];
      size_t off = b + batch_size*i;
      FloatType yp_val = *(ypred_v.data() + off);
      FloatType y_val = *(y_v.data() + off);
      
      shared[b] = pow(yp_val - y_val,2);
      acceleratorSynchronizeBlock();
      if(!b){
       	FloatType sum = shared[0];
	for(int i=1;i<batch_size;i++)
       	  sum += shared[i];	
       	atomicAdd(out_d, sum);
      }
    });

  FloatType out;
  acceleratorCopyFromDevice(&out, out_d, sizeof(FloatType));
  acceleratorFreeDevice(out_d);
  return out / (other_sz * batch_size);
#else
  autoView(ypred_v,ypred,HostRead);
  autoView(y_v,y,HostRead);
  
  FloatType out = 0.;
  for(size_t i=0;i<other_sz;i++)
    for(int b=0;b<batch_size;b++){
      size_t off = b + batch_size*i;
      FloatType yp_val = *(ypred_v.data() + off);
      FloatType y_val = *(y_v.data() + off);
      out += pow(yp_val-y_val,2);
    }
  out /= (other_sz * batch_size);
  return out;
#endif
}

template<typename FloatType,int Dim>
Tensor<FloatType,Dim> MSEcostFunc<Tensor<FloatType,Dim> >::layer_deriv(const Tensor<FloatType,Dim> &y, const Tensor<FloatType,Dim> &ypred){
  //for reverse differentiation, we pass down the derivatives with respect to our inputs
  //dout / dparam(i) = \sum_j 2*(ypred(j) - y(j)) * dypred(j)/dparam(i)

  //dout / dypred(i) = 2*(ypred(i) - y(i)) /dim
  int dim = y.size(0);
  int const* dims = y.sizeArray();
  size_t other_sz = 1;
  for(int i=0;i<y.dimension()-1;i++)
    other_sz *= dims[i];
  
  int batch_size = dims[y.dimension()-1];
    
  Tensor<FloatType,Dim> layer_deriv_m(dim,batch_size);
  autoView(ypred_v,ypred,DeviceRead);
  autoView(y_v,y,DeviceRead);
  autoView(layer_deriv_v,layer_deriv_m,DeviceWrite);
  
  //Might be optimal to have more than just batch_size elements per block but this is a fair start
  accelerator_for2d(b,batch_size,i,other_sz, 1,{
      size_t off = b + batch_size*i;
      layer_deriv_v(i,b) = 2*(   *(ypred_v.data()+off) - *(y_v.data()+off)  ) / (other_sz*batch_size);
    });  
  
  return layer_deriv_m;
}
