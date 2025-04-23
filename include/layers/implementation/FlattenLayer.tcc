template<typename FloatType, typename InputType, typename Store>
Matrix<FloatType> FlattenLayer<FloatType,InputType,Store>::value(const InputType &x){
  LayerInputTensorType in = leaf.v.value(x);
  if(!init){
    memcpy(_input_tens_size, in.sizeArray(), in.dimension() * sizeof(int));
    init = true;
  }
  
  constexpr int tens_dim = in.dimension();
  int batch_size = in.size(tens_dim-1);
  int out_size = 1;
  for(int i=0;i<tens_dim-1;i++)
    out_size *= in.size(i);

  Matrix<FloatType> out(out_size,batch_size);

  //std::cout << "FLATTEN input tensor of dim " << tens_dim << " of size " << in.sizeArrayString() << " to matrix of size " << out_size << " " << batch_size << std::endl;
  
  autoView(out_v,out,DeviceWrite);
  autoView(in_v,in,DeviceRead);
  accelerator_for2d(b,batch_size, i,out_size, 1,{
      //rely on the fact that the batch index is the fastest moving,  eg. for a 3 tensor   off = b + batch_size*(z + zsize*(y + ysize*x))      i=(z + zsize*(y + ysize*x)) 
      out_v(i,b) = in_v.data()[b + i*batch_size];
    });
  return out;
}
template<typename FloatType, typename InputType, typename Store>
void FlattenLayer<FloatType,InputType,Store>::deriv(Vector<FloatType> &cost_deriv, int off, Matrix<FloatType> &&_above_deriv, InputType* input_above_deriv_return) const{
  LayerInputTensorType above_deriv_passdown(_input_tens_size);
  {
    Matrix<FloatType> above_deriv_in(std::move(_above_deriv)); //dcost/dvalue_i
    int batch_size = above_deriv_in.size(1);
    size_t flat_size = above_deriv_in.size(0);

    autoView(above_deriv_passdown_v,above_deriv_passdown,DeviceWrite);
    autoView(above_deriv_in_v,above_deriv_in,DeviceRead);
    accelerator_for2d(b,batch_size, i,flat_size, 1,{
	//rely on the fact that the batch index is the fastest moving
	above_deriv_passdown_v.data()[b + i*batch_size] = above_deriv_in_v(i,b);
      });
  }
  leaf.v.deriv(cost_deriv,off, std::move(above_deriv_passdown),input_above_deriv_return);
}
template<typename FloatType, typename InputType, typename Store>
void FlattenLayer<FloatType,InputType,Store>::update(int off, const Vector<FloatType> &new_params){
  leaf.v.update(off,new_params);
}
template<typename FloatType, typename InputType, typename Store>   
void FlattenLayer<FloatType,InputType,Store>::step(int off, const Vector<FloatType> &derivs, FloatType eps){
  leaf.v.step(off,derivs,eps);
}
template<typename FloatType, typename InputType, typename Store>  
void FlattenLayer<FloatType,InputType,Store>::getParams(Vector<FloatType> &into, int off){
  leaf.v.getParams(into,off);
}
