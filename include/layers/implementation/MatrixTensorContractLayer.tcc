template<typename FloatType, int TensDim, typename InputType, typename Store>
Tensor<FloatType,TensDim> MatrixTensorContractLayer<FloatType,TensDim,InputType,Store>::value(const InputType &x){
  return cpt.value(leaf.v.value(x));
}

template<typename FloatType, int TensDim, typename InputType, typename Store>
int MatrixTensorContractLayer<FloatType,TensDim,InputType,Store>::deriv(Vector<FloatType> &cost_deriv, int off, Tensor<FloatType,TensDim> &&_above_deriv, InputType* input_above_deriv_return) const{
  Tensor<FloatType,TensDim> layer_deriv;
  cpt.deriv(cost_deriv,off,std::move(_above_deriv), layer_deriv);
  return leaf.v.deriv(cost_deriv, off+cpt.nparams(), std::move(layer_deriv), input_above_deriv_return);
}

template<typename FloatType, int TensDim, typename InputType, typename Store>
int MatrixTensorContractLayer<FloatType,TensDim,InputType,Store>::update(int off, const Vector<FloatType> &new_params){
  cpt.update(off,new_params);
  return leaf.v.update(off+cpt.nparams(),new_params);
}

template<typename FloatType, int TensDim, typename InputType, typename Store>
int MatrixTensorContractLayer<FloatType,TensDim,InputType,Store>::step(int off, const Vector<FloatType> &derivs, FloatType eps){
  cpt.step(off,derivs,eps);
  return leaf.v.step(off+cpt.nparams(),derivs,eps);
}



//off measured from *end*, return new off
template<typename FloatType, int TensDim, typename InputType, typename Store>
int MatrixTensorContractLayer<FloatType,TensDim,InputType,Store>::getParams(Vector<FloatType> &into, int off) const{
  cpt.getParams(into,off);
  return leaf.v.getParams(into,off+cpt.nparams());
}
