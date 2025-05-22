template<typename FloatType, int TensDim, typename InputType, typename Store, typename ActivationFunc>
Tensor<FloatType,TensDim> BatchTensorDNNlayer<FloatType,TensDim,InputType,Store,ActivationFunc>::value(const InputType &x){
  return cpt.value(leaf.v.value(x));
}

template<typename FloatType, int TensDim, typename InputType, typename Store, typename ActivationFunc>
int BatchTensorDNNlayer<FloatType,TensDim,InputType,Store,ActivationFunc>::deriv(Vector<FloatType> &cost_deriv, int off, Tensor<FloatType,TensDim> &&_above_deriv, InputType* input_above_deriv_return) const{
  Tensor<FloatType,TensDim> layer_deriv;
  cpt.deriv(cost_deriv,off,std::move(_above_deriv), layer_deriv);
  return leaf.v.deriv(cost_deriv, off+cpt.nparams(), std::move(layer_deriv), input_above_deriv_return);
}

template<typename FloatType, int TensDim, typename InputType, typename Store, typename ActivationFunc>
int BatchTensorDNNlayer<FloatType,TensDim,InputType,Store,ActivationFunc>::update(int off, const Vector<FloatType> &new_params){
  cpt.update(off,new_params);
  return leaf.v.update(off+cpt.nparams(),new_params);
}

template<typename FloatType, int TensDim, typename InputType, typename Store, typename ActivationFunc>
int BatchTensorDNNlayer<FloatType,TensDim,InputType,Store,ActivationFunc>::step(int off, const Vector<FloatType> &derivs, FloatType eps){
  cpt.step(off,derivs,eps);
  return leaf.v.step(off+cpt.nparams(),derivs,eps);
}

template<typename FloatType, int TensDim, typename InputType, typename Store, typename ActivationFunc>
int BatchTensorDNNlayer<FloatType,TensDim,InputType,Store,ActivationFunc>::getParams(Vector<FloatType> &into, int off){
  cpt.getParams(into,off);
  return leaf.v.getParams(into,off+cpt.nparams());
}
