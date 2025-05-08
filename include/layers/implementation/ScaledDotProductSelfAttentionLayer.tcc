template<typename FloatType, typename InputType, typename Store>
Tensor<FloatType,3> ScaledDotProductSelfAttentionLayer<FloatType,InputType,Store>::value(const InputType &x){
  Tensor<FloatType,3> X = leaf.v.value(x);
  assert(X.size(1) == E);
  if(!setup){
    C = X.size(0);
    B = X.size(2);
    setup = true;
  }else assert(X.size(0) == C && X.size(2) == B);

  return attentionQKV.value(X,X,X);
}

template<typename FloatType, typename InputType, typename Store>
void ScaledDotProductSelfAttentionLayer<FloatType,InputType,Store>::deriv(Vector<FloatType> &cost_deriv, int off, Tensor<FloatType,3> &&_above_deriv, InputType* input_above_deriv_return) const{
  Tensor<FloatType,3> layer_deriv(C,E,B);
  {
    Tensor<FloatType,3> above_deriv_Out = std::move(_above_deriv);
    assert(above_deriv_Out.size(0) == C && above_deriv_Out.size(1) == d_v && above_deriv_Out.size(2) == B);
    
    Tensor<FloatType,3> layer_deriv_Q, layer_deriv_K, layer_deriv_V; //these are  dCost/dX_{ceb} through the different routes
    attentionQKV.deriv(cost_deriv, off, std::move(above_deriv_Out), layer_deriv_Q, layer_deriv_K, layer_deriv_V);

    //Sum dCost/dX_{ceb} through the different routes
    {
      autoView(layer_deriv_K_v,layer_deriv_K,DeviceRead);
      autoView(layer_deriv_Q_v,layer_deriv_Q,DeviceRead);
      autoView(layer_deriv_V_v,layer_deriv_V,DeviceRead);
      autoView(layer_deriv_v,layer_deriv,DeviceWrite);
      accelerator_for3d(b,B,e,E,c,C, 1, {
	  layer_deriv_v(c,e,b) = layer_deriv_K_v(c,e,b) + layer_deriv_Q_v(c,e,b) + layer_deriv_V_v(c,e,b);
	});
    }
  }
  leaf.v.deriv(cost_deriv, off + attentionQKV.nparams(), std::move(layer_deriv),  input_above_deriv_return);
}

template<typename FloatType, typename InputType, typename Store>
void ScaledDotProductSelfAttentionLayer<FloatType,InputType,Store>::update(int off, const Vector<FloatType> &new_params){
  attentionQKV.update(off,new_params);
  leaf.v.update(off + attentionQKV.nparams(),new_params);
}
  
template<typename FloatType, typename InputType, typename Store>
void ScaledDotProductSelfAttentionLayer<FloatType,InputType,Store>::step(int off, const Vector<FloatType> &derivs, FloatType eps){
  attentionQKV.step(off,derivs,eps);
  leaf.v.step(off + attentionQKV.nparams(),derivs,eps);
}

//off measured from *end*, return new off
template<typename FloatType, typename InputType, typename Store>
void ScaledDotProductSelfAttentionLayer<FloatType,InputType,Store>::getParams(Vector<FloatType> &into, int off){
  attentionQKV.getParams(into,off);
  leaf.v.getParams(into,off + attentionQKV.nparams());
}
