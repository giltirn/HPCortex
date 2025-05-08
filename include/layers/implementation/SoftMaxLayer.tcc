template<typename FloatType, typename InputType, typename Store >
Matrix<FloatType> SoftMaxLayer<FloatType,InputType,Store>::value(const InputType &x){	
  Matrix<FloatType> in = leaf.v.value(x);
  return cpt.value(in);
}

template<typename FloatType, typename InputType, typename Store >
void SoftMaxLayer<FloatType,InputType,Store>::deriv(Vector<FloatType> &cost_deriv, int off, Matrix<FloatType> &&above_deriv_, InputType* input_above_deriv_return) const{
  Matrix<FloatType> layer_deriv;
  cpt.deriv(std::move(above_deriv_), layer_deriv);
  leaf.v.deriv(cost_deriv,off,std::move(layer_deriv),input_above_deriv_return);        
}
