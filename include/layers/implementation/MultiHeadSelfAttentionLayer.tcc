template<typename Config, typename InputType, typename Store>
MultiHeadSelfAttentionLayer<Config,InputType,Store>::MultiHeadSelfAttentionLayer(Store &&leaf, int Nheads, Matrix<FloatType> const* const* W_Q, Matrix<FloatType> const* const* W_K, Matrix<FloatType> const* const* W_V, const Matrix<FloatType> &W_O, bool use_mask):
  leaf(std::move(leaf)),
  mha(Nheads,W_Q,W_K,W_V,W_O,use_mask){}

template<typename Config, typename InputType, typename Store>
MultiHeadSelfAttentionLayer<Config,InputType,Store>::MultiHeadSelfAttentionLayer(Store &&leaf, int Nheads, const std::vector<Matrix<FloatType> > &W_Q, const std::vector<Matrix<FloatType> > &W_K, const std::vector<Matrix<FloatType> > &W_V, const Matrix<FloatType> &W_O, bool use_mask):
  leaf(std::move(leaf)),
  mha(Nheads,W_Q,W_K,W_V,W_O,use_mask){}


template<typename Config, typename InputType, typename Store>
Tensor<typename Config::FloatType,3> MultiHeadSelfAttentionLayer<Config,InputType,Store>::value(const InputType &x){
  Tensor<FloatType,3> X = leaf.v.value(x);
  return mha.value(X,X,X);
}
template<typename Config, typename InputType, typename Store>
int MultiHeadSelfAttentionLayer<Config,InputType,Store>::deriv(Vector<FloatType> &cost_deriv, int off, Tensor<FloatType,3> &&_above_deriv, InputType* input_above_deriv_return) const{
  Tensor<FloatType,3> dCost_by_dX;
  {
    Tensor<FloatType,3> dCost_by_dX_Q, dCost_by_dX_K, dCost_by_dX_V;
    mha.deriv(cost_deriv,off,std::move(_above_deriv),dCost_by_dX_Q, dCost_by_dX_K, dCost_by_dX_V);

    int C = dCost_by_dX_Q.size(0);
    int E = dCost_by_dX_Q.size(1);
    int B = dCost_by_dX_Q.size(2);
    assert(dCost_by_dX_K.size(0) == C && dCost_by_dX_K.size(1) == E && dCost_by_dX_K.size(2) == B);
    assert(dCost_by_dX_V.size(0) == C && dCost_by_dX_V.size(1) == E && dCost_by_dX_V.size(2) == B);
       
    dCost_by_dX = Tensor<FloatType,3>(C,E,B);
    {
      autoView(dCost_by_dX_v, dCost_by_dX, DeviceWrite);
      autoView(dCost_by_dX_Q_v, dCost_by_dX_Q, DeviceRead);
      autoView(dCost_by_dX_K_v, dCost_by_dX_K, DeviceRead);
      autoView(dCost_by_dX_V_v, dCost_by_dX_V, DeviceRead);
      
      accelerator_for3d(b,B,e,E,c,C, 1, {
	  dCost_by_dX_v(c,e,b) = dCost_by_dX_Q_v(c,e,b) + dCost_by_dX_K_v(c,e,b) + dCost_by_dX_V_v(c,e,b);
	});      
    }
  }
  return leaf.v.deriv(cost_deriv, off+mha.nparams(), std::move(dCost_by_dX), input_above_deriv_return);
}

template<typename Config, typename InputType, typename Store>
int MultiHeadSelfAttentionLayer<Config,InputType,Store>::update(int off, const Vector<FloatType> &new_params){
  mha.update(off,new_params);
  return leaf.v.update(off + mha.nparams(),new_params);
}

template<typename Config, typename InputType, typename Store>
int MultiHeadSelfAttentionLayer<Config,InputType,Store>::step(int off, const Vector<FloatType> &derivs, FloatType eps){
  mha.step(off,derivs,eps);
  return leaf.v.step(off+mha.nparams(),derivs,eps);
}
template<typename Config, typename InputType, typename Store>
int MultiHeadSelfAttentionLayer<Config,InputType,Store>::getParams(Vector<FloatType> &into, int off) const{
  mha.getParams(into,off);
  return leaf.v.getParams(into,off+mha.nparams());
}

template<typename Config, typename InputType, typename Store>
void MultiHeadSelfAttentionLayer<Config,InputType,Store>::resizeInputBuffer(size_t to){
  mha.resizeInputBuffer(to);
  leaf.v.resizeInputBuffer(to);
}
