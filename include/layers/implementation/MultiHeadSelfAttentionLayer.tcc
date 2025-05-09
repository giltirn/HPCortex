template<typename FloatType, typename InputType, typename Store>
MultiHeadSelfAttentionLayer<FloatType,InputType,Store>::MultiHeadSelfAttentionLayer(Store &&leaf, int Nheads, Matrix<FloatType> const* const* W_Q, Matrix<FloatType> const* const* W_K, Matrix<FloatType> const* const* W_V, const Matrix<FloatType> &W_O):
  leaf(std::move(leaf)),
  concatY(1,Nheads),
  multW_O(W_O),
  heads(Nheads),
  setup(false)
{
  assert(Nheads > 0);
  //ensure that d_v all sum to the right size
  int sdv=0;
  for(int h=0;h<Nheads;h++)
    sdv += W_V[h]->size(0);
  assert(sdv == W_O.size(1));

  E = W_K[0]->size(1);
  for(int h=1;h<Nheads;h++)
    assert(W_K[h]->size(1) == E);

  Nparams_layer = multW_O.nparams();
  for(int h=0;h<Nheads;h++){
    heads[h].reset(new ScaledDotProductAttentionHeadComponent<FloatType>(*W_Q[h],*W_K[h],*W_V[h]));
    Nparams_layer += heads[h]->nparams();
  }

}

template<typename FloatType, typename InputType, typename Store>
Tensor<FloatType,3> MultiHeadSelfAttentionLayer<FloatType,InputType,Store>::value(const InputType &x){ 
  Tensor<FloatType,3> X = leaf.v.value(x);
  assert(X.size(1)==E);
  if(!setup){
    C=X.size(0);
    B=X.size(2);
    setup = true;
  }else assert(X.size(0)==C && X.size(2)==B);
  
  std::vector< Tensor<FloatType,3> > Y(heads.size());
  std::vector< Tensor<FloatType,3> const* > Yp(heads.size());
  for(int h=0;h<heads.size();h++){  
    Y[h] = heads[h]->value(X,X,X);
    Yp[h] = &Y[h];
  }
  return multW_O.value( concatY.value(Yp.data()) );
}
template<typename FloatType, typename InputType, typename Store>
void MultiHeadSelfAttentionLayer<FloatType,InputType,Store>::deriv(Vector<FloatType> &cost_deriv, int off, Tensor<FloatType,3> &&_above_deriv, InputType* input_above_deriv_return) const{
  int p = off;
  Tensor<FloatType,3> dcost_by_dX;
  {
    Tensor<FloatType,3> dCost_by_dO = std::move(_above_deriv);
    Tensor<FloatType,3> dCost_by_dYconcat;
    multW_O.deriv(cost_deriv, p, std::move(dCost_by_dO), dCost_by_dYconcat);
    p += multW_O.nparams();
    
    std::vector< Tensor<FloatType,3> > dcost_by_dY(heads.size());
    std::vector< Tensor<FloatType,3>* > dcost_by_dYp(heads.size());
    for(int h=0;h<heads.size();h++) dcost_by_dYp[h] = &dcost_by_dY[h];
    concatY.deriv(std::move(dCost_by_dYconcat), dcost_by_dYp.data());

    dcost_by_dX = Tensor<FloatType,3>(C,E,B, 0.);
    for(int h=0;h<heads.size();h++){
      Tensor<FloatType,3> dcost_by_dX_Q, dcost_by_dX_K, dcost_by_dX_V;
      heads[h]->deriv(cost_deriv, p, std::move(dcost_by_dY[h]), dcost_by_dX_Q, dcost_by_dX_K, dcost_by_dX_V);
      p += heads[h]->nparams();

      autoView(dcost_by_dX_v, dcost_by_dX, DeviceReadWrite);
      autoView(dcost_by_dX_Q_v, dcost_by_dX_Q, DeviceRead);
      autoView(dcost_by_dX_K_v, dcost_by_dX_K, DeviceRead);
      autoView(dcost_by_dX_V_v, dcost_by_dX_V, DeviceRead);

      accelerator_for3d(b,B,e,E,c,C, 1, {
	  dcost_by_dX_v(c,e,b) += dcost_by_dX_Q_v(c,e,b) + dcost_by_dX_K_v(c,e,b) + dcost_by_dX_V_v(c,e,b);
	});      
    }
  }
  leaf.v.deriv(cost_deriv, p, std::move(dcost_by_dX), input_above_deriv_return);
}

template<typename FloatType, typename InputType, typename Store>
void MultiHeadSelfAttentionLayer<FloatType,InputType,Store>::update(int off, const Vector<FloatType> &new_params){
  int p=off;
  multW_O.update(p,new_params);
  p += multW_O.nparams();
  for(int h=0;h<heads.size();h++){
    heads[h]->update(p,new_params);
    p += heads[h]->nparams();
  }
  leaf.v.update(p,new_params);
}

template<typename FloatType, typename InputType, typename Store>
void MultiHeadSelfAttentionLayer<FloatType,InputType,Store>::step(int off, const Vector<FloatType> &derivs, FloatType eps){
  int p=off;
  multW_O.step(p,derivs,eps);
  p += multW_O.nparams();
  for(int h=0;h<heads.size();h++){
    heads[h]->step(p,derivs,eps);
    p += heads[h]->nparams();
  }
  leaf.v.step(p,derivs,eps);
}
template<typename FloatType, typename InputType, typename Store>
void MultiHeadSelfAttentionLayer<FloatType,InputType,Store>::getParams(Vector<FloatType> &into, int off){
  int p=off;
  multW_O.getParams(into,p);
  p += multW_O.nparams();
  for(int h=0;h<heads.size();h++){
    heads[h]->getParams(into,p);
    p += heads[h]->nparams();
  }
  leaf.v.getParams(into,p);
}

template<typename FloatType, typename InputType, typename Store>
void MultiHeadSelfAttentionLayer<FloatType,InputType,Store>::resizeInputBuffer(size_t to){
  multW_O.resizeInputBuffer(to);
  for(int h=0;h<heads.size();h++)
    heads[h]->resizeInputBuffer(to);
  leaf.v.resizeInputBuffer(to);
}
