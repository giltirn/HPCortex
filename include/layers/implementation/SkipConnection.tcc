template<typename FloatType, typename InputType, typename ChainInternal, typename ChainBelow>
SkipConnection<FloatType,InputType,ChainInternal,ChainBelow>::LayerInputOutputType SkipConnection<FloatType,InputType,ChainInternal,ChainBelow>::value(const InputType &x){
  LayerInputOutputType in = leaf_below.v.value(x);
  LayerInputOutputType out = in + leaf_internal.v.value(in);
  
  in_buf.push(std::move(in));
  return out;
}

template<typename FloatType, typename InputType, typename ChainInternal, typename ChainBelow>
void SkipConnection<FloatType,InputType,ChainInternal,ChainBelow>::deriv(Vector<FloatType> &cost_deriv, int off, LayerInputOutputType &&_above_deriv, InputType* input_above_deriv_return) const{
  int p=off;
  LayerInputOutputType layer_deriv;
  {
    LayerInputOutputType above_deriv(std::move(_above_deriv)); //inside the braces above ensures this object is freed before the next layer is called
      
    //until the pipeline is "primed", the ring buffers will pop uninitialized values. We could in principle skip doing any computation until then
    //but for now we just initialize with zero values (TODO: revisit)
    LayerInputOutputType in = in_buf.isFilled() ? in_buf.pop(): LayerInputOutputType(in_buf.latest());
      
    //f_i(x) = g_i(x) + x_i

    //deriv wrt inputs for backprop
    //df_i/dx_j = dg_i/dx_j + delta_ij
      
    //dcost / dx_j = \sum_i dcost/df_i df_i/dx_j
    //             = \sum_i dcost/df_i dg_i/dx_j  + \sum_i dcost/df_i delta_ij
    //             = \sum_i dcost/df_i dg_i/dx_j  + \sum_i dcost/df_j
      
    //deriv wrt params for filling cost_deriv
    //df_i/dparam_p = dg_i/dparam_p

    layer_deriv = above_deriv; //dcost/df_j
    LayerInputOutputType leaf_internal_deriv; //\sum_i dcost/df_i dg_i/dx_j
    leaf_internal.v.deriv(cost_deriv, p, std::move(above_deriv), &leaf_internal_deriv);

    layer_deriv += leaf_internal_deriv;

    p += leaf_internal.v.nparams();  
  }//close views and free temporaries before calling layer below
    
  leaf_below.v.deriv(cost_deriv, p, std::move(layer_deriv), input_above_deriv_return);
}
template<typename FloatType, typename InputType, typename ChainInternal, typename ChainBelow>
void SkipConnection<FloatType,InputType,ChainInternal,ChainBelow>::update(int off, const Vector<FloatType> &new_params){
  int p=off;
  leaf_internal.v.update(p, new_params);
  p += leaf_internal.v.nparams();
  leaf_below.v.update(p, new_params);
}

template<typename FloatType, typename InputType, typename ChainInternal, typename ChainBelow>
void SkipConnection<FloatType,InputType,ChainInternal,ChainBelow>::step(int off, const Vector<FloatType> &derivs, FloatType eps){
  int p=off;
  leaf_internal.v.step(p, derivs, eps);
  p += leaf_internal.v.nparams();
  leaf_below.v.step(p, derivs, eps);
}

//off measured from *end*, return new off
template<typename FloatType, typename InputType, typename ChainInternal, typename ChainBelow>
void SkipConnection<FloatType,InputType,ChainInternal,ChainBelow>::getParams(Vector<FloatType> &into, int off){
  int p = off;
  leaf_internal.v.getParams(into, p);
  p += leaf_internal.v.nparams();
  leaf_below.v.getParams(into,p);
}
