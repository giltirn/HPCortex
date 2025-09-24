#pragma once

#include <Tensors.hpp>
#include <random>

template<typename FloatType>
bool near(FloatType a, FloatType b, FloatType rel_tol, FloatType *reldiff_p = nullptr){
  FloatType diff = a - b;
  FloatType avg = (a + b)/2.;
  FloatType reldiff;
  if(avg == 0.0){
    if(diff != 0.0) reldiff=1.0;
    else reldiff = 0.0;
  }else{
    reldiff = diff / avg;
  }
  if(reldiff_p)  *reldiff_p = reldiff;
  
  if(fabs(reldiff) > rel_tol) return false;
  else return true;
}


template<typename FloatType>
bool near(const Vector<FloatType> &a,const Vector<FloatType> &b, FloatType rel_tol, bool verbose=false){
  if(a.size(0) != b.size(0)) return false;
  autoView(a_v,a,HostRead);
  autoView(b_v,b,HostRead);
  for(size_t i=0;i<a.size(0);i++){
    FloatType reldiff;
    bool nr = near(a_v(i),b_v(i),rel_tol,&reldiff);
    if(!nr){
      if(verbose) std::cout << i << " a:" << a_v(i) << " b:" << b_v(i) << " rel.diff:" << reldiff << std::endl;
      return false;
    }
  }
  return true;
}


template<typename FloatType>
bool near(const Matrix<FloatType> &a,const Matrix<FloatType> &b, FloatType rel_tol, bool verbose=false){
  if(a.size(0) != b.size(0) || a.size(1) != b.size(1)) return false;
  autoView(a_v,a,HostRead);
  autoView(b_v,b,HostRead);
  for(size_t i=0;i<a.size(0);i++){
    for(size_t j=0;j<a.size(1);j++){
    
      FloatType reldiff;
      bool nr = near(a_v(i,j),b_v(i,j),rel_tol,&reldiff);
      if(!nr){
	if(verbose) std::cout << i << " " << j << " a:" << a_v(i,j) << " b:" << b_v(i,j) << " rel.diff:" << reldiff << std::endl;
	return false;
      }
    }
  }
  return true;
}


template<typename FloatType>
bool abs_near(FloatType a, FloatType b, FloatType abs_tol, FloatType *absdiff_p = nullptr){
  FloatType absdiff = fabs(a - b);
  if(absdiff_p) *absdiff_p = absdiff;
  if(absdiff > abs_tol) return false;
  else return true;
}


template<typename FloatType>
bool abs_near(const Matrix<FloatType> &a,const Matrix<FloatType> &b, FloatType abs_tol, bool verbose=false){
  if(a.size(0) != b.size(0) || a.size(1) != b.size(1)){
    if(verbose) std::cout << "Size mismatch " << a.size(0) << ":" << b.size(0) << "  " << a.size(1) << ":" << b.size(1) << std::endl;
    return false;
  }
  autoView(a_v,a,HostRead);
  autoView(b_v,b,HostRead);
  for(size_t i=0;i<a.size(0);i++){
    for(size_t j=0;j<a.size(1);j++){
    
      FloatType absdiff;
      bool nr = abs_near(a_v(i,j),b_v(i,j),abs_tol,&absdiff);
      if(!nr){
	if(verbose) std::cout << i << " " << j << " a:" << a_v(i,j) << " b:" << b_v(i,j) << " abs.diff:" << absdiff << std::endl;
	return false;
      }
    }
  }
  return true;
}

template<typename FloatType, int Dim>
bool abs_near(const Tensor<FloatType,Dim> &a,const Tensor<FloatType,Dim> &b, FloatType abs_tol, bool verbose=false){
  int const* a_sz = a.sizeArray();
  int const* b_sz = b.sizeArray();
  for(int d=0;d<Dim;d++)
    if(a_sz[d] != b_sz[d]){
      if(verbose) std::cout << "Dimension mismatch on idx " << d << " a.size(idx)=" << a_sz[d] << " b.size(idx)=" << b_sz[d] << std::endl;
      return false;
    }
  
  autoView(a_v,a,HostRead);
  autoView(b_v,b,HostRead);
  for(size_t v = 0; v < a_v.data_len(); v++){
    FloatType absdiff;
    FloatType aa = a_v.data()[v];
    FloatType bb = b_v.data()[v];
    
    bool nr = abs_near(aa,bb,abs_tol,&absdiff);
    if(!nr){
      if(verbose){     
	int coord[Dim];    
	tensorOffsetUnmap<Dim>(coord,a_sz,v);
	for(int d=0;d<Dim;d++)
	  std::cout << coord[d] << " ";
	std::cout << "a:" << aa << " b:" << bb << " abs.diff:" << absdiff << std::endl;
      }
      return false;
    }
  }
  return true;
}

template<typename FloatType, int Dim>
bool equal(const Tensor<FloatType,Dim> &a,const Tensor<FloatType,Dim> &b, bool verbose=false){
  int const* a_sz = a.sizeArray();
  int const* b_sz = b.sizeArray();
  for(int d=0;d<Dim;d++) if(a_sz[d] != b_sz[d]) return false;
  
  autoView(a_v,a,HostRead);
  autoView(b_v,b,HostRead);
  for(size_t v = 0; v < a_v.data_len(); v++){
    FloatType aa = a_v.data()[v];
    FloatType bb = b_v.data()[v];
    
    bool eq = aa == bb;
    if(!eq){
      if(verbose){     
	int coord[Dim];    
	tensorOffsetUnmap<Dim>(coord,a_sz,v);
	for(int d=0;d<Dim;d++)
	  std::cout << coord[d] << " ";
	std::cout << "a:" << aa << " b:" << bb << " abs.diff:" << aa-bb << std::endl;
      }
      return false;
    }
  }
  return true;
}

template<typename Op, typename PreOp>
void benchmark(double &mean, double &std, int nrpt, int nwarmup, const Op &op, const PreOp &preop){
  auto t = now();
  for(int i=0;i<nwarmup;i++){
    preop();
    op();
  }
  double twarm = since(t);
  //std::cout << "Warmup " << twarm << std::endl;
  
  mean = std = 0.;
  for(int i=0;i<nrpt;i++){
    preop();
    t = now();
    op();
    double dt = since(t);
    mean += dt;
    std += dt*dt;

    //std::cout << i << " " << dt  << std::endl;
  }
  mean /= nrpt;
  std = sqrt( std/nrpt - mean*mean ); 
}

//A simple cost model for easy evaluation of derivatives : cost = \sum_i c_i * out_i   for linearized index i
template<typename TensType>
typename TensType::FloatType testCost(const Vector<typename TensType::FloatType> &c, const TensType &v){
  typename TensType::FloatType out = 0.;
  doHost2(c,v,{
      for(size_t i=0;i<c.size(0);i++)
	out += v_v.data()[i] * c_v(i);
    });
  return out;
}

//Test the derivative of the model is implemented correctly using discrete derivative. Works for any model with tensor input/output of arbitrary dimension. It only assumes value() and nparams() are correct.
//model: the ML model (sans cost function)
//in_sizes, out_sizes : the tensor dimensions of the input and output
//delta : the shift used to evaluate the discrete derivatives
template<typename ModelType>
void testDeriv(ModelType &model, int const* in_sizes, int const* out_sizes, typename ModelType::FloatType delta = typename ModelType::FloatType(1e-4)){
  typedef LAYEROUTPUTTYPE(ModelType) OutputType;
  typedef typename ModelType::InputType InputType;
  typedef typename ModelType::FloatType FloatType;
  
  constexpr int OutDim = OutputType::Dimension;
  constexpr int InDim = InputType::Dimension;

  std::mt19937 rng(1987);

  //Check basic functionality
  int nparam = model.nparams(); //assumed correct
  std::cout << "Nparam " << nparam << std::endl;
  
  Vector<FloatType> base_params(nparam);
  uniformRandom(base_params, rng);

  model.update(0, base_params);
  if(nparam>0){
    Vector<FloatType> testp(nparam);
    model.getParams(testp,0);
    assert(equal(testp,base_params,true));
  }
  if(nparam>0){
    Vector<FloatType> shifts(nparam);
    uniformRandom(shifts, rng);
    model.step(0,shifts,0.33);
    
    Vector<FloatType> pexpect = base_params - 0.33*shifts;
    Vector<FloatType> pgot(nparam);
    model.getParams(pgot,0);
    assert(abs_near(pexpect, pgot, 1e-4, true));
    
    model.update(0, base_params);
  }

  //Check derivatives  
  size_t vout = 1;
  for(int d=0;d<OutDim;d++)
    vout *= out_sizes[d];
  
  //let  cost = \sum_i c_i * out_i
  //above_deriv = dcost/dout_i = c_i
  Vector<FloatType> c(vout);
  uniformRandom(c,rng);

  InputType in_base(in_sizes);
  uniformRandom(in_base, rng);
  
  OutputType val_base = model.value(in_base, DerivYes);
  Vector<FloatType> pderiv_got(nparam,0.);

  OutputType above_deriv(out_sizes);
  doHost2(above_deriv, c, {
      for(size_t i=0;i<vout;i++)
	above_deriv_v.data()[i] = c_v(i);
    });
  InputType inderiv_got;
  model.deriv(pderiv_got,0,std::move(above_deriv),&inderiv_got);
  
  //Test parameter derivs
  //param_deriv = \sum_i dcost/dout_i  dout_i/dparam_j  =  \sum_i c_i dout_i/dparam_j
  FloatType cost_base = testCost(c, val_base);
  
  for(int j=0;j<nparam;j++){
    Vector<FloatType> pup(base_params);
    doHost(pup, { pup_v(j) += delta; });
    model.update(0,pup);
          
    OutputType vup = model.value(in_base);
    FloatType new_cost = testCost(c, vup);

    FloatType der = (new_cost - cost_base)/delta;
    doHost(pderiv_got, {
	std::cout << "Cost deriv wrt param " << j << " got " << pderiv_got_v(j) << " expect " << der << std::endl;
	assert(abs_near(der, pderiv_got_v(j), 1e-4)); 
      });
  }

  //Test layer deriv
  //layer_deriv_j = \sum_i dcost/dout_i dout_i/din_j
  model.update(0,base_params);

  size_t vin =  1;
  for(int d=0;d<InDim;d++)
    vin *= in_sizes[d];

  for(size_t j=0; j<vin; j++){
    InputType xup(in_base);
    doHost(xup, { xup_v.data()[j] += delta; });

    OutputType vup = model.value(xup);
    FloatType new_cost = testCost(c, vup);

    FloatType der = (new_cost - cost_base)/delta;
    doHost(inderiv_got, {
	FloatType der_got = inderiv_got_v.data()[j];
	std::cout << "Cost deriv wrt input linear idx " << j << " got " << der_got << " expect " << der << std::endl;
	assert(abs_near(der, der_got, 1e-4)); 
      });      
  }
};


//Same as the above but uses a wrapper object to linearize all inputs and outputs to vectors, allowing more complex usage patterns
template<typename ComponentWrapper>
void testComponentDeriv(ComponentWrapper &cpt, typename ComponentWrapper::FloatType delta = typename ComponentWrapper::FloatType(1e-4), bool _2nd_order =false){
  typedef typename ComponentWrapper::FloatType FloatType;
  
  std::mt19937 rng(1987);

  //Check basic functionality
  int nparam = cpt.nparams(); //assumed correct
  Vector<FloatType> base_params(nparam);
  uniformRandom(base_params, rng);

  cpt.update(0, base_params);
  if(nparam > 0){
    Vector<FloatType> testp(nparam);
    cpt.getParams(testp,0);
    assert(equal(testp,base_params,true));
  }
  if(nparam > 0){
    Vector<FloatType> shifts(nparam);
    uniformRandom(shifts, rng);
    cpt.step(0,shifts,0.33);
    
    Vector<FloatType> pexpect = base_params - 0.33*shifts;
    Vector<FloatType> pgot(nparam);
    cpt.getParams(pgot,0);
    assert(abs_near(pexpect, pgot, 1e-4, true));
    
    cpt.update(0, base_params);
  }

  //Check derivatives  
  size_t vout = cpt.outputLinearSize();
  size_t vin = cpt.inputLinearSize();
  
  //let  cost = \sum_i c_i * out_i
  //above_deriv = dcost/dout_i = c_i
  Vector<FloatType> c(vout);
  uniformRandom(c,rng);

  Vector<FloatType> in_base(vin);
  uniformRandom(in_base, rng);

  Vector<FloatType> val_base = cpt.value(in_base, DerivYes);
  Vector<FloatType> pderiv_got(nparam,0.);

  Vector<FloatType> inderiv_got;
  cpt.deriv(pderiv_got,0, Vector<FloatType>(c), inderiv_got);
  
  //Test parameter derivs
  //param_deriv = \sum_i dcost/dout_i  dout_i/dparam_j  =  \sum_i c_i dout_i/dparam_j
  FloatType cost_base = testCost(c, val_base);
  
  for(int j=0;j<nparam;j++){
    Vector<FloatType> pup(base_params);
    doHost(pup, { pup_v(j) += delta; });
    cpt.update(0,pup);
    
    Vector<FloatType> vup = cpt.value(in_base);
    FloatType new_cost = testCost(c, vup);
    FloatType der = (new_cost - cost_base)/delta;
    
    if(_2nd_order){
      pup = base_params;
      doHost(pup, { pup_v(j) -= delta; });
      cpt.update(0,pup);
      vup = cpt.value(in_base);
      FloatType new_cost_neg = testCost(c, vup);

      der = (new_cost - new_cost_neg)/(2*delta);
    }
      
    doHost(pderiv_got, {
	std::cout << "Cost deriv wrt param " << j << " got " << pderiv_got_v(j) << " expect " << der << std::endl;
	assert(abs_near(der, pderiv_got_v(j), 1e-4)); 
      });
  }

  //Test layer deriv
  //layer_deriv_j = \sum_i dcost/dout_i dout_i/din_j
  cpt.update(0,base_params);

  for(size_t j=0; j<vin; j++){
    Vector<FloatType> xup(in_base);
    doHost(xup, { xup_v.data()[j] += delta; });

    Vector<FloatType> vup = cpt.value(xup);
    FloatType new_cost = testCost(c, vup);   
    FloatType der = (new_cost - cost_base)/delta;

    if(_2nd_order){
      xup = in_base;
      doHost(xup, { xup_v.data()[j] -= delta; });

      vup = cpt.value(xup);
      FloatType new_cost_neg = testCost(c, vup);
      der = (new_cost - new_cost_neg)/(2*delta);
    }     
    
    doHost(inderiv_got, {
	FloatType der_got = inderiv_got_v.data()[j];
	std::cout << "Cost deriv wrt input linear idx " << j << "=" << cpt.inCoord(j) << " got " << der_got << " expect " << der << std::endl;
	assert(abs_near(der, der_got, 1e-4)); 
      });      
  }
};


//Reference implementation of softmax
template<typename FloatType>
std::vector<FloatType> softMaxVector(const std::vector<FloatType> &v, FloatType beta = 1.0){
  int np=v.size();
  FloatType max = v[0];
  for(int i=1;i<np;i++)
    max = std::max(max, v[i]);

  FloatType norm = exp(beta*(v[0]-max));
  for(int i=1;i<np;i++)
    norm += exp(beta*(v[i]-max));

  std::vector<FloatType> out(np);
  for(int i=0;i<np;i++)
    out[i] = exp(beta*(v[i]-max)) / norm;

  return out;
}

/**
 * @brief For a model/layer with tensor input/output type, test the value and deriv functions to ensure they are consistent across multiple choices of batch size
 */
template<typename LayerType>
void testModelDiffBatchSizes(LayerType &layer, int const* other_dim_sizes){
  std::mt19937 rng(7666);
  typedef typename LayerType::InputType XtensorType;
  constexpr int Dim = XtensorType::Dimension;
  
  int size[Dim]; memcpy(size, other_dim_sizes, (Dim-1)*sizeof(int));

  //1 batch of 4
  size[Dim-1] = 4;  
  XtensorType x4(size);
  uniformRandom(x4,rng);
  auto y4 = layer.value(x4,DerivYes);
  typedef decltype(y4) YtensorType;

  YtensorType above_deriv4(y4.sizeArray());
  uniformRandom(above_deriv4,rng);
  XtensorType in_deriv4(size);
  Vector<FloatType> param_deriv4(layer.nparams(),0.);
  layer.deriv(param_deriv4,0,YtensorType(above_deriv4), &in_deriv4);

  //2 batches of 2
  YtensorType y4_rec2(y4.sizeArray());
  XtensorType in_deriv4_rec2(size);
  Vector<FloatType> param_deriv4_rec2(layer.nparams(),0.);
  for(int i=0;i<2;i++){
    XtensorType x2 = x4.sliceLastDimension(2*i,2*i+1);
    YtensorType y2 = layer.value(x2,DerivYes);
    y4_rec2.insertSliceLastDimension(y2,2*i,2*i+1);

    YtensorType above_deriv = above_deriv4.sliceLastDimension(2*i,2*i+1);
    XtensorType in_deriv(x2.sizeArray());
    Vector<FloatType> param_deriv(layer.nparams(),0.);
    layer.deriv(param_deriv,0, std::move(above_deriv), &in_deriv);

    param_deriv4_rec2 += param_deriv;
    in_deriv4_rec2.insertSliceLastDimension(in_deriv,2*i,2*i+1);
  }
  assert(abs_near(y4,y4_rec2,FloatType(1e-6),true));
  assert(abs_near(param_deriv4,param_deriv4_rec2,FloatType(1e-6),true));
  assert(abs_near(in_deriv4,in_deriv4_rec2,FloatType(1e-6),true));
  
  YtensorType y4_rec1(y4.sizeArray());
  XtensorType in_deriv4_rec1(size);
  Vector<FloatType> param_deriv4_rec1(layer.nparams(),0.);
  for(int i=0;i<4;i++){
    XtensorType x1 = x4.sliceLastDimension(i,i);
    YtensorType y1 = layer.value(x1,DerivYes);
    y4_rec1.insertSliceLastDimension(y1,i,i);

    YtensorType above_deriv = above_deriv4.sliceLastDimension(i,i);
    XtensorType in_deriv(x1.sizeArray());
    Vector<FloatType> param_deriv(layer.nparams(),0.);
    layer.deriv(param_deriv,0, std::move(above_deriv), &in_deriv);

    param_deriv4_rec1 += param_deriv;
    in_deriv4_rec1.insertSliceLastDimension(in_deriv,i,i);
  }
  assert(abs_near(y4,y4_rec1,FloatType(1e-6),true));
  assert(abs_near(param_deriv4,param_deriv4_rec1,FloatType(1e-6),true));
  assert(abs_near(in_deriv4,in_deriv4_rec1,FloatType(1e-6),true));
}
