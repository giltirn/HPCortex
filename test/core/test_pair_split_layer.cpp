#include <HPCortex.hpp>
#include <Testing.hpp>
#include <layers/PairSplitLayer.hpp>

typedef confDouble Config;
typedef typename Config::FloatType FloatType;

//To test the layer by itself, this wrapper concetates the matrix output of two chains
template<typename ChainType1, typename ChainType2>
struct PairSplitLayerWrapper{
  typedef typename ChainType1::FloatType FloatType;

  ChainType1 &chain1;
  ChainType2 &chain2;

  size_t size_lin_in;
  size_t size_lin_out;

  int sz1;
  int in_sz1;
  int sz2;
  int in_sz2;
  
  PairSplitLayerWrapper(ChainType1 &chain1, ChainType2 &chain2, int sz1, int in_sz1, int sz2, int in_sz2): chain1(chain1),chain2(chain2),sz1(sz1),in_sz1(in_sz1),sz2(sz2),in_sz2(in_sz2){
    size_lin_in = in_sz1 + in_sz2;
    size_lin_out = sz1 + sz2;
  }
  
  size_t outputLinearSize() const{ return size_lin_out; }
  size_t inputLinearSize() const{ return size_lin_in; }
  
  Matrix<FloatType> value(const Matrix<FloatType> &in, EnableDeriv enable_deriv = DerivNo){
    int batch_size = in.size(1);
    std::pair< Matrix<FloatType>, Matrix<FloatType> > inm;
    inm.first = Matrix<FloatType>(in_sz1,batch_size);
    inm.second = Matrix<FloatType>(in_sz2,batch_size);
    unflattenFromBatchVector(inm.first,in,0);
    unflattenFromBatchVector(inm.second,in,in_sz1);

    Matrix<FloatType> out(size_lin_out,batch_size);
    auto v1 = chain1.value(inm,enable_deriv);
    auto v2 = chain2.value(inm,enable_deriv);
    flattenToBatchVector(out,v1,0);
    flattenToBatchVector(out,v2,sz1);
    return out;
  }
  void deriv(Vector<FloatType> &cost_deriv_params, int off, Matrix<FloatType> &&_above_deriv_lin, Matrix<FloatType> &cost_deriv_inputs){
    int batch_size = _above_deriv_lin.size(1);
    Matrix<FloatType> above_deriv_1(sz1,batch_size), above_deriv_2(sz2,batch_size);
    unflattenFromBatchVector(above_deriv_1,_above_deriv_lin,0);
    unflattenFromBatchVector(above_deriv_2,_above_deriv_lin,sz1);
    
    std::pair< Matrix<FloatType>, Matrix<FloatType> > cost_deriv_inputs_m;
    off = chain1.deriv(cost_deriv_params, off, std::move(above_deriv_1), &cost_deriv_inputs_m);
    chain2.deriv(cost_deriv_params, off, std::move(above_deriv_2), &cost_deriv_inputs_m);
    assert( cost_deriv_inputs_m.first.size(0) == in_sz1 && cost_deriv_inputs_m.first.size(1) == batch_size );
    assert( cost_deriv_inputs_m.second.size(0) == in_sz2 && cost_deriv_inputs_m.second.size(1) == batch_size );

    cost_deriv_inputs = Matrix<FloatType>(size_lin_in,batch_size);
    flattenToBatchVector(cost_deriv_inputs, cost_deriv_inputs_m.first, 0);
    flattenToBatchVector(cost_deriv_inputs, cost_deriv_inputs_m.second, in_sz1);
  }
    
  void update(int off, const Vector<FloatType> &new_params){
    off = chain1.update(off,new_params);
    chain2.update(off,new_params);
  }
  void step(int off, const Vector<FloatType> &derivs, FloatType eps){
    off = chain1.step(off,derivs,eps);
    chain2.step(off,derivs,eps);    
  }
  inline int nparams() const{ return chain1.nparams() + chain2.nparams(); }
  void getParams(Vector<FloatType> &into, int off){
    off = chain1.getParams(into,off);
    chain2.getParams(into,off);
  }

  std::string inCoord(size_t i,int b, int batch_size) const{
    return std::to_string(i)+","+std::to_string(b);
  }       
};

void testPairSplitLayer(){
  std::mt19937 rng(1234);

  typedef std::pair< Matrix<FloatType>, Matrix<FloatType> > InputType;

  auto splt = pair_split_layer(input_layer<Config,InputType>());

  int B = 4;
  int in_sz1 = 3;
  int in_sz2 = 5;
   
  int sz1 = 7;
  Matrix<FloatType> w1(sz1,in_sz1);
  uniformRandom(w1,rng);
  Vector<FloatType> b1(sz1);
  uniformRandom(b1,rng);
  auto chain1 = dnn_layer(w1,b1,*splt.first);

  int sz2 = 8;
  Matrix<FloatType> w2(sz2,in_sz2);
  uniformRandom(w2,rng);
  Vector<FloatType> b2(sz2);
  uniformRandom(b2,rng);
  auto chain2 = dnn_layer(w2,b2,*splt.second);

  //Check nparams
  int nparams = chain1.nparams() + chain2.nparams();
  assert(nparams == (sz2*(in_sz2+1) + sz1*(in_sz1+1)) );

  //For these functions we need to ensure always to call in the same order so as to preserve the indexing of the parameters
  //Specifically the parameters will be ordered as :  w1, b1, w2, b2
  //Check getParams
  Vector<FloatType> orig_params(nparams);
  int off = chain1.getParams(orig_params, 0);
  assert( chain2.getParams(orig_params, off) == nparams );
  {
    Matrix<FloatType> w1check(sz1,in_sz1);
    Vector<FloatType> b1check(sz1);
    Matrix<FloatType> w2check(sz2,in_sz2);
    Vector<FloatType> b2check(sz2);
    
    autoView(orig_params_v,orig_params, HostRead);
    FloatType const* orig_params_p = orig_params_v.data();
    orig_params_p = unflatten(w1check,orig_params_p);
    orig_params_p = unflatten(b1check,orig_params_p);
    orig_params_p = unflatten(w2check,orig_params_p);
    orig_params_p = unflatten(b2check,orig_params_p);

    assert(equal(w1,w1check,true));
    assert(equal(b1,b1check,true));
    assert(equal(w2,w2check,true));
    assert(equal(b2,b2check,true));
  }
  //Check update and step
  {
    Vector<FloatType> np(nparams);
    uniformRandom(np,rng);
    int off = chain1.update(0,np);
    assert( chain2.update(off,np) == nparams );

    Vector<FloatType> pcheck(nparams);
    off = chain1.getParams(pcheck, 0);
    assert( chain2.getParams(pcheck, off) == nparams );

    assert(equal(np,pcheck,true));

    //restore orig
    off = chain1.update(0,orig_params);
    assert( chain2.update(off,orig_params) == nparams );

    Vector<FloatType> derivs(nparams);
    uniformRandom(derivs,rng);
    FloatType eps = 0.535;
    off = chain1.step(0,derivs,eps);
    assert( chain2.step(off,derivs,eps) == nparams );

    off = chain1.getParams(pcheck, 0);
    assert( chain2.getParams(pcheck, off) == nparams );

    Vector<FloatType> expect(nparams);
    doHost3(expect, orig_params, derivs, {
	for(int p=0;p<nparams;p++)
	  expect_v(p) = orig_params_v(p) - derivs_v(p)*eps;
      });
    assert(abs_near(expect, pcheck, FloatType(1e-8), true) );

    //restore orig
    off = chain1.update(0,orig_params);
    assert( chain2.update(off,orig_params) == nparams );
  }
  {
    //Check value    
    InputType x;
    x.first = Matrix<FloatType>(in_sz1,B);
    x.second = Matrix<FloatType>(in_sz2,B);
    uniformRandom(x.first,rng);
    uniformRandom(x.second,rng);
    
    auto chain1_solo = dnn_layer(w1,b1,input_layer<Config,Matrix<FloatType>>());
    auto chain2_solo = dnn_layer(w2,b2,input_layer<Config,Matrix<FloatType>>());

    auto expect1 = chain1_solo.value(x.first);
    auto expect2 = chain2_solo.value(x.second);

    auto got1 = chain1.value(x); //this call will store the output
    auto got2 = chain2.value(x); //input technically irrelevant here!

    assert(abs_near(expect1,got1,FloatType(1e-6),true));
    assert(abs_near(expect2,got2,FloatType(1e-6),true));
  }
  PairSplitLayerWrapper<decltype(chain1),decltype(chain2)> wrp(chain1, chain2, sz1, in_sz1, sz2, in_sz2);
  testComponentDeriv(wrp, B, FloatType(1e-6));
  testComponentDiffBatchSizes(wrp);
  
  std::cout << "testPairSplitLayer passed" << std::endl;
}

template<typename Model>
struct PairSplitJoinLayerWrapper{
  typedef typename Model::FloatType FloatType;

  Model &model;

  size_t size_lin_in;
  size_t size_lin_out;

  int sz1;
  int in_sz1;
  int sz2;
  int in_sz2;

  PairSplitJoinLayerWrapper(Model &model, int sz1, int in_sz1, int sz2, int in_sz2): model(model),sz1(sz1),in_sz1(in_sz1),sz2(sz2),in_sz2(in_sz2){
    size_lin_in = in_sz1 + in_sz2;
    size_lin_out = sz1 + sz2;
  }
  
  size_t outputLinearSize() const{ return size_lin_out; }
  size_t inputLinearSize() const{ return size_lin_in; }
  
  Matrix<FloatType> value(const Matrix<FloatType> &in, EnableDeriv enable_deriv = DerivNo){
    int batch_size = in.size(1);
    std::pair< Matrix<FloatType>, Matrix<FloatType> > inm;
    inm.first = Matrix<FloatType>(in_sz1,batch_size);
    inm.second = Matrix<FloatType>(in_sz2,batch_size);
    int off = unflattenFromBatchVector(inm.first,in,0);
    unflattenFromBatchVector(inm.second,in,off);
    
    Matrix<FloatType> out(size_lin_out,batch_size);
    auto v = model.value(inm,enable_deriv);
    off = flattenToBatchVector(out,v.first,0);
    flattenToBatchVector(out,v.second,off);
    return out;
  }
  void deriv(Vector<FloatType> &cost_deriv_params, int off, Matrix<FloatType> &&_above_deriv_lin, Matrix<FloatType> &cost_deriv_inputs){
    int batch_size = _above_deriv_lin.size(1);
    std::pair< Matrix<FloatType>, Matrix<FloatType> > above_deriv;
    above_deriv.first = Matrix<FloatType>(sz1,batch_size);
    above_deriv.second = Matrix<FloatType>(sz2,batch_size);
    int poff = unflattenFromBatchVector(above_deriv.first, _above_deriv_lin, 0);
    unflattenFromBatchVector(above_deriv.second, _above_deriv_lin, poff);    

    std::pair< Matrix<FloatType>, Matrix<FloatType> > cost_deriv_inputs_m;
    model.deriv(cost_deriv_params,off, std::move(above_deriv), &cost_deriv_inputs_m);
    assert( cost_deriv_inputs_m.first.size(0) == in_sz1 && cost_deriv_inputs_m.first.size(1) == batch_size );
    assert( cost_deriv_inputs_m.second.size(0) == in_sz2 && cost_deriv_inputs_m.second.size(1) == batch_size );

    cost_deriv_inputs = Matrix<FloatType>(size_lin_in,batch_size);
    poff = flattenToBatchVector(cost_deriv_inputs, cost_deriv_inputs_m.first, 0);
    flattenToBatchVector(cost_deriv_inputs, cost_deriv_inputs_m.second, poff);
  }
    
  void update(int off, const Vector<FloatType> &new_params){
    model.update(off,new_params);
  }
  void step(int off, const Vector<FloatType> &derivs, FloatType eps){
    model.step(off,derivs,eps);    
  }
  inline int nparams() const{ return model.nparams(); }
  void getParams(Vector<FloatType> &into, int off){
    model.getParams(into,off);
  }

  std::string inCoord(size_t i,int b,int batch_size) const{
    return std::to_string(i)+","+std::to_string(b);
  }       
};



void testDivergingConverging(){
  std::mt19937 rng(1234);

  typedef std::pair< Matrix<FloatType>, Matrix<FloatType> > InputType;

  auto splt = pair_split_layer(input_layer<Config,InputType>());

  int B = 4;
  int in_sz1 = 3;
  int in_sz2 = 5;
   
  int sz1 = 7;
  Matrix<FloatType> w1(sz1,in_sz1);
  uniformRandom(w1,rng);
  Vector<FloatType> b1(sz1);
  uniformRandom(b1,rng);
  auto chain1 = dnn_layer(w1,b1, *splt.first);

  int sz2 = 8;
  Matrix<FloatType> w2(sz2,in_sz2);
  uniformRandom(w2,rng);
  Vector<FloatType> b2(sz2);
  uniformRandom(b2,rng);
  auto chain2 = dnn_layer(w2,b2, *splt.second);


  auto conv = pair_join_layer(chain1,chain2);
  
  int nparams = chain1.nparams() + chain2.nparams();
  assert(nparams == (sz2*(in_sz2+1) + sz1*(in_sz1+1)) );

  //Check getParams
  Vector<FloatType> orig_params(nparams);
  assert( conv.getParams(orig_params,0) == nparams );
  {
    Matrix<FloatType> w1check(sz1,in_sz1);
    Vector<FloatType> b1check(sz1);
    Matrix<FloatType> w2check(sz2,in_sz2);
    Vector<FloatType> b2check(sz2);
    
    autoView(orig_params_v,orig_params, HostRead);
    FloatType const* orig_params_p = orig_params_v.data();
    orig_params_p = unflatten(w1check,orig_params_p);
    orig_params_p = unflatten(b1check,orig_params_p);
    orig_params_p = unflatten(w2check,orig_params_p);
    orig_params_p = unflatten(b2check,orig_params_p);

    assert(equal(w1,w1check,true));
    assert(equal(b1,b1check,true));
    assert(equal(w2,w2check,true));
    assert(equal(b2,b2check,true));
  }
  //Check update and step
  {
    Vector<FloatType> np(nparams);
    uniformRandom(np,rng);
    assert(conv.update(0,np) == nparams );

    Vector<FloatType> pcheck(nparams);
    assert( conv.getParams(pcheck, 0) == nparams );

    assert(equal(np,pcheck,true));

    //restore orig
    assert( conv.update(0,orig_params) == nparams );

    Vector<FloatType> derivs(nparams);
    uniformRandom(derivs,rng);
    FloatType eps = 0.535;
    assert( conv.step(0,derivs,eps) == nparams );

    assert( conv.getParams(pcheck, 0) == nparams );

    Vector<FloatType> expect(nparams);
    doHost3(expect, orig_params, derivs, {
	for(int p=0;p<nparams;p++)
	  expect_v(p) = orig_params_v(p) - derivs_v(p)*eps;
      });
    assert(abs_near(expect, pcheck, FloatType(1e-8), true) );

    //restore orig
    assert( conv.update(0,orig_params) == nparams );
  }
  {
    //Check value    
    InputType x;
    x.first = Matrix<FloatType>(in_sz1,B);
    x.second = Matrix<FloatType>(in_sz2,B);
    uniformRandom(x.first,rng);
    uniformRandom(x.second,rng);
    
    auto chain1_solo = dnn_layer(w1,b1, input_layer<Config,Matrix<FloatType>>() );
    auto chain2_solo = dnn_layer(w2,b2, input_layer<Config,Matrix<FloatType>>() );

    auto expect1 = chain1_solo.value(x.first);
    auto expect2 = chain2_solo.value(x.second);

    auto got = conv.value(x);

    assert(abs_near(expect1,got.first,FloatType(1e-6),true));
    assert(abs_near(expect2,got.second,FloatType(1e-6),true));
  }
  PairSplitJoinLayerWrapper<decltype(conv)> wrp(conv, sz1, in_sz1, sz2, in_sz2);
  testComponentDeriv(wrp,B, FloatType(1e-6));
  testComponentDiffBatchSizes(wrp);
  std::cout << "testDivergingConverging passed" << std::endl;
}
  
int main(int argc, char** argv){
  initialize(argc,argv);
  testPairSplitLayer();
  testDivergingConverging();
  return 0;
}


