#include <HPCortex.hpp>
#include <Testing.hpp>
#include <layers/ReplicateLayer.hpp>

typedef confDouble Config;
typedef typename Config::FloatType FloatType;

//To test the layer by itself, this wrapper concetates the matrix output of two chains
template<typename ChainType1, typename ChainType2>
struct ReplicateLayerWrapper{
  typedef typename ChainType1::FloatType FloatType;

  ChainType1 &chain1;
  ChainType2 &chain2;

  size_t size_lin_out;
  size_t size_lin_in;
  int szin;
  int sz1;
  int sz2;
  
  ReplicateLayerWrapper(ChainType1 &chain1, ChainType1 &chain2, int szin, int sz1, int sz2): chain1(chain1),chain2(chain2),sz1(sz1),sz2(sz2),szin(szin){
    size_lin_in = szin;
    size_lin_out = sz1 + sz2;
  }

  size_t outputLinearSize() const{ return size_lin_out; }
  size_t inputLinearSize() const{ return size_lin_in; }

  Matrix<FloatType> value(const Matrix<FloatType> &in, EnableDeriv enable_deriv = DerivNo){    
    Matrix<FloatType> out(size_lin_out,in.size(1));
    auto v1 = chain1.value(in,enable_deriv);
    auto v2 = chain2.value(in,enable_deriv);
    int off = flattenToBatchVector(out,v1,0);
    flattenToBatchVector(out,v2,off);
    return out;
  }
  void deriv(Vector<FloatType> &cost_deriv_params, int off, Matrix<FloatType> &&_above_deriv_lin, Matrix<FloatType> &cost_deriv_inputs){
    int batch_size = _above_deriv_lin.size(1);    
    Matrix<FloatType> above_deriv_1(sz1,batch_size), above_deriv_2(sz2,batch_size);
    int poff = unflattenFromBatchVector(above_deriv_1, _above_deriv_lin, 0);
    unflattenFromBatchVector(above_deriv_2, _above_deriv_lin, poff);

    off = chain1.deriv(cost_deriv_params, off, std::move(above_deriv_1), &cost_deriv_inputs);
    chain2.deriv(cost_deriv_params, off, std::move(above_deriv_2), &cost_deriv_inputs);

    assert(cost_deriv_inputs.size(0) == szin && cost_deriv_inputs.size(1) == batch_size);
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

  std::string inCoord(size_t i,int b,int batch_size) const{
    return std::to_string(i)+","+std::to_string(b);
  }       
};

void testReplicateLayer(){
  std::mt19937 rng(1234);

  int B = 4;
  int in_sz = 3;
  int szbase = 6;

  Matrix<FloatType> wbase(szbase,in_sz);
  uniformRandom(wbase,rng);
  Vector<FloatType> bbase(szbase);
  uniformRandom(bbase,rng);
  
  auto base = dnn_layer(wbase,bbase,input_layer<Config,Matrix<FloatType>>());
  auto repls = replicate_layer(2, base);
  
  assert(repls.size() == 2);

  int sz1 = 7;
  Matrix<FloatType> w1(sz1,szbase);
  uniformRandom(w1,rng);
  Vector<FloatType> b1(sz1);
  uniformRandom(b1,rng);
  auto chain1 = dnn_layer(w1,b1, *repls[0]);

  int sz2 = 8;
  Matrix<FloatType> w2(sz2,szbase);
  uniformRandom(w2,rng);
  Vector<FloatType> b2(sz2);
  uniformRandom(b2,rng);
  auto chain2 = dnn_layer(w2,b2, *repls[1]);
  
  //Check nparams
  int nparams = chain1.nparams() + chain2.nparams();
  assert(nparams == (sz2*(szbase+1) + sz1*(szbase+1) + szbase*(in_sz + 1) ) );

  //For these functions we need to ensure always to call in the same order so as to preserve the indexing of the parameters
  //Specifically the parameters will be ordered as :  w1, b1, w2, b2, wbase, bbase
  //Check getParams
  Vector<FloatType> orig_params(nparams);
  int off = chain1.getParams(orig_params, 0);
  assert( chain2.getParams(orig_params, off) == nparams );
  {
    Matrix<FloatType> w1check(sz1,szbase);
    Vector<FloatType> b1check(sz1);
    Matrix<FloatType> w2check(sz2,szbase);
    Vector<FloatType> b2check(sz2);
    Matrix<FloatType> wbasecheck(szbase,in_sz);
    Vector<FloatType> bbasecheck(szbase);
    
    autoView(orig_params_v,orig_params, HostRead);
    FloatType const* orig_params_p = orig_params_v.data();
    orig_params_p = unflatten(w1check,orig_params_p);
    orig_params_p = unflatten(b1check,orig_params_p);
    orig_params_p = unflatten(w2check,orig_params_p);
    orig_params_p = unflatten(b2check,orig_params_p);
    orig_params_p = unflatten(wbasecheck,orig_params_p);
    orig_params_p = unflatten(bbasecheck,orig_params_p);

    assert(equal(w1,w1check,true));
    assert(equal(b1,b1check,true));
    assert(equal(w2,w2check,true));
    assert(equal(b2,b2check,true));
    assert(equal(wbase,wbasecheck,true));
    assert(equal(bbase,bbasecheck,true));
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
    Matrix<FloatType> x(in_sz, B);
    uniformRandom(x,rng);
    
    auto base_solo = dnn_layer(wbase,bbase,input_layer<Config,Matrix<FloatType>>());
    auto vbase = base_solo.value(x);

    auto chain1_solo = dnn_layer(w1,b1,input_layer<Config,Matrix<FloatType>>());
    auto chain2_solo = dnn_layer(w2,b2,input_layer<Config,Matrix<FloatType>>());

    auto expect1 = chain1_solo.value(vbase);
    auto expect2 = chain2_solo.value(vbase);

    auto got1 = chain1.value(x); //this call will store the output
    auto got2 = chain2.value(x); //input technically irrelevant here!

    assert(abs_near(expect1,got1,FloatType(1e-6),true));
    assert(abs_near(expect2,got2,FloatType(1e-6),true));
  }

  ReplicateLayerWrapper<decltype(chain1),decltype(chain2)> wrp(chain1, chain2, in_sz, sz1, sz2);
  testComponentDeriv(wrp,B, FloatType(1e-6));
  testComponentDiffBatchSizes(wrp);
  std::cout << "testReplicateLayer passed" << std::endl;
}


int main(int argc, char** argv){
  initialize(argc,argv);
  testReplicateLayer();
  return 0;
}


