#include <HPCortex.hpp>
#include <Testing.hpp>
#include <layers/ReplicateLayer.hpp>
typedef double FloatType;

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
  int batch_size;
  
  ReplicateLayerWrapper(ChainType1 &chain1, ChainType1 &chain2, int szin, int sz1, int sz2, int batch_size): chain1(chain1),chain2(chain2),sz1(sz1),sz2(sz2),szin(szin),batch_size(batch_size){
    size_lin_in = szin * batch_size;
    size_lin_out = sz1 * batch_size + sz2 * batch_size;
  }

  size_t outputLinearSize() const{ return size_lin_out; }
  size_t inputLinearSize() const{ return size_lin_in; }
  
  Vector<FloatType> value(const Vector<FloatType> &in){
    Matrix<FloatType> inm(szin,batch_size);
    unflatten(inm,in);
    Vector<FloatType> out(size_lin_out);
    {
      autoView(out_v,out,HostWrite);
      auto v1 = chain1.value(inm);
      auto v2 = chain2.value(inm);
      FloatType* p = out_v.data();
      p = flatten(p, v1);
      p = flatten(p, v2);
    }
    return out;
  }
  void deriv(Vector<FloatType> &cost_deriv_params, int off, Vector<FloatType> &&_above_deriv_lin, Vector<FloatType> &cost_deriv_inputs){
    Vector<FloatType> above_deriv_lin = std::move(_above_deriv_lin);
    Matrix<FloatType> above_deriv_1(sz1,batch_size), above_deriv_2(sz2,batch_size);
    {
      autoView(above_deriv_lin_v,above_deriv_lin,HostRead);
      FloatType const* p = above_deriv_lin_v.data();
      p = unflatten(above_deriv_1, p);
      p = unflatten(above_deriv_2, p);
    }
    Matrix<FloatType> cost_deriv_inputs_m;
    off = chain1.deriv(cost_deriv_params, off, std::move(above_deriv_1), &cost_deriv_inputs_m);
    chain2.deriv(cost_deriv_params, off, std::move(above_deriv_2), &cost_deriv_inputs_m);

    assert(cost_deriv_inputs_m.size(0) == szin && cost_deriv_inputs_m.size(1) == batch_size);
    cost_deriv_inputs = flatten(cost_deriv_inputs_m);
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

  std::string inCoord(size_t i) const{
    return std::to_string(i);
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
  
  auto base = dnn_layer(input_layer<FloatType,Matrix<FloatType>>(), wbase,bbase);
  auto repls = replicate_layer(base,2);
  
  assert(repls.size() == 2);

  int sz1 = 7;
  Matrix<FloatType> w1(sz1,szbase);
  uniformRandom(w1,rng);
  Vector<FloatType> b1(sz1);
  uniformRandom(b1,rng);
  auto chain1 = dnn_layer(*repls[0], w1,b1);

  int sz2 = 8;
  Matrix<FloatType> w2(sz2,szbase);
  uniformRandom(w2,rng);
  Vector<FloatType> b2(sz2);
  uniformRandom(b2,rng);
  auto chain2 = dnn_layer(*repls[1], w2,b2);
  
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
    
    auto base_solo = dnn_layer(input_layer<FloatType,Matrix<FloatType>>(), wbase,bbase);
    auto vbase = base_solo.value(x);

    auto chain1_solo = dnn_layer(input_layer<FloatType,Matrix<FloatType>>(), w1,b1);
    auto chain2_solo = dnn_layer(input_layer<FloatType,Matrix<FloatType>>(), w2,b2);

    auto expect1 = chain1_solo.value(vbase);
    auto expect2 = chain2_solo.value(vbase);

    auto got1 = chain1.value(x); //this call will store the output
    auto got2 = chain2.value(x); //input technically irrelevant here!

    assert(abs_near(expect1,got1,FloatType(1e-6),true));
    assert(abs_near(expect2,got2,FloatType(1e-6),true));
  }

  ReplicateLayerWrapper<decltype(chain1),decltype(chain2)> wrp(chain1, chain2, in_sz, sz1, sz2, B);
  testComponentDeriv(wrp, FloatType(1e-6));
  
  std::cout << "testReplicateLayer passed" << std::endl;
}

// //Another non-trivial test we can perform is to reproduce multi-head self-attention from multi-head cross-attention
// void testReplicateLayerAttention(){
//   std::mt19937 rng(1234);

//   int C=2;
//   int E=6;
//   int B=4;
//   int nheads = 3;
//   //Require W_Q[i], W_K[i] :  d_qk^(i) x E,     W_V[i] : d_v^(i) x E      W_O :  E x sum_i d_v^(i)
//   int d_qk = 3;
//   int d_v = 2;
  
//   std::vector<Matrix<FloatType> > W_Q(nheads), W_K(nheads), W_V(nheads);
//   for(int i=0;i<nheads;i++){
//     W_Q[i] = Matrix<FloatType>(d_qk,E); uniformRandom(W_Q[i],rng);
//     W_K[i] = Matrix<FloatType>(d_qk,E); uniformRandom(W_K[i],rng);
//     W_V[i] = Matrix<FloatType>(d_v,E); uniformRandom(W_V[i],rng);
//   }
//   Matrix<FloatType> W_O(E, nheads*d_v);
//   uniformRandom(W_O,rng);
//   typedef Tensor<FloatType,3> TensorType;
  
//   auto slf = multihead_self_attention_layer(input_layer<FloatType,TensorType>(),
// 					    nheads, W_Q, W_K, W_V, W_O);

//   auto splt = replicate_layer(input_layer<FloatType,TensorType>(),2);
//   auto jn = pair_join_layer(*splt[0],*splt[1]);

//   auto crs = multihead_cross_attention_layer(ChainQK &&chain_QK, ChainV &&chain_V,
// 				     int Nheads,
// 				     const std::vector<Matrix<FLOATTYPE(ChainQK)> > &W_Q,
// 				     const std::vector<Matrix<FLOATTYPE(ChainQK)> > &W_K,
// 				     const std::vector<Matrix<FLOATTYPE(ChainQK)> > &W_V,
// 				     const Matrix<FLOATTYPE(ChainQK)> &W_O,
//   auto crs = multihead_self_attention_layer(input_layer<FloatType,TensorType>(),
// 					    nheads, W_Q, W_K, W_V, W_O);

  
// }


int main(int argc, char** argv){
  initialize(argc,argv);
  testReplicateLayer();
  return 0;
}


