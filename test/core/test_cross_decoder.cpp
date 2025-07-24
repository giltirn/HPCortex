#include <HPCortex.hpp>
#include <Testing.hpp>

typedef confDouble Config;
typedef typename Config::FloatType FloatType;

template<typename ChainType>
struct CrossDecoderWrapper{
  typedef typename ChainType::FloatType FloatType;

  ChainType &chain;

  size_t size_lin_in;
  size_t size_lin_out;

  int C;
  int E;
  int B;
  int sz[3];
  
  CrossDecoderWrapper(ChainType &chain, int C, int E, int B): chain(chain), C(C), E(E), B(B){
    size_lin_in = 2*C*E*B;
    size_lin_out = C*E*B;
    sz[0] = C; sz[1] = E; sz[2] = B;
  }
  
  size_t outputLinearSize() const{ return size_lin_out; }
  size_t inputLinearSize() const{ return size_lin_in; }
  
  Vector<FloatType> value(const Vector<FloatType> &in){
    std::pair< Tensor<FloatType,3>, Tensor<FloatType,3> > inm;
    inm.first = Tensor<FloatType,3>(sz);
    inm.second = Tensor<FloatType,3>(sz);
    {
      autoView(in_v,in,HostRead);
      FloatType const* p = in_v.data();
      p = unflatten(inm.first,p);
      p = unflatten(inm.second,p);
    }
    Vector<FloatType> out(size_lin_out);
    {
      autoView(out_v,out,HostWrite);
      auto v = chain.value(inm);
      FloatType* p = out_v.data();
      p = flatten(p, v);
    }
    return out;
  }
  void deriv(Vector<FloatType> &cost_deriv_params, int off, Vector<FloatType> &&_above_deriv_lin, Vector<FloatType> &cost_deriv_inputs){
    Vector<FloatType> above_deriv_lin = std::move(_above_deriv_lin);
    Tensor<FloatType,3> above_deriv(sz);
    unflatten(above_deriv, above_deriv_lin);

    std::pair< Tensor<FloatType,3>, Tensor<FloatType,3> > cost_deriv_inputs_m;
    off = chain.deriv(cost_deriv_params, off, std::move(above_deriv), &cost_deriv_inputs_m);
    cost_deriv_inputs = Vector<FloatType>(size_lin_in);
    {
      autoView(out_v, cost_deriv_inputs,HostWrite);
      FloatType * p = out_v.data();
      p = flatten(p, cost_deriv_inputs_m.first);
      p = flatten(p, cost_deriv_inputs_m.second);
    }
  }
    
  void update(int off, const Vector<FloatType> &new_params){
    chain.update(off,new_params);
  }
  void step(int off, const Vector<FloatType> &derivs, FloatType eps){
    chain.step(off,derivs,eps);    
  }
  inline int nparams() const{ return chain.nparams(); }

  void getParams(Vector<FloatType> &into, int off){
    chain.getParams(into,off);
  }

  std::string inCoord(size_t i) const{
    return std::to_string(i);
  }       
};



void testCrossDecoder(){
  std::mt19937 rng(1234);

  typedef Tensor<FloatType,3> TensorType;
  typedef std::pair<TensorType,TensorType> InputType;

  auto splt = pair_split_layer(input_layer<Config,InputType>());

  int C = 3;
  int E = 4;
  int B = 5;
  int nheads = 2;
  int d_act = 7;

  //Do something non-trivial for the "encoder" side
  Matrix<FloatType> w1(E,E);
  uniformRandom(w1,rng);
  Vector<FloatType> b1(E);
  uniformRandom(b1,rng);  
  
  auto encoder_in = batch_tensor_dnn_layer<3>(w1, b1, 1, ReLU<FloatType>(), *splt.first);

  //Do something non-trivial prior to the decoder block
  Matrix<FloatType> w2(E,E);
  uniformRandom(w2,rng);
  Vector<FloatType> b2(E);
  uniformRandom(b2,rng);  
  
  auto decoder_in = batch_tensor_dnn_layer<3>(w2, b2, 1, ReLU<FloatType>(), *splt.second);

  reseedGlobalRNG(9876);
  auto xdecoder = transformer_cross_decoder_block(E, nheads, d_act, GeLU<FloatType>(), encoder_in, decoder_in );

  auto xdecoder_solo_in = pair_split_layer(input_layer<Config,InputType>());
  reseedGlobalRNG(9876); //ensure it gets the same initial params as the above
  auto xdecoder_solo = transformer_cross_decoder_block(E, nheads, d_act, GeLU<FloatType>(), *xdecoder_solo_in.first, *xdecoder_solo_in.second );
  int decoder_solo_nparam = xdecoder_solo.nparams();

  auto encoder_in_solo = batch_tensor_dnn_layer<3>(w1, b1, 1, ReLU<FloatType>(), input_layer<Config,TensorType>());
  auto decoder_in_solo = batch_tensor_dnn_layer<3>(w2, b2, 1, ReLU<FloatType>(), input_layer<Config,TensorType>());
  
  //Check nparams
  int nparams = xdecoder.nparams();
  assert(nparams == (E*(E+1) + E*(E+1)) + decoder_solo_nparam );

  InputType x;
  x.first = TensorType(C,E,B);
  x.second = TensorType(C,E,B);
  uniformRandom(x.first,rng);
  uniformRandom(x.second,rng);
  
  auto got = xdecoder.value(x);

  auto y1 = encoder_in_solo.value(x.first);
  auto y2 = decoder_in_solo.value(x.second);
  InputType y12(y1,y2);
  
  auto expect = xdecoder_solo.value(y12);

  assert(abs_near(got,expect, FloatType(1e-8), true));
  
  CrossDecoderWrapper<typename std::decay<decltype(xdecoder)>::type> wrp(xdecoder,C,E,B);
  testComponentDeriv(wrp, FloatType(1e-9), true);

  std::cout << "testCrossDecoder passed" << std::endl;
}



void testCrossDecoderMultiBlock(){
  std::mt19937 rng(1234);

  typedef Tensor<FloatType,3> TensorType;
  typedef std::pair<TensorType,TensorType> InputType;

  auto splt = pair_split_layer(input_layer<Config,InputType>());

  int C = 3;
  int E = 4;
  int B = 2;
  int nheads = 2;
  int d_act = 7;

  //Do something non-trivial for the "encoder" side
  Matrix<FloatType> w1(E,E);
  uniformRandom(w1,rng);
  Vector<FloatType> b1(E);
  uniformRandom(b1,rng);  
  
  auto encoder_in = batch_tensor_dnn_layer<3>(w1, b1, 1, ReLU<FloatType>(), *splt.first);
  auto encoder_in_repl = replicate_layer(2, encoder_in);
  
  //Do something non-trivial prior to the decoder block
  Matrix<FloatType> w2(E,E);
  uniformRandom(w2,rng);
  Vector<FloatType> b2(E);
  uniformRandom(b2,rng);  
  
  auto decoder_in = batch_tensor_dnn_layer<3>(w2, b2, 1, ReLU<FloatType>(), *splt.second);

  reseedGlobalRNG(9876);
  auto xdecoder1 = transformer_cross_decoder_block(E, nheads, d_act, GeLU<FloatType>(), *encoder_in_repl[0], decoder_in );
  auto xdecoder2 = transformer_cross_decoder_block(E, nheads, d_act, GeLU<FloatType>(), *encoder_in_repl[1], xdecoder1 );

  auto xdecoder_solo_in = pair_split_layer(input_layer<Config,InputType>());
  auto xdecoder_solo2_in = pair_split_layer(input_layer<Config,InputType>());
  
  reseedGlobalRNG(9876); //ensure it gets the same initial params as the above
  auto xdecoder_solo = transformer_cross_decoder_block( E, nheads, d_act, GeLU<FloatType>(), *xdecoder_solo_in.first, *xdecoder_solo_in.second );
  auto xdecoder_solo2 = transformer_cross_decoder_block(E, nheads, d_act, GeLU<FloatType>(), *xdecoder_solo2_in.first, *xdecoder_solo2_in.second );
  
  int decoder_solo_nparam = xdecoder_solo.nparams();

  auto encoder_in_solo = batch_tensor_dnn_layer<3>(w1, b1, 1, ReLU<FloatType>(), input_layer<Config,TensorType>());
  auto decoder_in_solo = batch_tensor_dnn_layer<3>(w2, b2, 1, ReLU<FloatType>(), input_layer<Config,TensorType>());
  
  //Check nparams
  int nparams = xdecoder2.nparams();
  assert(nparams == (E*(E+1) + E*(E+1)) + 2*decoder_solo_nparam );

  InputType x;
  x.first = TensorType(C,E,B);
  x.second = TensorType(C,E,B);
  uniformRandom(x.first,rng);
  uniformRandom(x.second,rng);
  
  auto got = xdecoder2.value(x);

  auto y1 = encoder_in_solo.value(x.first);
  auto y2 = decoder_in_solo.value(x.second);
  InputType y12(y1,y2);
  
  auto dec1out = xdecoder_solo.value(y12);
  InputType yout1(y1, dec1out);
  auto expect = xdecoder_solo2.value(yout1);

  assert(abs_near(got,expect, FloatType(1e-8), true));
  
  CrossDecoderWrapper<typename std::decay<decltype(xdecoder2)>::type> wrp(xdecoder2,C,E,B);
  testComponentDeriv(wrp, FloatType(1e-9),true);

  std::cout << "testCrossDecoderMultiBlock passed" << std::endl;
}



int main(int argc, char** argv){
  initialize(argc,argv);
  testCrossDecoder();
  testCrossDecoderMultiBlock();
  return 0;
}


