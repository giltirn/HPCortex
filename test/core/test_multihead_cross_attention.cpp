#include <HPCortex.hpp>
#include <Testing.hpp>

template<typename Config>
Tensor<typename Config::FloatType,3> naiveImpl(const Tensor<typename Config::FloatType,3> &XKV, const Tensor<typename Config::FloatType,3> &XQ,
			      const std::vector< Matrix<typename Config::FloatType> > &W_Q,
			      const std::vector< Matrix<typename Config::FloatType> > &W_K,
			      const std::vector< Matrix<typename Config::FloatType> > &W_V,
			      const Matrix<typename Config::FloatType> &W_O, bool use_mask){
  int Nheads = W_V.size();
  std::vector< Matrix<typename Config::FloatType> const* > W_Q_p(Nheads), W_K_p(Nheads), W_V_p(Nheads);
  for(int i=0;i<Nheads;i++){
    W_Q_p[i] = &W_Q[i];
    W_K_p[i] = &W_K[i];
    W_V_p[i] = &W_V[i];
  }
  MultiHeadAttentionComponent<Config> cpt(Nheads, W_Q_p.data(), W_K_p.data(), W_V_p.data(), W_O, use_mask);
  return cpt.value(XQ, XKV, XKV);
}

template<typename LayerType>
struct MultiHeadCrossAttentionLayerWrapper{
  typedef typename LayerType::FloatType FloatType;
  
  LayerType &cpt;
  int sizeKV[2];
  int sizeQ[2];
  int sizeOut[2];
  
  size_t size_lin_KV;
  size_t size_lin_Q;
  size_t size_lin_Out;

  MultiHeadCrossAttentionLayerWrapper(LayerType &cpt, int const* szOut, int const *szKV, int const *szQ): cpt(cpt){
    memcpy(sizeKV,szKV,2*sizeof(int));
    memcpy(sizeQ,szQ,2*sizeof(int));
    memcpy(sizeOut,szOut,2*sizeof(int));
    
    size_lin_KV = size_lin_Q = size_lin_Out = 1;
    for(int i=0;i<2;i++){
      size_lin_KV *= szKV[i];
      size_lin_Q *= szQ[i];
      size_lin_Out *= szOut[i];
    }
  }

  size_t outputLinearSize() const{ return size_lin_Out; }
  size_t inputLinearSize() const{ return size_lin_KV + size_lin_Q; }
  
  Matrix<FloatType> value(const Matrix<FloatType> &in, EnableDeriv enable_deriv = DerivNo){
    int batch_size = in.size(1);
    batchTensorSize(sizeKV_b, 3, sizeKV, batch_size);
    batchTensorSize(sizeQ_b, 3, sizeQ, batch_size);
    
    std::pair<Tensor<FloatType,3>, Tensor<FloatType,3> > X({ Tensor<FloatType,3>(sizeKV_b), Tensor<FloatType,3>(sizeQ_b) });
    int off = unflattenFromBatchVector(X.first,in,0);
    unflattenFromBatchVector(X.second,in,off);
    return flattenToBatchVector(cpt.value(X,enable_deriv));
  }
  void deriv(Vector<FloatType> &cost_deriv_params, int off, Matrix<FloatType> &&_above_deriv_lin, Matrix<FloatType> &cost_deriv_inputs){
    int batch_size = _above_deriv_lin.size(1);
    batchTensorSize(sizeOut_b, 3, sizeOut, batch_size);

    Tensor<FloatType,3> above_deriv = unflattenFromBatchVector<3>(_above_deriv_lin,sizeOut_b);

    std::pair<Tensor<FloatType,3>, Tensor<FloatType,3> > dcost_by_dIn;
    cpt.deriv(cost_deriv_params, off, std::move(above_deriv), &dcost_by_dIn);
    
    cost_deriv_inputs = Matrix<FloatType>(size_lin_KV+size_lin_Q, batch_size);
    int poff = flattenToBatchVector(cost_deriv_inputs,dcost_by_dIn.first,0);
    flattenToBatchVector(cost_deriv_inputs,dcost_by_dIn.second,poff);
  }
    
  void update(int off, const Vector<FloatType> &new_params){
    cpt.update(off,new_params);
  }
  void step(int off, const Vector<FloatType> &derivs, FloatType eps){
    cpt.step(off,derivs,eps);
  }
  inline int nparams() const{ return cpt.nparams(); }
  void getParams(Vector<FloatType> &into, int off){
    cpt.getParams(into, off);
  }

  std::string inCoord(size_t i, int b, int batch_size) const{
    std::ostringstream ss; ss << i << "," << b;
    return ss.str();
  }
   
    
};


void testMultiHeadCrossAttention(){
  typedef confDouble Config;
  typedef typename Config::FloatType FloatType;
  std::mt19937 rng(1234);
   
  typedef std::vector<FloatType> vecD;

  int C = 4;  //context window size
  int E = 20; //input embedding size
  int B = 5; //batch size

  int Nheads = 3;

  int d_k[3] = {8,9,10};
  int d_v[3] = {5,6,7};

  std::vector<Matrix<FloatType> > in_W_Q(Nheads);
  std::vector<Matrix<FloatType> > in_W_K(Nheads);
  std::vector<Matrix<FloatType> > in_W_V(Nheads);

  std::vector<Matrix<FloatType> const* > in_W_Q_p(Nheads);
  std::vector<Matrix<FloatType> const* > in_W_K_p(Nheads);
  std::vector<Matrix<FloatType> const* > in_W_V_p(Nheads);
  
  for(int h=0;h<Nheads;h++){
    in_W_Q[h] = Matrix<FloatType>(d_k[h],E);
    in_W_K[h] = Matrix<FloatType>(d_k[h],E);
    in_W_V[h] = Matrix<FloatType>(d_v[h],E);    
    uniformRandom(in_W_Q[h],rng);
    uniformRandom(in_W_K[h],rng);
    uniformRandom(in_W_V[h],rng);
    in_W_Q_p[h] = &in_W_Q[h];
    in_W_K_p[h] = &in_W_K[h];
    in_W_V_p[h] = &in_W_V[h];
  }
  
  int dsv = 5+6+7;
  int d_o = 8;
  Matrix<FloatType> in_W_O(d_o,dsv);
  uniformRandom(in_W_O,rng);

  typedef Tensor<FloatType,3> TensorType;
  typedef std::pair<TensorType,TensorType> TensorPair;
  
  for(int use_mask=0;use_mask<2;use_mask++){
    auto splt = pair_split_layer(input_layer<Config, TensorPair>());
    
    auto model = multihead_cross_attention_layer(Nheads, in_W_Q_p.data(), in_W_K_p.data(), in_W_V_p.data(), in_W_O, use_mask,
						 *splt.first,*splt.second
						 );
    typedef Tensor<FloatType,3> TensorType;
    std::pair<TensorType, TensorType> X({ TensorType(C,E,B), TensorType(C,E,B) });
    uniformRandom(X.first,rng);
    uniformRandom(X.second,rng);
      
    Tensor<FloatType,3> got = model.value(X);
    Tensor<FloatType,3> expect = naiveImpl<Config>(X.first,X.second,in_W_Q,in_W_K,in_W_V,in_W_O,use_mask);

    assert(abs_near(got,expect,FloatType(1e-4),true));

    MultiHeadCrossAttentionLayerWrapper<std::decay<decltype(model)>::type> wrp(model, expect.sizeArray(), X.first.sizeArray(), X.second.sizeArray());
    testComponentDeriv(wrp, B, FloatType(1e-6)); //same testing method as layers but more flexible, requiring a wrapper. We use this here because the input type is not a Tensor
    testComponentDiffBatchSizes(wrp);
  }
  std::cout << "Tests passed" << std::endl;
}

int main(int argc, char** argv){
  initialize(argc,argv);
  
  testMultiHeadCrossAttention();
  return 0;
}

