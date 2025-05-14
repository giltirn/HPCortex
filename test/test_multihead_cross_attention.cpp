#include <HPCortex.hpp>
#include <Testing.hpp>

template<typename FloatType>
Tensor<FloatType,3> naiveImpl(const Tensor<FloatType,3> &XQK, const Tensor<FloatType,3> &XV,
			      const std::vector< Matrix<FloatType> > &W_Q,
			      const std::vector< Matrix<FloatType> > &W_K,
			      const std::vector< Matrix<FloatType> > &W_V,
			      const Matrix<FloatType> &W_O, bool use_mask){
  int Nheads = W_V.size();
  std::vector< Matrix<FloatType> const* > W_Q_p(Nheads), W_K_p(Nheads), W_V_p(Nheads);
  for(int i=0;i<Nheads;i++){
    W_Q_p[i] = &W_Q[i];
    W_K_p[i] = &W_K[i];
    W_V_p[i] = &W_V[i];
  }
  MultiHeadAttentionComponent<FloatType> cpt(Nheads, W_Q_p.data(), W_K_p.data(), W_V_p.data(), W_O, use_mask);
  return cpt.value(XQK, XQK, XV);
}

template<typename LayerType>
struct MultiHeadCrossAttentionLayerWrapper{
  typedef typename LayerType::FloatType FloatType;
  
  LayerType &cpt;
  int sizeQK[3];
  int sizeV[3];
  int sizeOut[3];
  
  size_t size_lin_QK;
  size_t size_lin_V;
  size_t size_lin_Out;

  MultiHeadCrossAttentionLayerWrapper(LayerType &cpt, int const* szOut, int const *szQK, int const *szV): cpt(cpt){
    memcpy(sizeQK,szQK,3*sizeof(int));
    memcpy(sizeV,szV,3*sizeof(int));
    memcpy(sizeOut,szOut,3*sizeof(int));
    
    size_lin_QK = size_lin_V = size_lin_Out = 1;
    for(int i=0;i<3;i++){
      size_lin_QK *= szQK[i];
      size_lin_V *= szV[i];
      size_lin_Out *= szOut[i];
    }
  }

  size_t outputLinearSize() const{ return size_lin_Out; }
  size_t inputLinearSize() const{ return size_lin_QK + size_lin_V; }
  
  Vector<FloatType> value(const Vector<FloatType> &in){
    std::pair<Tensor<FloatType,3>, Tensor<FloatType,3> > X({ Tensor<FloatType,3>(sizeQK), Tensor<FloatType,3>(sizeV) });
    {
      autoView(XQK_v, X.first, HostWrite);
      autoView(XV_v, X.second, HostWrite);
      autoView(in_v, in, HostRead);
    
      memcpy(XQK_v.data(),in_v.data()              , size_lin_QK*sizeof(FloatType));
      memcpy(XV_v.data(), in_v.data() + size_lin_QK, size_lin_V*sizeof(FloatType));
    }           
    return flatten(cpt.value(X));
  }
  void deriv(Vector<FloatType> &cost_deriv_params, int off, Vector<FloatType> &&_above_deriv_lin, Vector<FloatType> &cost_deriv_inputs){
    Vector<FloatType> above_deriv_lin = std::move(_above_deriv_lin);
    Tensor<FloatType,3> above_deriv(sizeOut);
    unflatten(above_deriv,above_deriv_lin);

    std::pair<Tensor<FloatType,3>, Tensor<FloatType,3> > dcost_by_dIn;
    cpt.deriv(cost_deriv_params, off, std::move(above_deriv), &dcost_by_dIn);
    
    cost_deriv_inputs = Vector<FloatType>(size_lin_QK+size_lin_V);
    {
      autoView(XQK_v, dcost_by_dIn.first, HostRead);
      autoView(XV_v, dcost_by_dIn.second, HostRead);
      autoView(out_v, cost_deriv_inputs, HostRead);
    
      memcpy(out_v.data(), XQK_v.data(), size_lin_QK*sizeof(FloatType));
      memcpy(out_v.data() + size_lin_QK, XV_v.data(), size_lin_V*sizeof(FloatType));
    }      
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

  std::string inCoord(size_t i) const{
    std::ostringstream ss; ss << i << std::endl;
    return ss.str();
  }
   
    
};


void testMultiHeadCrossAttention(){
  typedef double FloatType;
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
    random(in_W_Q[h],rng);
    random(in_W_K[h],rng);
    random(in_W_V[h],rng);
    in_W_Q_p[h] = &in_W_Q[h];
    in_W_K_p[h] = &in_W_K[h];
    in_W_V_p[h] = &in_W_V[h];
  }
  
  int dsv = 5+6+7;
  int d_o = 8;
  Matrix<FloatType> in_W_O(d_o,dsv);
  random(in_W_O,rng);

  for(int use_mask=0;use_mask<2;use_mask++){
    auto model = multihead_cross_attention_layer(
						 input_layer<FloatType, Tensor<FloatType,3> >(),
						 input_layer<FloatType, Tensor<FloatType,3> >(),
						 Nheads, in_W_Q_p.data(), in_W_K_p.data(), in_W_V_p.data(), in_W_O, use_mask
						 );
    typedef Tensor<FloatType,3> TensorType;
    std::pair<TensorType, TensorType> X({ TensorType(C,E,B), TensorType(C,E,B) });
    random(X.first,rng);
    random(X.second,rng);
      
    Tensor<FloatType,3> got = model.value(X);
    Tensor<FloatType,3> expect = naiveImpl(X.first,X.second,in_W_Q,in_W_K,in_W_V,in_W_O,use_mask);

    assert(abs_near(got,expect,FloatType(1e-4),true));

    MultiHeadCrossAttentionLayerWrapper<std::decay<decltype(model)>::type> wrp(model, expect.sizeArray(), X.first.sizeArray(), X.second.sizeArray());
    testComponentDeriv(wrp, FloatType(1e-6)); //same testing method as layers but more flexible, requiring a wrapper. We use this here because the input type is not a Tensor
  }
  std::cout << "Tests passed" << std::endl;
}

int main(int argc, char** argv){
  initialize(argc,argv);
  
  testMultiHeadCrossAttention();
  return 0;
}

