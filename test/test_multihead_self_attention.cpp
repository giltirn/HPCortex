#include <HPCortex.hpp>
#include <Testing.hpp>

template<typename FloatType>
Tensor<FloatType,3> naiveImpl(const Tensor<FloatType,3> &X,
			      const std::vector< Matrix<FloatType> > &W_Q,
			      const std::vector< Matrix<FloatType> > &W_K,
			      const std::vector< Matrix<FloatType> > &W_V,
			      const Matrix<FloatType> &W_O,
			      int C, int E, int B){
  int Nheads = W_V.size();
  int sdv=0;
  for(int i=0;i<Nheads; i++)
    sdv += W_V[i].size(0);
  assert(W_O.size(1) == sdv);
  int d_o = W_O.size(0);
  
  Tensor<FloatType,3> Yconcat(C,sdv,B,0.);
      
  //We assume the attention head is correct as it is tested elsewhere
  int off =0;
  for(int h=0;h<Nheads;h++){
    ScaledDotProductAttentionHeadComponent<FloatType> head(W_Q[h],W_K[h],W_V[h]);
    Tensor<FloatType,3> Yh = head.value(X,X,X);
    doHost2(Yh,Yconcat, {    
	for(int c=0;c<C;c++)
	  for(int b=0;b<B;b++)
	    for(int k=0;k<Yh.size(1);k++)
	      Yconcat_v(c,off+k,b) = Yh_v(c,k,b);
      });
    off += Yh.size(1);
  }
  assert(off == sdv);
  MatrixTensorContractComponent<FloatType,3> mulWO(W_O);
  Tensor<FloatType,3> out = mulWO.value(Yconcat);
  assert(out.size(0) == C && out.size(1) == d_o && out.size(2) == B);
  return out;
}

void testMultiHeadSelfAttention(){
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
 
  auto model = multihead_self_attention_layer(input_layer<FloatType, Tensor<FloatType,3> >(), Nheads, in_W_Q_p.data(), in_W_K_p.data(), in_W_V_p.data(), in_W_O);

  Tensor<FloatType,3> X(C,E,B);
  random(X,rng);
  
  Tensor<FloatType,3> got = model.value(X);
  Tensor<FloatType,3> expect = naiveImpl(X,in_W_Q,in_W_K,in_W_V,in_W_O,C,E,B);

  assert(abs_near(got,expect,FloatType(1e-4),true));

  int in_sizes[3] = {C,E,B};
  int out_sizes[3] = {C,d_o,B};
  
  testDeriv(model, in_sizes, out_sizes, FloatType(1e-6));

  std::cout << "Tests passed" << std::endl;
}

int main(int argc, char** argv){
  initialize(argc,argv);
  
  testMultiHeadSelfAttention();
  return 0;
}
