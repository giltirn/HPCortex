#include <HPCortex.hpp>
#include <Testing.hpp>

template<typename FloatType>
Tensor<FloatType,3> naiveImpl(const Tensor<FloatType,3> &X,
			      const Matrix<FloatType> &W_Q,
			      const Matrix<FloatType> &W_K,
			      const Matrix<FloatType> &W_V,
			      int C, int E, int B, int d_k, int d_v,
			      bool use_mask){
  
  //Attention:
  // 1)  Q_{ckb} = \sum_e (W_Q)_{k e} X_{c e b}     k \in {1.. d_k}
  // 2)  K_{ckb} = \sum_e (W_K)_{k e} X_{c e b}
  // 3)  S_{c c' b} = \sum_k  Q_{c k b} K_{c' k b}
  // 4)  V_{cvb} = \sum_e (W_V)_{v e} X_{c e b}     v \in {1.. d_v}
  // 5)  SS_{c c' b} = scaled_softmax_c' ( S )    such that the probability weights sum to unity along c'
  // 6)  Out_{c v b} = \sum_c' SS_{c c' b} V_{c' v b}

  autoView(W_Q_v,W_Q,HostReadWrite);
  autoView(W_K_v,W_K,HostReadWrite);
  autoView(W_V_v,W_V,HostReadWrite);
  autoView(X_v,X,HostReadWrite);
  
  Tensor<FloatType,3> Q(C,d_k,B, 0.), K(C,d_k,B, 0.), V(C,d_v,B, 0.);
  
  autoView(Q_v,Q,HostReadWrite);
  autoView(K_v,K,HostReadWrite);
  autoView(V_v,V,HostReadWrite);

  Tensor<FloatType,3> S(C,C,B, 0.), SS(C,C,B,0.), Out(C,d_v,B,0.);

  autoView(S_v,S,HostReadWrite);
  autoView(SS_v,S,HostReadWrite);
  autoView(Out_v,Out,HostReadWrite);
  

  //Q_{ckb} = \sum_e (W_Q)_{k e} X_{c e b}     k \in {1.. d_k}
  //K_{ckb} = \sum_e (W_K)_{k e} X_{c e b}
  //V_{cvb} = \sum_e (W_V)_{v e} X_{c e b}     v \in {1.. d_v}
  for(int c=0;c<C;c++){
    for(int b=0;b<B;b++){
      for(int e=0;e<E;e++){
	
	for(int k=0;k<d_k;k++){	  
	  Q_v(c,k,b) += W_Q_v(k,e) * X_v(c,e,b);
	  K_v(c,k,b) += W_K_v(k,e) * X_v(c,e,b);
	}
	for(int v=0;v<d_v;v++)
	  V_v(c,v,b) += W_V_v(v,e) * X_v(c,e,b);
      }
    }
  }

  //S_{c c' b} = \sum_k  Q_{c k b} K_{c' k b}
  for(int c=0;c<C;c++){
    for(int cp=0;cp<C;cp++){
      for(int b=0;b<B;b++){
	for(int k=0;k<d_k;k++)
	  S_v(c,cp,b) += Q_v(c,k,b) * K_v(cp,k,b);
	if(use_mask && cp > c) S_v(c,cp,b) += -10000;	  
      }
    }
  }

  //SS_{c c' b} = scaled_softmax_c' ( S )  
  for(int c=0;c<C;c++){
    for(int b=0;b<B;b++){
      std::vector<FloatType> spencil(C);
      for(int cp=0; cp<C; cp++)
	spencil[cp] = S_v(c,cp,b) / sqrt(FloatType(d_k));
      std::vector<FloatType> sspencil = softMaxVector(spencil);
      for(int cp=0; cp<C; cp++)
	SS_v(c,cp,b) = sspencil[cp];
    }
  }

  //Out_{c v b} = \sum_c' SS_{c c' b} V_{c' v b}
  for(int c=0;c<C;c++){
    for(int v=0;v<d_v;v++){
      for(int b=0;b<B;b++){

	for(int cp=0;cp<C;cp++)
	  Out_v(c,v,b) += SS_v(c,cp,b) * V_v(cp,v,b);
      }
    }
  }
  return Out;  
}

void testScaledDotProductSelfAttention(){
  typedef double FloatType;
  std::mt19937 rng(1234);
   
  typedef std::vector<FloatType> vecD;

  int C = 4;  //context window size
  int E = 20; //input embedding size
  int B = 5; //batch size

  int d_k = 15; //inner index between Q and K
  int d_v = 12; //output embedding size

  Matrix<FloatType> in_W_Q(d_k,E), in_W_K(d_k,E), in_W_V(d_v,E);
  uniformRandom(in_W_Q,rng);
  uniformRandom(in_W_K,rng);
  uniformRandom(in_W_V,rng);

  for(int use_mask = 0; use_mask < 2; use_mask++){
    auto model = scaled_dotproduct_self_attention_layer(input_layer<FloatType, Tensor<FloatType,3> >(), in_W_Q, in_W_K, in_W_V, use_mask);

    Tensor<FloatType,3> X(C,E,B);
    uniformRandom(X,rng);
  
    Tensor<FloatType,3> got = model.value(X);  
    Tensor<FloatType,3> expect = naiveImpl(X,in_W_Q,in_W_K,in_W_V,C,E,B,d_k,d_v,use_mask);

    assert(abs_near(got,expect,FloatType(1e-4),true));

    int in_sizes[3] = {C,E,B};
    int out_sizes[3] = {C,d_v,B};
  
    testDeriv(model, in_sizes, out_sizes, FloatType(1e-6));
  }

  std::cout << "Tests passed" << std::endl;
}

int main(int argc, char** argv){
  initialize(argc,argv);
  
  testScaledDotProductSelfAttention();
  return 0;
}
