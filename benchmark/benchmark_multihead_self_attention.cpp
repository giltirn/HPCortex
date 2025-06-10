#include<HPCortex.hpp>
#include<Testing.hpp>

void benchmarkMultiHeadSelfAttentionLayer(){
  std::mt19937 rng(1234);

  std::vector<int> context_sizes = { 32, 64, 128, 512 };
  std::vector<int> embedding_sizes = { 64, 128, 512 };
  std::vector<int> batch_sizes = { 1, 8, 32, 64 };
  std::vector<int> nheads = {1,2,4,8};
  //only those that are divisors of embedding_size will be used

  // std::vector<int> context_sizes = { 128 };
  // std::vector<int> embedding_sizes = { 512 };
  // std::vector<int> batch_sizes = { 64 };
  // std::vector<int> nheads = {4}; 

  for(int C : context_sizes){
    for(int E : embedding_sizes){
      for(int B : batch_sizes){
	for(int nhead : nheads){
	  if(E % nhead != 0) break;
	  std::cout << "C:" << C << " E:" << E << " B:" << B << " nhead:" << nhead << std::endl;

	  auto m = multihead_self_attention_layer(input_layer<float, Tensor<float,3> >(), nhead, E, false);

	  int tsize[3] = {C,E,B};
	  Tensor<float,3> x(tsize);
	  uniformRandom(x,rng);

	  double mu, sigma;
#if 1
	  Tensor<float,3> got;
	  benchmark(mu, sigma, 100, 1, [&]{
	    got = m.value(x);
	  }, []{});
	  
	  std::cout << "value: " << mu/1e-6 << "+-" << sigma/1e-6 << "us" << std::endl;
#endif
	  Tensor<float,3> above;
	  Vector<float> deriv(m.nparams());

	  benchmark(mu, sigma, 100, 1,
		    [&]{
		      profileStart();
		      m.deriv(deriv,0,std::move(above));
		      profileStop();
		    },
		    [&]{
		      above = Tensor<float,3>(tsize, 0.);
		      m.value(x);
		    }  );

	  std::cout << "deriv: " << mu/1e-6 << "+-" << sigma/1e-6 << "us" << std::endl;
	}
      }
    }
  }
}


int main(int argc, char** argv){
  initialize(argc,argv);
  benchmarkMultiHeadSelfAttentionLayer();
  return 0;
}
