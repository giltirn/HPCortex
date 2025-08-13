#include<HPCortex.hpp>
#include<Testing.hpp>

template<typename FloatType>
void step_v1(int off, const Vector<FloatType> &derivs, FloatType eps, Matrix<FloatType> &weights, Vector<FloatType> &bias, bool use_bias){
  autoView(derivs_v,derivs,DeviceRead);
  int p = off;
  {
    autoView(weights_v,weights,DeviceReadWrite);
    accelerator_for3d(dummy1,1,j,weights.size(1),i,weights.size(0),64,{
	int pp = p + j + weights_v.size(1)*i;
	weights_v(i,j) -= derivs_v(pp)*eps;
    });
    p += weights.size(0)*weights.size(1);
  }
  if(use_bias){
    autoView(bias_v,bias,DeviceReadWrite);
    accelerator_for2d(dummy1,1,i,weights.size(0),64,{
      int pp = p + i;
      bias_v(i) -= derivs_v(pp)*eps;
    });
  }
}

template<typename FloatType>
void step_v2(int off, const Vector<FloatType> &derivs, FloatType eps, Matrix<FloatType> &weights, Vector<FloatType> &bias, bool use_bias){
  autoView(derivs_v,derivs,DeviceRead);
  FloatType const* dp = derivs_v.data() + off;
  {
    autoView(weights_v,weights,DeviceReadWrite);    
    accelerator_for_gen(1,0,splitBlock<32>(),o,weights_v.data_len(), {
	weights_v.data()[o] -= dp[o]*eps;
      });
  }

  if(use_bias){
    dp += weights.data_len();
    
    autoView(bias_v,bias,DeviceReadWrite);
    accelerator_for_gen(1,0,splitBlock<32>(),o,bias.size(0), {
	bias_v.data()[o] -= dp[o]*eps;
      });	
  }
}


int main(int argc, char** argv){  
  initialize(argc,argv);
  std::mt19937 rng(1234);

  typedef float FloatType;
  
  std::vector<int> dims = {8,32,128,400,512,1024};

  for(int use_bias =0 ; use_bias <2; use_bias++){
    for(auto dim : dims){
      Matrix<FloatType> weights(dim, dim);
      Vector<FloatType> bias(dim);
      Vector<FloatType> derivs(dim*(dim+1));
      uniformRandom(weights,rng);
      uniformRandom(bias,rng);
      uniformRandom(derivs,rng);

      Matrix<FloatType> wcp(weights);
      Vector<FloatType> bcp(bias);
      step_v1(0, derivs, FloatType(1e-3), wcp, bcp, use_bias);

      step_v2(0, derivs, FloatType(1e-3), weights, bias, use_bias);

      assert(abs_near(wcp,weights,FloatType(1e-5),true));
      assert(abs_near(bcp,bias,FloatType(1e-5),true));
      
      double mu, sigma;
    
      profileStart();
      benchmark(mu, sigma, 100, 1, [&]{
	step_v2(0, derivs, FloatType(1e-3), weights, bias, use_bias);
      }, []{});
      profileStop();

      double mu_orig, sigma_orig;
      benchmark(mu_orig, sigma_orig, 100, 1, [&]{
	step_v1(0, derivs, FloatType(1e-3), weights, bias, use_bias);
      }, []{});
      
      std::cout << "dimenson: " << dim << "\tuse_bias: " << use_bias << "\tresult: " << mu/1e-6 << " +/- " << sigma/1e-6 << "us vs orig " << mu_orig/1e-6 << " +/- " << sigma_orig/1e-6 << std::endl;
    }
  }
  return 0;
}
