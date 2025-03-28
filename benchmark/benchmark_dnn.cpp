#include<HPCortex.hpp>
#include<Testing.hpp>

int main(int argc, char** argv){
  initialize(argc,argv);
  std::mt19937 rng(1234);
  
  int data_size = 1000;
  int batch_size = 32;
  
  Matrix<float> weight(data_size,data_size,0.);
  Vector<float> bias(data_size,1.);
  auto m = dnn_layer(input_layer<float>(), weight, bias, ReLU<float>());
  
  Matrix<float> data(data_size,batch_size,0.);
  Matrix<float> got;
  
  double mu, sigma;

  benchmark(mu, sigma, 300, 1, [&]{
    got = m.value(data);
  }, []{});

  std::cout << "value: " << mu/1e-6 << "us " << sigma/1e-6 << "us" << std::endl;

  Matrix<float> above;
  Vector<float> deriv(m.nparams());

  benchmark(mu, sigma, 300, 1,
	    [&]{
	      m.deriv(deriv,0,std::move(above));
	    },
	    [&]{
	      above = Matrix<float>(data_size, batch_size, 0.);
	      m.value(data);
	    }  );

  std::cout << "deriv: " << mu/1e-6 << "us " << sigma/1e-6 << "us" << std::endl;
  
  return 0;
}
