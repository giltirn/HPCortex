#include<HPCortex.hpp>
#include<Testing.hpp>

void benchmarkMatrixDNN(){
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
}

void benchmarkTensorDNN(){
  std::mt19937 rng(1234);

  int contract_dim = 0;
  int matrix_dim = 512;
  int batch_size = 64;
  int other_dim_size= 64;
  
  Matrix<float> a(matrix_dim, matrix_dim);
  uniformRandom(a,rng);

  Vector<float> y(matrix_dim);
  uniformRandom(y,rng);
	  
  int tsize[3];
  tsize[2] = batch_size;
  tsize[contract_dim] = matrix_dim;
  tsize[1-contract_dim] = other_dim_size;
	  
  Tensor<float,3> x(tsize);
  uniformRandom(x, rng);

  auto m = batch_tensor_dnn_layer<3>(input_layer<float, Tensor<float,3> >(), a, y, contract_dim, ReLU<float>());

  Tensor<float,3> got;
  
  double mu, sigma;

  // benchmark(mu, sigma, 300, 1, [&]{
  //   got = m.value(x);
  // }, []{});

  // std::cout << "value: " << mu/1e-6 << "us " << sigma/1e-6 << "us" << std::endl;

  Tensor<float,3> above(tsize);
  Vector<float> deriv(m.nparams());

  benchmark(mu, sigma, 3, 1,
	    [&]{
	      m.deriv(deriv,0,std::move(above));
	    },
	    [&]{
	      above = Tensor<float,3>(tsize);
	      m.value(x);
	    }  );

  std::cout << "deriv: " << mu/1e-6 << "us " << sigma/1e-6 << "us" << std::endl;
}
  


int main(int argc, char** argv){
  initialize(argc,argv);
  benchmarkTensorDNN();
  return 0;
}
