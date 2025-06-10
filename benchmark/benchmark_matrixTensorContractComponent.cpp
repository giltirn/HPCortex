#include<HPCortex.hpp>
#include<Testing.hpp>

int main(int argc, char** argv){
  initialize(argc,argv);

  std::mt19937 rng(1234);

  std::vector<int> matrix_dims = { 2, 5, 8, 16, 64, 256, 512, 1024 };
  std::vector<int> batch_sizes = {1, 5, 8, 16, 32, 64};
  std::vector<int> other_dim_sizes = {2,5,12,24};
  
  for(int matrix_dim: matrix_dims){
    for(int other_dim_size : other_dim_sizes){
      for(int batch_size : batch_sizes){
	std::cout << "matrix_dim:" << matrix_dim << " other_dim_size:"<< other_dim_size << " batch_size:" << batch_size << std::endl;
	
	Matrix<float> a(matrix_dim, matrix_dim);
	uniformRandom(a,rng);
		  
	int tsize[4];
	tsize[0] = other_dim_size;
	tsize[1] = other_dim_size;
	tsize[2] = matrix_dim;
	tsize[3] = batch_size;	
		  
	Tensor<float,4> x(tsize);
	uniformRandom(x, rng);

	MatrixTensorContractComponent<float,4> cpt(a);

	Tensor<float,4> got;
	
	double mu, sigma;
	
	benchmark(mu, sigma, 100, 1, [&]{
	  got = cpt.value(x);
	}, []{});

	std::cout << "value: " << mu/1e-6 << "+-" << sigma/1e-6 << "us" << std::endl;

	Tensor<float,4> above(tsize);
	Tensor<float,4> below(tsize);
	Vector<float> deriv(cpt.nparams());

	benchmark(mu, sigma, 100, 1,
		  [&]{
		    cpt.deriv(deriv,0,std::move(above),below);
		  },
		  [&]{
		    above = Tensor<float,4>(tsize);
		    cpt.value(x);
		  }  );

	std::cout << "deriv: " << mu/1e-6 << "+-" << sigma/1e-6 << "us" << std::endl;
      }
    }
  }
  return 0;
}
