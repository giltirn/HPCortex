#include<HPCortex.hpp>
#include<Testing.hpp>

void benchmarkMatrixDNN(){
  std::mt19937 rng(1234);
  
  int data_size = 1000;
  int batch_size = 32;
  
  Matrix<float> weight(data_size,data_size,0.);
  Vector<float> bias(data_size,1.);
  auto m = dnn_layer(weight, bias, ReLU<float>(), input_layer<float>());
  
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

  // int contract_dim_max = 2;
  // std::vector<int> matrix_dims = { 2, 5, 8, 16, 64, 256, 512, 1024 };
  // std::vector<int> batch_sizes = {1, 5, 8, 16, 32, 64};
  // std::vector<int> other_dim_sizes = { 2, 5, 8, 64, 128, 256 };

  int contract_dim_max = 1;
  std::vector<int> matrix_dims = { 512 };
  std::vector<int> batch_sizes = { 64};
  std::vector<int> other_dim_sizes = { 64 };

  
  for(int contract_dim = 0; contract_dim< contract_dim_max; contract_dim++){
    for(int other_dim_size : other_dim_sizes){
      for(int matrix_dim : matrix_dims){
	for(int batch_size : batch_sizes){
	  std::cout << "contract_dim:" << contract_dim << " other_dim_size:" << other_dim_size << " matrix_dim:" << matrix_dim << " batch_size:" << batch_size << std::endl;
	  
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

	  auto m = batch_tensor_dnn_layer<3>(a, y, contract_dim, ReLU<float>(), input_layer<float, Tensor<float,3> >());

	  Tensor<float,3> got;
  
	  double mu, sigma;

	  benchmark(mu, sigma, 300, 1, [&]{
	    got = m.value(x);
	  }, []{});
	  double Mflops_mu = double(m.FLOPS(0))/1e6/mu;
	  double Mflops_sigma = double(m.FLOPS(0))/1e6/mu/mu * sigma;
	  
	  std::cout << "value: " << mu/1e-6 << "+-" << sigma/1e-6 << "us, " << Mflops_mu << "+-" << Mflops_sigma << " Mflops" << std::endl;

	  Tensor<float,3> above(tsize);
	  Vector<float> deriv(m.nparams());

	  benchmark(mu, sigma, 300, 1,
		    [&]{
		      m.deriv(deriv,0,std::move(above));
		    },
		    [&]{
		      above = Tensor<float,3>(tsize);
		      m.value(x);
		    }  );

	  Mflops_mu = double(m.FLOPS(1))/1e6/mu;
	  Mflops_sigma = double(m.FLOPS(1))/1e6/mu/mu * sigma;
	  
	  std::cout << "deriv: " << mu/1e-6 << "+-" << sigma/1e-6 << "us, "  << Mflops_mu << "+-" << Mflops_sigma << " Mflops" << std::endl;
	}
      }
    }
  }
}
  
void benchmarkCompareMatrixTensorDNN(){
  std::mt19937 rng(1234);

  int contract_dim = 0;
  std::vector<int> matrix_dims = { 2, 5, 8, 16, 64, 256, 512, 1024 };
  std::vector<int> batch_sizes = {1, 5, 8, 16, 32, 64};

  // std::vector<int> matrix_dims = { 1024 };
  // std::vector<int> batch_sizes = { 64};
  
  for(auto matrix_dim : matrix_dims){
    for(auto batch_size : batch_sizes){
      std::cout << "matrix_dim:" << matrix_dim << " batch_size:" << batch_size << std::endl;
      
      Matrix<float> a(matrix_dim, matrix_dim);
      uniformRandom(a,rng);
      
      Vector<float> y(matrix_dim);
      uniformRandom(y,rng);

      Matrix<float> x(matrix_dim,batch_size);
      uniformRandom(x, rng);

      auto mt = batch_tensor_dnn_layer<2>(a, y, contract_dim, ReLU<float>(), input_layer<float>());
      auto mm = dnn_layer(a, y, ReLU<float>(), input_layer<float>());
       
      double mum, sigmam;
      double mut, sigmat;

#if 1      
      Matrix<float> gotm;    
      benchmark(mum, sigmam, 300, 1, [&]{
	gotm = mm.value(x);
      }, []{});

      Matrix<float> gott;
      benchmark(mut, sigmat, 300, 1, [&]{
	gott = mt.value(x);
      }, []{});

      assert(abs_near(gotm,gott,1e-4f,true));
      
      std::cout << "value --  matrix-version:" << mum/1e-6 << "+-" << sigmam/1e-6 << "us  tensor-version:" << mut/1e-6 << "+-" << sigmat/1e-6 << "us" << std::endl;
#endif

      
      Matrix<float> above(matrix_dim, batch_size);
      uniformRandom(above,rng);

      //Test before benchmark
      Matrix<float> belowm, belowt;
      Vector<float> derivm(mm.nparams(),0.), derivt(mt.nparams(),0.);
#if 1
      mm.value(x);
      mm.deriv(derivm,0,Matrix<float>(above), &belowm);
      mt.value(x);
      mt.deriv(derivt,0,Matrix<float>(above), &belowt);
      assert(abs_near(derivm,derivt,1e-4f,true));
      assert(abs_near(belowm,belowt,1e-4f,true));      
      
      benchmark(mum, sigmam, 300, 1,
		[&]{
		  mm.deriv(derivm,0,std::move(above));
		},
		[&]{
		  above = Matrix<float>(matrix_dim, batch_size);
		  mm.value(x);
		}  );
#endif
      
      benchmark(mut, sigmat, 300, 1,
		[&]{
		  profileStart();
		  mt.deriv(derivt,0,std::move(above));
		  profileStop();
		},
		[&]{
		  above = Matrix<float>(matrix_dim, batch_size);
		  mt.value(x);
		}  );
      
      std::cout << "deriv --  matrix-version:" << mum/1e-6 << "+-" << sigmam/1e-6 << "us  tensor-version:" << mut/1e-6 << "+-" << sigmat/1e-6 << "us" << std::endl;
    }
  }
}




int main(int argc, char** argv){
  initialize(argc,argv);
  benchmarkTensorDNN();
  //benchmarkCompareMatrixTensorDNN();
  return 0;
}
