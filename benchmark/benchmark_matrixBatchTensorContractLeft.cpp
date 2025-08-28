#include<HPCortex.hpp>
#include<Testing.hpp>

#ifdef USE_GPU

template<typename FloatType,int Dim>
Tensor<FloatType,Dim> matrixBatchTensorContractLeftBase(const Matrix<FloatType> &A, const Tensor<FloatType,Dim> &X, const int contract_dim){
  int out_dims[Dim];
  memcpy(out_dims,X.sizeArray(),Dim*sizeof(int));
  out_dims[contract_dim] = A.size(0);

  assert(X.size(contract_dim) == A.size(1));
  assert(contract_dim != Dim-1); //not the batch dimension

  size_t other_size = 1;
  for(int d=0;d<Dim-1;d++)
    if(d!= contract_dim)
      other_size *= X.size(d);

  int batch_size = X.size(Dim-1);

  int _sizej = A.size(1);
  int _sizei = A.size(0);
  int _contract_dim = contract_dim;
  size_t _stride = tensorDimensionStride<Dim>(contract_dim, X.sizeArray());

  Tensor<FloatType,Dim> out(out_dims); 
  {
    autoView(X_v, X, DeviceRead);
    autoView(A_v, A, DeviceRead);
    autoView(out_v, out, DeviceWrite);
  
    accelerator_for3d(b, batch_size, i, _sizei, o, other_size,    1, {
	size_t off_X = batchTensorDimensionBaseLin<Dim>(_contract_dim, b, o, X_v.sizeArray());
	size_t off_out = batchTensorDimensionBaseLin<Dim>(_contract_dim, b, o, out_v.sizeArray());
	FloatType *X_p = X_v.data() + off_X;
	
	FloatType out_oib = A_v(i,0) * (*X_p);
	X_p += _stride;
	
	for(int j=1;j<_sizej;j++){
	  out_oib += A_v(i,j) * (*X_p);
	  X_p += _stride;
	}	  
	
	out_v.data()[off_out + _stride*i] = out_oib;
      });
  }
  return out;
}

template<typename FloatType,int Dim>
Tensor<FloatType,Dim> matrixBatchTensorContractLeft_v2(const Matrix<FloatType> &A, const Tensor<FloatType,Dim> &X, const int contract_dim){
#ifdef USE_BLAS
  Vector<FloatType> Xvec = transformBatchVector(contract_dim,X);

  int sizei = A.size(0);  //contract over
  int sizej = A.size(1);

  int out_dims[Dim];
  memcpy(out_dims,X.sizeArray(),Dim*sizeof(int));
  out_dims[contract_dim] = sizei;
  
  size_t other_size = 1; //dimensions other than the dimensions along which the vector resides
  for(int d=0;d<Dim;d++)
    if(d!= contract_dim)
      other_size *= X.size(d);
     
  Vector<FloatType> ovec(other_size * sizei);

  {
    autoView(Xvec_v, Xvec, DeviceRead);
    autoView(A_v, A, DeviceRead);
    autoView(ovec_v, ovec, DeviceWrite);

    //c_oj = A_ij x_oj = x_oj A_ji^T
    GEMM(NoTranspose, Transpose,
	   other_size, sizei, sizej,
	   FloatType(1.0),
	   Xvec_v.data(), sizej,
	   A_v.data(), sizej,	   
	   FloatType(0.0),
	   ovec_v.data());
  }
  
  Tensor<FloatType,Dim> out(out_dims);
  untransformBatchVector(contract_dim, out, ovec);  
  return out;
#else
  return matrixBatchTensorContractLeft(A,X,contract_dim);
#endif
  
}

int main(int argc, char** argv){
  initialize(argc,argv);
  std::mt19937 rng(1234);

 
  std::vector<int> other_dim_sizes = { 2, 5, 8, 16, 64, 256, 512 };
  std::vector<int> matrix_dims = { 2, 5,  8, 16, 64, 256, 512 };
  std::vector<int> batch_sizes = {1, 5, 8 , 16, 32, 64};


  // std::vector<int> other_dim_sizes = { 64 };
  // std::vector<int> matrix_dims = { 512 };
  // std::vector<int> batch_sizes = { 64};

  // std::vector<int> other_dim_sizes = { 256 };
  // std::vector<int> matrix_dims = { 256 };
  // std::vector<int> batch_sizes = { 32 };

  
  
  // std::vector<int> other_dim_sizes = { 3 };
  // std::vector<int> matrix_dims = { 4 };
  // std::vector<int> batch_sizes = { 2 };
 

    
  for(int contract_dim=0; contract_dim<2; contract_dim++){
    for(auto other_dim_size : other_dim_sizes){
      for(auto matrix_dim : matrix_dims){
	for(auto batch_size : batch_sizes){
	  int tsize[3];
	  tsize[2] = batch_size;
	  tsize[contract_dim] = matrix_dim;
	  tsize[1-contract_dim] = other_dim_size;

	  {
	    Matrix<double> a(matrix_dim, matrix_dim);
	    uniformRandom(a,rng);
	    
	    Tensor<double,3> x(tsize);
	    uniformRandom(x,rng);
	    Tensor<double,3> cexpect = matrixBatchTensorContractLeftBase(a,x,contract_dim);
	    Tensor<double,3> cgot = matrixBatchTensorContractLeft_v2(a,x,contract_dim);
	    assert(abs_near(cexpect,cgot,1e-6,true));
	  }

	  Matrix<float> a(matrix_dim, matrix_dim);
	  uniformRandom(a,rng);
	  
	  Tensor<float,3> x(tsize);
	  uniformRandom(x, rng);
	  
	  double mu, sigma;
	  
	  Tensor<float,3> c;
	  benchmark(mu, sigma, 100, 1, [&]{
	    profileStart();
	    c = matrixBatchTensorContractLeft_v2(a,x,contract_dim);
	    profileStop();
	  }, []{});
	  
	  double mu_base, sigma_base;
	  benchmark(mu_base, sigma_base, 100, 1, [&]{
	    profileStart();
	    c = matrixBatchTensorContractLeft(a,x,contract_dim);
	    profileStop();
	  }, []{});

	  //x_oi A_ij 
	  
	  size_t FLOPS = size_t(other_dim_size)*size_t(batch_size)*size_t(matrix_dim)*(1+2*(size_t(matrix_dim)-1) );
	  double Gflops = FLOPS/mu/1.e9, Gflops_base = FLOPS/mu_base/1.e9;	  
	  
	  std::cout << "contract_dim:" << contract_dim << "\tother_dim_size:" << other_dim_size << "\tmatrix_dim:" << matrix_dim << "\tbatch_size:" << batch_size << "\tresult: " << mu/1e-6 << "+-" << sigma/1e-6 << "us (" << Gflops << " Gflops) base: " << mu_base/1e-6 << "+-" << sigma_base/1e-6 << "us (" << Gflops_base << " Gflops)" << std::endl;
	}
      }
    }
  }

  
  return 0;
  
}


#else
int main(void){
  std::cout << "Benchmarks currently GPU-only" << std::endl;
  return 0;
}
#endif
