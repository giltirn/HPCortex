#include<HPCortex.hpp>
#include<Testing.hpp>

#ifdef USE_CUDA

//X_{..., i, ..., b}A_{ij} 
template<typename FloatType,int Dim>
Tensor<FloatType,Dim> matrixBatchTensorContractRightBase(const Tensor<FloatType,Dim> &X, const Matrix<FloatType> &A, const int contract_dim){
  int out_dims[Dim];
  memcpy(out_dims,X.sizeArray(),Dim*sizeof(int));
  out_dims[contract_dim] = A.size(1);

  assert(X.size(contract_dim) == A.size(0));
  assert(contract_dim != Dim-1); //not the batch dimension

  size_t other_size = 1;
  for(int d=0;d<Dim-1;d++)
    if(d!= contract_dim)
      other_size *= X.size(d);

  int batch_size = X.size(Dim-1);
  
  int sizei = A.size(0);
  int sizej = A.size(1);

  size_t stride = tensorDimensionStride<Dim>(contract_dim, X.sizeArray());
  
  Tensor<FloatType,Dim> out(out_dims); 
  {
    autoView(X_v, X, DeviceRead);
    autoView(A_v, A, DeviceRead);
    autoView(out_v, out, DeviceWrite);

    accelerator_for3d(b, batch_size, j, sizej, o, other_size, 1, { 
	size_t out_poff = batchTensorDimensionBaseLin<Dim>(contract_dim, b, o, out_v.sizeArray());
	size_t X_poff =  batchTensorDimensionBaseLin<Dim>(contract_dim, b, o, X_v.sizeArray());

	FloatType* X_p =  X_v.data() + X_poff;
	FloatType v = (*X_p)*A_v(0,j);
	X_p += stride;
	
	for(int i=1;i<sizei;i++){
	  v +=(*X_p)*A_v(i,j);
	  X_p += stride;
	}
	out_v.data()[out_poff + j*stride] = v;
      });
  }
  return out;
}

template<typename FloatType,int Dim>
Tensor<FloatType,Dim> matrixBatchTensorContractRight_v2(const Tensor<FloatType,Dim> &X, const Matrix<FloatType> &A, const int contract_dim){
  int out_dims[Dim];
  memcpy(out_dims,X.sizeArray(),Dim*sizeof(int));
  out_dims[contract_dim] = A.size(1);

  assert(X.size(contract_dim) == A.size(0));
  assert(contract_dim != Dim-1); //not the batch dimension

  size_t other_size = 1;
  for(int d=0;d<Dim-1;d++)
    if(d!= contract_dim)
      other_size *= X.size(d);

  int batch_size = X.size(Dim-1);
  
  int sizei = A.size(0);
  int sizej = A.size(1);

  size_t stride = tensorDimensionStride<Dim>(contract_dim, X.sizeArray());
  
  Tensor<FloatType,Dim> out(out_dims); 
  {
    autoView(X_v, X, DeviceRead);
    autoView(A_v, A, DeviceRead);
    autoView(out_v, out, DeviceWrite);
    int iblocksz = 64;
    int iblocks = (sizei + iblocksz -1)/iblocksz;

    constexpr int oblocksz = 4;
    int oblocks = (other_size + oblocksz - 1)/oblocksz;
    
    accelerator_for3d_shm(b, batch_size, j, sizej, bo, oblocks,    1, (iblocksz*sizeof(FloatType) + 2*oblocksz*sizeof(size_t)  ), {
	extern __shared__ char shared[];
	size_t* off_X_o = (size_t*)shared;
	size_t* off_out_o = (size_t*)(shared + oblocksz*sizeof(size_t));
	FloatType* shared_A = (FloatType*)(shared + 2*oblocksz*sizeof(size_t));
	
	//Compute offsets
	int oblocksz_actual = other_size - bo*oblocksz < oblocksz ? other_size - bo*oblocksz : oblocksz;
	int oo=b;
	while(oo < oblocksz_actual){
	  int o = oo + bo*oblocksz;
	  off_X_o[oo] = batchTensorDimensionBaseLin<Dim>(contract_dim, 0, o, X_v.sizeArray());
	  off_out_o[oo] = batchTensorDimensionBaseLin<Dim>(contract_dim, 0, o, out_v.sizeArray());
	  oo += batch_size;
	}
	acceleratorSynchronizeBlock();
	
	FloatType out_ojb[oblocksz] = {0.};
	int ibase = 0;
	for(int bi=0;bi<iblocks;bi++){
	  int iblocksz_actual = sizei - ibase < iblocksz ? sizei - ibase : iblocksz;

	  //Load A_v(:,j) in parallel
	  int ii=b;
	  while(ii<iblocksz_actual){
	    shared_A[ii] = A_v(ibase+ii,j);
	    ii+= batch_size;
	  }
	  acceleratorSynchronizeBlock();

	  for(int oo=0;oo<oblocksz_actual;oo++){	  
	    FloatType *X_p = X_v.data() + off_X_o[oo] + b + stride*ibase;
	  
	    for(int ii=0;ii<iblocksz_actual;ii++){
	      out_ojb[oo] += shared_A[ii] * (*X_p);
	      X_p += stride;
	    }
	  }

	  acceleratorSynchronizeBlock();
	  
	  ibase += iblocksz;
	}

	for(int oo=0;oo<oblocksz_actual;oo++){	
	  out_v.data()[off_out_o[oo] + b + stride*j] = out_ojb[oo];
	}
	
      });
  }
  return out;
}



int main(int argc, char** argv){
  initialize(argc,argv);
  std::mt19937 rng(1234);

  std::vector<int> matrix_dims = { 2, 5,  8, 16, 64, 256, 512 };
  std::vector<int> batch_sizes = {1, 5, 8 , 16, 32, 64};
  std::vector<int> other_dim_sizes = { 2, 5, 8, 16, 64, 256, 512 };

  // std::vector<int> matrix_dims = { 512 };
  // std::vector<int> batch_sizes = { 64};
  // std::vector<int> other_dim_sizes = { 64 };
  
  for(int contract_dim=0; contract_dim<2; contract_dim++){
    for(auto other_dim_size : other_dim_sizes){
      for(auto matrix_dim : matrix_dims){
	for(auto batch_size : batch_sizes){
	  Matrix<float> a(matrix_dim, matrix_dim);
	  uniformRandom(a,rng);

	  int tsize[3];
	  tsize[2] = batch_size;
	  tsize[contract_dim] = matrix_dim;
	  tsize[1-contract_dim] = other_dim_size;
	  
	  Tensor<float,3> x(tsize);
	  uniformRandom(x, rng);
	  
	  double mu, sigma;
	  
	  Tensor<float,3> c;
	  benchmark(mu, sigma, 100, 1, [&]{
	    profileStart();
	    c = matrixBatchTensorContractRight_v2(x,a,contract_dim);
	    profileStop();
	  }, []{});

	  Tensor<float,3> ctest = matrixBatchTensorContractRightBase(x,a,contract_dim);
	  assert(abs_near(c,ctest,1e-4f,true));
	  
	  double mu_base, sigma_base;
	  benchmark(mu_base, sigma_base, 100, 1, [&]{
	    profileStart();
	    c = matrixBatchTensorContractRightBase(x,a,contract_dim);
	    profileStop();
	  }, []{});

	  std::cout << "contract_dim:" << contract_dim << "\tother_dim_size:" << other_dim_size << "\tmatrix_dim:" << matrix_dim << "\tbatch_size:" << batch_size << "\tresult: " << mu/1e-6 << "+-" << sigma/1e-6 << "us" << " base: " << mu_base/1e-6 << "+-" << sigma_base/1e-6 << "us" << std::endl;
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
