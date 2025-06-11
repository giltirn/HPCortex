#include<HPCortex.hpp>
#include<Testing.hpp>

#ifdef USE_CUDA

//A_{ij} X_{..., j, ..., b}  + Y_i
template<typename FloatType,int Dim>
Tensor<FloatType,Dim> matrixBatchTensorAxpyBase(const Matrix<FloatType> &A, const Tensor<FloatType,Dim> &X, const Vector<FloatType> &Y, const int contract_dim){
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
  
  Tensor<FloatType,Dim> out(out_dims); 
  {
    autoView(X_v, X, DeviceRead);
    autoView(Y_v, Y, DeviceRead);
    autoView(A_v, A, DeviceRead);
    autoView(out_v, out, DeviceWrite);
    
    int _sizej = A.size(1);
    int _sizei = A.size(0);
    int _contract_dim = contract_dim;
    size_t _stride = tensorDimensionStride<Dim>(contract_dim, X.sizeArray());
        
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
	
	out_v.data()[off_out + _stride*i] = out_oib  + Y_v(i);
      });
  }
  return out;
}

//this version is slower
template<typename FloatType,int Dim>
Tensor<FloatType,Dim> matrixBatchTensorAxpy_v2(const Matrix<FloatType> &A, const Tensor<FloatType,Dim> &X, const Vector<FloatType> &Y, const int contract_dim){
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
  
  Tensor<FloatType,Dim> out(out_dims); 
  {
    autoView(X_v, X, DeviceRead);
    autoView(Y_v, Y, DeviceRead);
    autoView(A_v, A, DeviceRead);
    autoView(out_v, out, DeviceWrite);
    
    int _sizej = A.size(1);
    int _sizei = A.size(0);
    int _contract_dim = contract_dim;
    size_t _stride = tensorDimensionStride<Dim>(contract_dim, X.sizeArray());

    int iblocksz = 16;
    int jblocksz = 24;
    int bblocksz = 8;
    int iblocks = (_sizei + iblocksz - 1)/iblocksz;
    int jblocks = (_sizej + jblocksz - 1)/jblocksz;
    int bblocks = (batch_size + bblocksz - 1)/bblocksz;

    int group_threads = iblocksz * bblocksz;
    
    accelerator_for_1_3_shm(thr, group_threads, bblock, bblocks, iblock, iblocks, o, other_size,    1, ( sizeof(FloatType) + iblocksz*jblocksz*sizeof(FloatType) + jblocksz*bblocksz*sizeof(FloatType) ),  {
	extern __shared__ FloatType shared_A[];
	FloatType *shared_X = shared_A + iblocksz*jblocksz + 1;
	
	int ibase = iblock * iblocksz;
	int bbase = bblock * bblocksz;

	FloatType *X_p = X_v.data() + batchTensorDimensionBaseLin<Dim>(_contract_dim, 0, o, X_v.sizeArray()) + bbase;
	
	int iblocksz_actual = _sizei - ibase < iblocksz ?  _sizei - ibase : iblocksz;
	int bblocksz_actual = batch_size - bbase < bblocksz ? batch_size - bbase : bblocksz;

	int bb = thr % bblocksz;
	int ii = thr / bblocksz;

	FloatType out_oib = 0.;

	for(int blockj=0; blockj < jblocks; blockj++){
	  int jbase = blockj * jblocksz;

	  int jblocksz_actual = _sizej - jbase < jblocksz ?  _sizej - jbase : jblocksz;
	  
	  //parallel load of A block
	  int iijj = thr;
	  while(iijj < iblocksz_actual*jblocksz_actual){
	    int ajj = iijj % jblocksz_actual;
	    int aii = iijj / jblocksz_actual;
	    shared_A[iijj] = A_v(aii + ibase,ajj + jbase);
	    iijj += group_threads;
	  }
	  
	  //parallel load of X block
	  int jjbb = thr;
	  while(jjbb < jblocksz_actual*bblocksz_actual){ //jjbb = xbb + bblocksz_actual * xjj
	    int xbb = jjbb % bblocksz_actual;
	    int xjj = jjbb / bblocksz_actual;

	    //shared_X[jjbb] = *(X_v.data() + off_X + (xjj+jbase)*_stride + xbb + bbase);
	    shared_X[jjbb] = *(X_p + xjj*_stride + xbb);
	    jjbb += group_threads;
	  }
	  X_p += jblocksz * _stride;
	    
	  acceleratorSynchronizeBlock();

	  FloatType *sA = shared_A + ii*jblocksz_actual;
	  FloatType *sX = shared_X + bb;
	  for(int jj=0;jj<jblocksz_actual;jj++){
	    out_oib += (*sA) * (*sX);
	    sA ++ ; 
	    sX += bblocksz_actual;
	  }
	  
	  acceleratorSynchronizeBlock();
	}
	int b = bbase + bb;
	int i = ibase + ii;
	
	if(b < batch_size && i < _sizei)
	  out_v.data()[batchTensorDimensionBaseLin<Dim>(_contract_dim, 0, o, out_v.sizeArray()) + i*_stride + b] = out_oib + Y_v(i);
	
      });
  }
  return out;
}

//this version is considerably faster
template<typename FloatType,int Dim>
Tensor<FloatType,Dim> matrixBatchTensorAxpy_v3(const Matrix<FloatType> &A, const Tensor<FloatType,Dim> &X, const Vector<FloatType> &Y, const int contract_dim){
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
  
  Tensor<FloatType,Dim> out(out_dims); 
  {
    autoView(X_v, X, DeviceRead);
    autoView(Y_v, Y, DeviceRead);
    autoView(A_v, A, DeviceRead);
    autoView(out_v, out, DeviceWrite);
    
    int _sizej = A.size(1);
    int _sizei = A.size(0);
    int _contract_dim = contract_dim;
    size_t _stride = tensorDimensionStride<Dim>(contract_dim, X.sizeArray());

    int jblocksz = 64;
    int jblocks = (_sizej + jblocksz -1)/jblocksz;

    constexpr int oblocksz = 4;
    int oblocks = (other_size + oblocksz - 1)/oblocksz;
    
    accelerator_for3d_shm(b, batch_size, i, _sizei, bo, oblocks,    1, (jblocksz*sizeof(FloatType) + 2*oblocksz*sizeof(size_t)  ), {
	extern __shared__ char shared[];
	size_t* off_X_o = (size_t*)shared;
	size_t* off_out_o = (size_t*)(shared + oblocksz*sizeof(size_t));
	FloatType* shared_A = (FloatType*)(shared + 2*oblocksz*sizeof(size_t));
	
	//Compute offsets
	int oblocksz_actual = other_size - bo*oblocksz < oblocksz ? other_size - bo*oblocksz : oblocksz;
	int oo=b;
	while(oo < oblocksz_actual){
	  int o = oo + bo*oblocksz;
	  off_X_o[oo] = batchTensorDimensionBaseLin<Dim>(_contract_dim, 0, o, X_v.sizeArray());
	  off_out_o[oo] = batchTensorDimensionBaseLin<Dim>(_contract_dim, 0, o, out_v.sizeArray());
	  oo += batch_size;
	}
	acceleratorSynchronizeBlock();
	
	FloatType out_oib[oblocksz] = {0.};
	int jbase = 0;
	for(int bj=0;bj<jblocks;bj++){
	  int jblocksz_actual = _sizej - jbase < jblocksz ? _sizej - jbase : jblocksz;

	  //Load A_v(i,:) in parallel
	  int jj=b;
	  while(jj<jblocksz_actual){
	    shared_A[jj] = A_v(i,jbase+jj);
	    jj+= batch_size;
	  }
	  acceleratorSynchronizeBlock();

	  for(int oo=0;oo<oblocksz_actual;oo++){	  
	    FloatType *X_p = X_v.data() + off_X_o[oo] + b + _stride*jbase;
	  
	    for(int jj=0;jj<jblocksz_actual;jj++){
	      out_oib[oo] += shared_A[jj] * (*X_p);
	      X_p += _stride;
	    }
	  }

	  acceleratorSynchronizeBlock();
	  
	  jbase += jblocksz;
	}

	for(int oo=0;oo<oblocksz_actual;oo++){	
	  out_v.data()[off_out_o[oo] + b + _stride*i] = out_oib[oo]  + Y_v(i);
	}
	
      });
  }

  return out;
}



template<typename FloatType,int Dim>
Tensor<FloatType,Dim> matrixBatchTensorAxpy_v4(const Matrix<FloatType> &A, const Tensor<FloatType,Dim> &X, const Vector<FloatType> &Y, const int contract_dim){
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
  
  Tensor<FloatType,Dim> out(out_dims); 
  {
    autoView(X_v, X, DeviceRead);
    autoView(Y_v, Y, DeviceRead);
    autoView(A_v, A, DeviceRead);
    autoView(out_v, out, DeviceWrite);
    
    int _sizej = A.size(1);
    int _sizei = A.size(0);
    int _contract_dim = contract_dim;
    size_t _stride = tensorDimensionStride<Dim>(contract_dim, X.sizeArray());

    int jblocksz = 64;
    int jblocks = (_sizej + jblocksz -1)/jblocksz;

    constexpr int oblocksz = 4;
    int oblocks = (other_size + oblocksz - 1)/oblocksz;
    
    accelerator_for3d_shm(b, batch_size, i, _sizei, bo, oblocks,    1, (jblocksz*sizeof(FloatType) + 2*oblocksz*sizeof(size_t)  ), {
	extern __shared__ char shared[];
	size_t* off_X_o = (size_t*)shared;
	size_t* off_out_o = (size_t*)(shared + oblocksz*sizeof(size_t));
	FloatType* shared_A = (FloatType*)(shared + 2*oblocksz*sizeof(size_t));
	
	//Compute offsets
	int oblocksz_actual = other_size - bo*oblocksz < oblocksz ? other_size - bo*oblocksz : oblocksz;
	int oo=b;
	while(oo < oblocksz_actual){
	  int o = oo + bo*oblocksz;
	  off_X_o[oo] = batchTensorDimensionBaseLin<Dim>(_contract_dim, 0, o, X_v.sizeArray());
	  off_out_o[oo] = batchTensorDimensionBaseLin<Dim>(_contract_dim, 0, o, out_v.sizeArray()) + _stride*i;
	  oo += batch_size;
	}
	acceleratorSynchronizeBlock();
	
	FloatType out_oib[oblocksz] = {0.};
	int jbase = 0;
	for(int bj=0;bj<jblocks;bj++){
	  int jblocksz_actual = _sizej - jbase < jblocksz ? _sizej - jbase : jblocksz;

	  //Load A_v(i,:) in parallel
	  int jj=b;
	  while(jj<jblocksz_actual){
	    shared_A[jj] = A_v(i,jbase+jj);
	    jj+= batch_size;
	  }
	  acceleratorSynchronizeBlock();

	  for(int oo=0;oo<oblocksz_actual;oo++){	  
	    FloatType *X_p = X_v.data() + off_X_o[oo] + b + _stride*jbase;
	  
	    for(int jj=0;jj<jblocksz_actual;jj++){
	      out_oib[oo] += shared_A[jj] * (*X_p);
	      X_p += _stride;
	    }
	  }

	  acceleratorSynchronizeBlock();
	  
	  jbase += jblocksz;
	}

	for(int oo=0;oo<oblocksz_actual;oo++){	
	  out_v.data()[off_out_o[oo] + b] = out_oib[oo]  + Y_v(i);
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

	  Vector<float> y(matrix_dim);
	  uniformRandom(y,rng);
	  
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
	    c = matrixBatchTensorAxpy_v4(a,x,y,contract_dim);
	    profileStop();
	  }, []{});

	  Tensor<float,3> ctest = matrixBatchTensorAxpyBase(a,x,y,contract_dim);
	  assert(abs_near(c,ctest,1e-4f,true));
	  
	  double mu_base, sigma_base;
	  benchmark(mu_base, sigma_base, 100, 1, [&]{
	    profileStart();
	    c = matrixBatchTensorAxpy_v3(a,x,y,contract_dim);
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
