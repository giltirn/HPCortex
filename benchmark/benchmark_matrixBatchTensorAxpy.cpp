#include<HPCortex.hpp>
#include<Testing.hpp>

#ifdef USE_GPU

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
	FloatType *shared_A = (FloatType*)shared;
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

#ifdef USE_CUDA
template<typename lambda>  __global__
void LambdaApply(uint64_t num1, uint64_t num2, uint64_t num3, uint64_t block2, lambda Lambda)
{
   extern __shared__ char shared[];
   uint64_t iter1 = threadIdx.x;
   //uint64_t iter2 = threadIdx.y + block2*blockIdx.x  ;
   uint64_t iter2 = block2*blockIdx.x  ; 
   //uint64_t iter2 = blockIdx.x; //slow
   
   uint64_t iter3 = blockIdx.y;
   if ( (iter1 < num1) && (iter2 <num2) && (iter3 <num3) ) {   
   Lambda(iter1, iter2, iter3, shared);
   }
						
}
#define accelerator_for3d_shm_test( iter1, num1, iter2, num2, iter3, num3, block2, shm_size, ... ) \
     {									\
        if ( num1*num2*num3 ) {							\
          typedef uint64_t Iterator;					\
          auto lambda = [=] __device__					\
    	(Iterator iter1, Iterator iter2, Iterator iter3,char* shared) mutable { \
    		      __VA_ARGS__;					\
             };							\
          dim3 cu_threads(num1,1,1);						\
          dim3 cu_blocks (num2,num3,1);				\
          LambdaApply<<<cu_blocks,cu_threads,shm_size,computeStream>>>(num1,num2,num3,1,lambda); \
        }									\
        accelerator_barrier(dummy);			\
      }
#endif

//A_{ij} X_{..., j, ..., b}  + Y_i
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
    
    accelerator_for_3d_gen(1,2,shm( jblocksz*sizeof(FloatType) + 2*oblocksz*sizeof(size_t) ), b, batch_size, i, _sizei, bo, oblocks, {
	//accelerator_for3d_shm_test(b, batch_size, i, _sizei, bo, oblocks, 1, (jblocksz*sizeof(FloatType) + 2*oblocksz*sizeof(size_t)), {
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

	  //FloatType *Xpj = X_v.data() + b + _stride*jbase;
	  
	  for(int oo=0;oo<oblocksz_actual;oo++){	  
	    //FloatType *X_p = Xpj + off_X_o[oo];
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

//A_{ij} X_{..., j, ..., b}  + Y_i
template<typename FloatType,int Dim>
Tensor<FloatType,Dim> matrixBatchTensorAxpy_v5(const Matrix<FloatType> &A, const Tensor<FloatType,Dim> &X, const Vector<FloatType> &Y, const int contract_dim){
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
  
  Tensor<FloatType,Dim> out(out_dims,0.); 
  {
    autoView(X_v, X, DeviceRead);
    autoView(Y_v, Y, DeviceRead);
    autoView(A_v, A, DeviceRead);
    autoView(out_v, out, DeviceReadWrite);
    
    int _sizej = A.size(1);
    int _sizei = A.size(0);
    int _contract_dim = contract_dim;
    size_t _stride = tensorDimensionStride<Dim>(contract_dim, X.sizeArray());
    
    constexpr int iblocksz = 32;
    int iblocks = (_sizei + iblocksz -1)/iblocksz;
    
    int jblocksz = 32;
    int jblocks = (_sizej + jblocksz -1)/jblocksz;

    constexpr int oblocksz = 4;
    int oblocks = (other_size + oblocksz - 1)/oblocksz;
    
    accelerator_for_4d_gen(1,3,shm( iblocksz*jblocksz*sizeof(FloatType) + 2*oblocksz*sizeof(size_t) ), b, batch_size, bi, iblocks, bj, jblocks, bo, oblocks, {
	size_t* off_X_o = (size_t*)shared;
	size_t* off_out_o = (size_t*)(shared + oblocksz*sizeof(size_t));
	FloatType* shared_A = (FloatType*)(shared + 2*oblocksz*sizeof(size_t));
	
	//Compute offsets
	int oblocksz_actual = other_size - bo*oblocksz < oblocksz ? other_size - bo*oblocksz : oblocksz;
	int jbase = bj*jblocksz;
	int jblocksz_actual = _sizej - jbase < jblocksz ? _sizej - jbase : jblocksz;
	int ibase = bi*iblocksz;
	int iblocksz_actual = _sizei - ibase < iblocksz ? _sizei - ibase : iblocksz;

	int oo=b;
	while(oo < oblocksz_actual){
	  int o = oo + bo*oblocksz;
	  off_X_o[oo] = batchTensorDimensionBaseLin<Dim>(_contract_dim, 0, o, X_v.sizeArray());
	  off_out_o[oo] = batchTensorDimensionBaseLin<Dim>(_contract_dim, 0, o, out_v.sizeArray());
	  oo += batch_size;
	}
	
	//Load A_v(:,:) in parallel
	for(int ii=0;ii<iblocksz_actual;ii++){
	  int jj=b;
	  while(jj<jblocksz_actual){
	    shared_A[jj + jblocksz * ii] = A_v(ibase+ii,jbase+jj);
	    jj+= batch_size;
	  }
	}
	  
	acceleratorSynchronizeBlock();

	for(int oo=0;oo<oblocksz_actual;oo++){	  
	  FloatType *X_p = X_v.data() + off_X_o[oo] + b + _stride*jbase;
	  FloatType out_oib[iblocksz] = { FloatType(0.) };
	  	  
	  for(int jj=0;jj<jblocksz_actual;jj++){
	    FloatType X_ojb = *X_p;
	    X_p += _stride;
	    for(int ii=0;ii<iblocksz_actual;ii++)
	      out_oib[ii] += shared_A[jj + jblocksz*ii] * X_ojb;
	  }
	  for(int ii=0;ii<iblocksz_actual;ii++){
	    int i = ibase+ii;
	    atomicAdd(out_v.data() + off_out_o[oo] + b + i*_stride, out_oib[ii] + (bj==0 ? Y_v(i) : FloatType(0.)) );
	  }
	}
	
      });
  }
    
  return out;
}




//A_{ij} X_{..., j, ..., b}  + Y_i
template<typename FloatType,int Dim>
Tensor<FloatType,Dim> matrixBatchTensorAxpy_v6(const Matrix<FloatType> &A, const Tensor<FloatType,Dim> &X, const Vector<FloatType> &Y, const int contract_dim){
#ifdef USE_BLAS
  int sizei = A.size(0);
  int sizej = A.size(1);

  int out_dims[Dim];
  memcpy(out_dims,X.sizeArray(),Dim*sizeof(int));
  out_dims[contract_dim] = sizei;
  
  assert(X.size(contract_dim) == sizej);
  assert(contract_dim != Dim-1); //not the batch dimension

  size_t other_size = 1;
  for(int d=0;d<Dim;d++)
    if(d!= contract_dim)
      other_size *= X.size(d);

  size_t out_size_lin = other_size * sizei;
  
  //A_ij X'_oj = X'_oj A^T_ji -> out_oi
  Vector<FloatType> outv(out_size_lin); //out_oi
  {
    Vector<FloatType> Xvec = transformBatchVector(contract_dim,X); //X'_oj

    autoView(A_v,A,DeviceRead);
    autoView(Xvec_v,Xvec,DeviceRead);
    autoView(outv_v,outv,DeviceWrite);

    autoView(Y_v,Y,DeviceRead);
    accelerator_for_2d_gen(1,1,splitBlock<32>(), i, sizei, o, other_size,
			   {
			     outv_v(i+sizei*o) = Y_v(i);
			   });

  
    GEMM(NoTranspose,Transpose,
	   other_size, sizei, sizej,
	   FloatType(1.0),
	   Xvec_v.data(), sizej,
	   A_v.data(), sizej,
	   FloatType(1.0), //initialized out to Y_i, so add
	   outv_v.data());
  }
  Tensor<FloatType,Dim> out(out_dims);
  untransformBatchVector(contract_dim,out,outv);
  return out;
#else
  return matrixBatchTensorAxpy(A, X, Y, contract_dim);
#endif
}
  


int main(int argc, char** argv){
  initialize(argc,argv);
  std::mt19937 rng(1234);

  int contract_dim_max = 2;
  std::vector<int> matrix_dims = { 2, 5,  8, 16, 64, 256, 512 };
  std::vector<int> batch_sizes = {1, 5, 8 , 16, 32, 64};
  std::vector<int> other_dim_sizes = { 2, 5, 8, 16, 64, 256, 512 };

  // int contract_dim_max = 1;
  // std::vector<int> matrix_dims = { 512 };
  // std::vector<int> batch_sizes = { 64};
  // std::vector<int> other_dim_sizes = { 64 };
  
  for(int contract_dim=0; contract_dim<contract_dim_max; contract_dim++){
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
	    
	    Vector<double> y(matrix_dim);
	    uniformRandom(y,rng);
	    
	    Tensor<double,3> x(tsize);
	    uniformRandom(x, rng);
	    
	    Tensor<double,3> c = matrixBatchTensorAxpy_v6(a,x,y,contract_dim);
	    Tensor<double,3> ctest = matrixBatchTensorAxpyBase(a,x,y,contract_dim);
	    assert(abs_near(c,ctest,1e-6,true));
	  }


	  Matrix<float> a(matrix_dim, matrix_dim);
	  uniformRandom(a,rng);

	  Vector<float> y(matrix_dim);
	  uniformRandom(y,rng);
	  
	  Tensor<float,3> x(tsize);
	  uniformRandom(x, rng);
	  
	  double mu, sigma;
	  
	  Tensor<float,3> c;
	  benchmark(mu, sigma, 100, 1, [&]{
	    profileStart();
	    c = matrixBatchTensorAxpy_v6(a,x,y,contract_dim);
	    profileStop();
	  }, []{});
	  
	  double mu_base, sigma_base;
	  benchmark(mu_base, sigma_base, 100, 1, [&]{
	    profileStart();
	    c = matrixBatchTensorAxpy(a,x,y,contract_dim);
	    profileStop();
	  }, []{});


	  size_t FLOPS = size_t(other_dim_size)*size_t(matrix_dim)*size_t(batch_size)*size_t(1 + 2*(matrix_dim-1));
	  	  
	  std::cout << "contract_dim:" << contract_dim << "\tother_dim_size:" << other_dim_size << "\tmatrix_dim:" << matrix_dim << "\tbatch_size:" << batch_size << "\tresult: " << mu/1e-6 << "+-" << sigma/1e-6 << "us (" << FLOPS/mu/1e9 << " Gflops) base: " << mu_base/1e-6 << "+-" << sigma_base/1e-6 << "us (" << FLOPS/mu_base/1e9 << " Gflops)" << std::endl;
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
