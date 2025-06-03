#include<HPCortex.hpp>
#include<Testing.hpp>

#ifdef USE_CUDA

template<typename FloatType, int Dim>
void batchTensorContractToMatrix_p_Base(FloatType* out_p, const Tensor<FloatType,Dim> &A, const Tensor<FloatType,Dim> &B, const int preserve_dim){
  int batch_size = A.size(Dim-1);
  assert(B.size(Dim-1)==batch_size);
  size_t other_size = 1;
  for(int d=0;d<Dim-1;d++)
    if(d != preserve_dim){
      other_size *= A.size(d);
      assert(A.size(d) == B.size(d));
    }
  int sizej = A.size(preserve_dim);
  int sizek = B.size(preserve_dim);

  //As the stride between elements in 'preserve_dim' does not depend on the size of this dimension (only those of larger dim), and other sizes are all the same, they will share the same stride
  size_t stride = tensorDimensionStride<Dim>(preserve_dim, A.sizeArray());

  autoView(A_v,A,DeviceRead);
  autoView(B_v,B,DeviceRead);

  accelerator_for3d(dummy,1, jk, sizej*sizek, o, other_size, 64,{ 
      int k = jk % sizek;
      int j = jk / sizek;  //jk = k+sizek*j
      //Sum over batch index, neighboring in memory
      FloatType* A_p = A_v.data() + batchTensorDimensionBaseLin<Dim>(preserve_dim, 0, o, A_v.sizeArray()) + stride*j;
      FloatType* B_p = B_v.data() + batchTensorDimensionBaseLin<Dim>(preserve_dim, 0, o, B_v.sizeArray()) + stride*k;

      FloatType v = (*A_p++) * (*B_p++);
      for(int b=1;b<batch_size;b++)
	v += (*A_p++) * (*B_p++);
      atomicAdd(out_p + jk,  v); //sum over o
    });
}

//this version performs better for larger batch size
template<typename FloatType, int Dim>
void batchTensorContractToMatrix_p_v2(FloatType* out_p, const Tensor<FloatType,Dim> &A, const Tensor<FloatType,Dim> &B, const int preserve_dim){
  int batch_size = A.size(Dim-1);
  assert(B.size(Dim-1)==batch_size);
  size_t other_size = 1;
  for(int d=0;d<Dim-1;d++)
    if(d != preserve_dim){
      other_size *= A.size(d);
      assert(A.size(d) == B.size(d));
    }
  int sizej = A.size(preserve_dim);
  int sizek = B.size(preserve_dim);

  //As the stride between elements in 'preserve_dim' does not depend on the size of this dimension (only those of larger dim), and other sizes are all the same, they will share the same stride
  size_t stride = tensorDimensionStride<Dim>(preserve_dim, A.sizeArray());

  autoView(A_v,A,DeviceRead);
  autoView(B_v,B,DeviceRead);

  int oblocksz = 32;
  int oblocks = (other_size + oblocksz - 1)/oblocksz;
  
  size_t shmsize = std::max(2*oblocksz*sizeof(size_t), batch_size*sizeof(FloatType));

  accelerator_for_1_3_shm(b,batch_size, j, sizej, k,sizej, bo, oblocks, 1, shmsize,{
      extern __shared__ char _shared[];
      size_t* aoff = (size_t*)_shared;
      size_t* boff = (size_t*)(_shared + oblocksz*sizeof(size_t) );
      FloatType* sharedp = (FloatType*)_shared;
      int jk = k+sizek*j;
      
      int oblocksz_actual = other_size - bo*oblocksz < oblocksz ? other_size - bo*oblocksz : oblocksz;

      int oo = b;
      while(oo < oblocksz_actual){
	int o = bo*oblocksz + oo;
	aoff[oo] = batchTensorDimensionBaseLin<Dim>(preserve_dim, 0, o, A_v.sizeArray()) + stride*j;
	boff[oo] = batchTensorDimensionBaseLin<Dim>(preserve_dim, 0, o, B_v.sizeArray()) + stride*k;	
	oo += batch_size;
      }
      acceleratorSynchronizeBlock();
      FloatType delta = 0;
      for(int oo=0;oo<oblocksz_actual;oo++){
	FloatType* A_p = A_v.data() + b + aoff[oo];
	FloatType* B_p = B_v.data() + b + boff[oo];	
	delta += (*A_p) * (*B_p);
      }
      acceleratorSynchronizeBlock();
      
      sharedp[b] = delta;
      acceleratorSynchronizeBlock();

      int rem = batch_size;
      while( (rem & 0x1) == 0x0){
	int remd2 = rem >> 1;
	if(b<remd2)
	  sharedp[b] += sharedp[b+remd2];
	rem = remd2;
	acceleratorSynchronizeBlock();
      }
      if(b == 0){
	delta = sharedp[0];
	for(int bb=1;bb<rem;bb++)
	  delta += sharedp[bb];
	atomicAdd(out_p + jk,  delta);
      }
    });
  
}

//this version performs better for low batch size
template<typename FloatType, int Dim>
void batchTensorContractToMatrix_p_v3(FloatType* out_p, const Tensor<FloatType,Dim> &A, const Tensor<FloatType,Dim> &B, const int preserve_dim){
  int batch_size = A.size(Dim-1);
  assert(B.size(Dim-1)==batch_size);
  size_t other_size = 1;
  for(int d=0;d<Dim-1;d++)
    if(d != preserve_dim){
      other_size *= A.size(d);
      assert(A.size(d) == B.size(d));
    }
  int sizej = A.size(preserve_dim);
  int sizek = B.size(preserve_dim);

  //As the stride between elements in 'preserve_dim' does not depend on the size of this dimension (only those of larger dim), and other sizes are all the same, they will share the same stride
  size_t stride = tensorDimensionStride<Dim>(preserve_dim, A.sizeArray());

  autoView(A_v,A,DeviceRead);
  autoView(B_v,B,DeviceRead);

  int jkblocksz = 32;
  int jkblocks = (sizej * sizek + jkblocksz - 1)/jkblocksz;

  int oblocksz = 32;
  int oblocks = (other_size + oblocksz - 1)/oblocksz;

  accelerator_for3d_shm(jjkk, jkblocksz, bjk, jkblocks, bo, oblocks, 1, (2*oblocksz*sizeof(size_t)),  {
      extern __shared__ char _shared[];
      size_t* aoff = (size_t*)_shared;
      size_t* boff = (size_t*)(_shared + oblocksz*sizeof(size_t) );

      int oblocksz_actual = other_size - bo*oblocksz < oblocksz ? other_size - bo*oblocksz : oblocksz;
      
      int jk = jjkk + jkblocksz * bjk;
      int k = jk % sizek;
      int j = jk / sizek;  //jk = k+sizek*j

      int oo=jjkk;
      while(oo < oblocksz_actual){
	int o = oo + bo*oblocksz;
	aoff[oo] =  batchTensorDimensionBaseLin<Dim>(preserve_dim, 0, o, A_v.sizeArray()) ;
	boff[oo] = batchTensorDimensionBaseLin<Dim>(preserve_dim, 0, o, B_v.sizeArray()) ;
	oo += jkblocksz;
      }
      acceleratorSynchronizeBlock();

      if(j < sizej && k < sizek){
	FloatType delta = 0;
	for(int oo=0;oo<oblocksz_actual;oo++){	
	  FloatType* A_p = A_v.data() + aoff[oo] + stride*j;
	  FloatType* B_p = B_v.data() + boff[oo] + stride*k;
	  for(int b=0;b<batch_size;b++)
	    delta += (*A_p++) * (*B_p++);
	}
	atomicAdd(out_p + jk,  delta); //sum over o
      }
	
    });
}

//this version performs much better than v2 at lower batch sizes while still retaining performance for large batch sizes
template<typename FloatType, int Dim>
void batchTensorContractToMatrix_p_v4(FloatType* out_p, const Tensor<FloatType,Dim> &A, const Tensor<FloatType,Dim> &B, const int preserve_dim){
  int batch_size = A.size(Dim-1);
  assert(B.size(Dim-1)==batch_size);
  size_t other_size = 1;
  for(int d=0;d<Dim-1;d++)
    if(d != preserve_dim){
      other_size *= A.size(d);
      assert(A.size(d) == B.size(d));
    }
  int sizej = A.size(preserve_dim);
  int sizek = B.size(preserve_dim);

  //As the stride between elements in 'preserve_dim' does not depend on the size of this dimension (only those of larger dim), and other sizes are all the same, they will share the same stride
  size_t stride = tensorDimensionStride<Dim>(preserve_dim, A.sizeArray());

  autoView(A_v,A,DeviceRead);
  autoView(B_v,B,DeviceRead);

  int oblocksz = 32;
  int oblocks = (other_size + oblocksz - 1)/oblocksz;

  int bblocksz = std::min(batch_size,16);
  int bblocks = (batch_size + bblocksz - 1)/bblocksz;

  int jkblocksz = 32;
  int jkblocks = (sizej * sizek + jkblocksz - 1)/jkblocksz;
  
  size_t shmsize = std::max(2*oblocksz*sizeof(size_t), bblocksz*jkblocksz*sizeof(FloatType));

  accelerator_for_2_3_shm(bb, bblocksz, jjkk, jkblocksz, bblock, bblocks, bjk, jkblocks, bo, oblocks, shmsize,  {
      extern __shared__ char _shared[];
      size_t* aoff = (size_t*)_shared;
      size_t* boff = (size_t*)(_shared + oblocksz*sizeof(size_t) );
  
      int oblocksz_actual = other_size - bo*oblocksz < oblocksz ? other_size - bo*oblocksz : oblocksz;
        
      int jk = jjkk + jkblocksz * bjk;
      int k = jk % sizek;
      int j = jk / sizek;  //jk = k+sizek*j
      int b = bb + bblocksz * bblock;
      
      int oo = bb + bblocksz*jjkk;
      while(oo < oblocksz_actual){
	int o = bo*oblocksz + oo;
	aoff[oo] = batchTensorDimensionBaseLin<Dim>(preserve_dim, 0, o, A_v.sizeArray());
	boff[oo] = batchTensorDimensionBaseLin<Dim>(preserve_dim, 0, o, B_v.sizeArray());
	oo += bblocksz*jkblocksz;
      }
      acceleratorSynchronizeBlock();
      FloatType delta = 0;
      if(b < batch_size){
	FloatType* A_pb = A_v.data() + b + stride*j;
	FloatType* B_pb = B_v.data() + b + stride*k;	
	
	for(int oo=0;oo<oblocksz_actual;oo++){
	  FloatType* A_p = A_pb + aoff[oo];
	  FloatType* B_p = B_pb + boff[oo];
	  delta += (*A_p) * (*B_p);
	}
      }
      acceleratorSynchronizeBlock();

      FloatType* sharedp = (FloatType*)(_shared + bblocksz*jjkk*sizeof(FloatType));
      
      sharedp[bb] = delta;
      acceleratorSynchronizeBlock();

      int rem = bblocksz;
      while( (rem & 0x1) == 0x0){
	int remd2 = rem >> 1;
	if(bb<remd2)
	  sharedp[bb] += sharedp[bb+remd2];
	rem = remd2;
	acceleratorSynchronizeBlock();
      }
      if(bb == 0){
	delta = sharedp[0];
	for(int bbb=1;bbb<rem;bbb++)
	  delta += sharedp[bbb];
	if(j<sizej && k<sizek) atomicAdd(out_p + jk,  delta);
      }
    });
  
}












int main(int argc, char** argv){
  initialize(argc,argv);
  std::mt19937 rng(1234);

  std::vector<int> matrix_dims = { 2, 5, 8, 16, 64, 256, 512 };
  std::vector<int> batch_sizes = {1, 5, 8, 16, 32, 64};
  std::vector<int> other_dim_sizes = { 2, 5, 8, 16, 64, 256, 512 };
 
  for(int preserve_dim=0; preserve_dim<2; preserve_dim++){
    for(auto other_dim_size : other_dim_sizes){
      for(auto matrix_dim : matrix_dims){
	for(auto batch_size : batch_sizes){

	  //out_jk =  \sum_{b,...} A_{..,j,.., b} B_{..,k,...b}  
	  int tsize[3];
	  tsize[2] = batch_size;
	  tsize[preserve_dim] = matrix_dim;
	  tsize[1-preserve_dim] = other_dim_size;

	  //Check in double
	  {
	    Tensor<double,3> A(tsize);
	    uniformRandom(A,rng);
	    Tensor<double,3> B(tsize);
	    uniformRandom(B,rng);

	    Matrix<double> out(matrix_dim, matrix_dim, 0.);
	    {
	      autoView(op,out,DeviceReadWrite);
	      batchTensorContractToMatrix_p_v4(op.data(),A,B,preserve_dim);
	    }

	    Matrix<double> outtest(matrix_dim, matrix_dim, 0.);
	    {
	      autoView(op,outtest,DeviceReadWrite);
	      batchTensorContractToMatrix_p_Base(op.data(),A,B,preserve_dim);
	    }
	    assert(abs_near(out,outtest,1e-5,true));
	  }
  
	  Tensor<float,3> A(tsize);
	  uniformRandom(A,rng);
	  Tensor<float,3> B(tsize);
	  uniformRandom(B,rng);

	  Matrix<float> out;
	  	  
	  double mu_base=0, sigma_base=0;
	  
	  benchmark(mu_base, sigma_base, 100, 1, [&]{
	    profileStart();
	    autoView(op,out,DeviceReadWrite);
	    batchTensorContractToMatrix_p_v2(op.data(),A,B,preserve_dim);
	    profileStop();
	  }, [&]{
	    out = Matrix<float>(matrix_dim, matrix_dim, 0.);
	  });

	  double mu=0, sigma=0;
	  
	  benchmark(mu, sigma, 100, 1, [&]{
	    profileStart();
	    autoView(op,out,DeviceReadWrite);
	    batchTensorContractToMatrix_p_v4(op.data(),A,B,preserve_dim);
	    profileStop();
	  }, [&]{
	    out = Matrix<float>(matrix_dim, matrix_dim, 0.);
	  });

	  
	  std::cout << "preserve_dim:" << preserve_dim << "\tother_dim_size:" << other_dim_size << "\tmatrix_dim:" << matrix_dim << "\tbatch_size:" << batch_size << "\tresult: " << mu/1e-6 << "+-" << sigma/1e-6 << "us" << " base: " << mu_base/1e-6 << "+-" << sigma_base/1e-6 << "us" << std::endl;
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
