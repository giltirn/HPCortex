#ifdef USE_GPU
//GPU-optimized linalg routines used even with BLAS enabled

//out_jk =  \sum_{b,...} A_{..,j,.., b} B_{..,k,...b}
//Both tensors must have the same dimension, and the sizes of dimensions other that preserve_dim must all be equal
//preserve_dim:  the index of the dimension that is preserved in the output matrix (that of j, k in the above)
//out: a *device* pointer to the output matrix' underlying array, that should be *zero initialized*. Output is stored in the usual lexicographic format, for the above   k+sizek*j
template<typename FloatType, int Dim>
void batchTensorContractToMatrix_p(FloatType* out_p, const Tensor<FloatType,Dim> &A, const Tensor<FloatType,Dim> &B, const int preserve_dim, FLOPScounter *flops){
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

  if(flops != nullptr && !flops->locked())
    flops->add(sizej*sizek* other_size*batch_size*2);
  
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

  accelerator_for_5d_gen(2,3,shm(shmsize), bb, bblocksz, jjkk, jkblocksz, bblock, bblocks, bjk, jkblocks, bo, oblocks, {
      size_t* aoff = (size_t*)shared;
      size_t* boff = (size_t*)(shared + oblocksz*sizeof(size_t) );
  
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
      if(b < batch_size && j < sizej && k < sizek){
	FloatType* A_pb = A_v.data() + b + stride*j;
	FloatType* B_pb = B_v.data() + b + stride*k;	
	
	for(int oo=0;oo<oblocksz_actual;oo++){
	  FloatType* A_p = A_pb + aoff[oo];
	  FloatType* B_p = B_pb + boff[oo];
	  delta += (*A_p) * (*B_p);
	}
      }
      acceleratorSynchronizeBlock();

      FloatType* sharedp = (FloatType*)(shared + bblocksz*jjkk*sizeof(FloatType));
      
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

#endif
