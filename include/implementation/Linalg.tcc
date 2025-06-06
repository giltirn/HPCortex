// C_jk = \sum_i  A_ji B_ki     for ni small-ish (batch size)  pointer version; pointer must be device writable
template<typename FloatType>
void thinMulMatMatTranspose_p(FloatType* out_p, const Matrix<FloatType> &a, const Matrix<FloatType> &b){
  int szj = a.size(0);
  int szi = a.size(1);
  int szk = b.size(0);
  assert(b.size(1) == szi);

  autoView(a_v,a,DeviceRead);
  autoView(b_v,b,DeviceRead);

#ifdef USE_CUDA 
  int jblocksize = 8;
  int jblocks = (szj + jblocksize-1)/jblocksize;

  int kblocksize = 8;
  int kblocks = (szk + kblocksize-1)/kblocksize;

  int iblocksize = 16;
  int iblocks = (szi + iblocksize-1)/iblocksize;

  assert(jblocksize == kblocksize);
  assert(iblocksize == 2*jblocksize);
  
  accelerator_for3d_shm(jk,jblocksize*kblocksize, bk, kblocks,bj,jblocks,   1, (jblocksize + kblocksize) * iblocksize*sizeof(FloatType),  {
      extern __shared__ char shared[];
      FloatType *abuf = (FloatType*)shared;
      FloatType *bbuf = (FloatType*)shared + jblocksize * iblocksize;
      
      //jk = kk+kblocksize*jj
      int kk = jk % kblocksize;
      int jj = jk / kblocksize;

      int j = jj + jblocksize*bj;
      int k = kk + kblocksize*bk;      

      FloatType v = 0.0;
      for(int bi=0;bi<iblocks;bi++){
	int istart = bi * iblocksize;
	int ilessthan = istart + iblocksize;
	if(ilessthan > szi) ilessthan = szi;
	int iblocksize_actual = ilessthan - istart;
  
	if(j<szj){
	  int ii = kk;
	  int i = ii + istart;	    
	  abuf[ii + iblocksize*jj] = i < szi ? a_v(j,i) : 0.;

	  ii += kblocksize;
	  i = ii + istart;
	  abuf[ii + iblocksize*jj] = i < szi ? a_v(j,i) : 0.;
	}
	{
	  //efficient to swap mapping of jk -> jj, kj here such that jj is the fast moving index
	  int jj_tmp = jk % jblocksize;
	  int kk_tmp = jk / jblocksize;
	  int k_tmp = kk_tmp + kblocksize*bk;

	  if(k_tmp < szk){
	    int ii = jj_tmp;
	    int i = ii + istart;	    
	    bbuf[ii + iblocksize*kk_tmp] = i < szi ? b_v(k_tmp,i) : 0.;
	      
	    ii += jblocksize;
	    i = ii + istart;
	    bbuf[ii + iblocksize*kk_tmp] = i < szi ? b_v(k_tmp,i) : 0.;
	  }
	}
    
	acceleratorSynchronizeBlock();
	  
	for(int ii=0;ii<iblocksize_actual;ii++){	
	  v += abuf[ii + iblocksize*jj] * bbuf[ii + iblocksize*kk];
	}
	acceleratorSynchronizeBlock();
      }
      if(j < szj && k < szk) out_p[k+szk*j] = v;
	 
    });

#else  
    
  accelerator_for3d(dummy,1, k,szk,j,szj,   64,{
      FloatType v = a_v(j,0) * b_v(k,0);
      for(int i=1;i<szi;i++)
	v += a_v(j,i) * b_v(k,i);
      out_p[k+szk*j] = v;
    });
  
#endif
}


//C_jk = \sum_i  A_ji B_ki     for ni small-ish (batch size)
template<typename FloatType>
Matrix<FloatType> thinMulMatMatTranspose(const Matrix<FloatType> &a, const Matrix<FloatType> &b){
  Matrix<FloatType> out(a.size(0),b.size(0));
  autoView(out_v,out,DeviceWrite);
  thinMulMatMatTranspose_p(out_v.data(),a,b);
  return out;  
}


//C_ik = \sum_j A_ji B_jk for nk small-ish (batch size)
template<typename FloatType>
Matrix<FloatType> mulMatTransposeThinMat(const Matrix<FloatType> &a, const Matrix<FloatType> &b){
  int sizei = a.size(1);
  int sizej = a.size(0);
  int sizek = b.size(1);
  assert(b.size(0) == sizej);

#ifdef USE_CUDA
  
  if(sizej < 300){
    Matrix<FloatType> c(sizei,sizek);
    autoView(c_v,c,DeviceWrite);
    autoView(a_v,a,DeviceRead);
    autoView(b_v,b,DeviceRead);
    
    accelerator_for2d(k, sizek,i,sizei,1,{
	FloatType v = a_v(0,i) * b_v(0,k);
	for(int j=1;j<sizej;j++)
	  v += a_v(j,i) * b_v(j,k);
	c_v(i,k) = v;
      });

    return c;
  }else{

    Matrix<FloatType> c(sizei,sizek,0.);
    autoView(c_v,c,DeviceReadWrite);
    autoView(a_v,a,DeviceRead);
    autoView(b_v,b,DeviceRead);
    
    constexpr int iblocksize = 32;
    constexpr int jblocksize = 16;
    constexpr int kblocksize = 64;
 
    int niblocks = (sizei + iblocksize  - 1) / iblocksize;
    int njblocks = (sizej + jblocksize  - 1) / jblocksize;  
    int nkblocks = (sizek + kblocksize  - 1) / kblocksize;
  
    int kthr = std::max(kblocksize,iblocksize);

 
    accelerator_for3d_shm(kk, kthr, bjk, nkblocks*njblocks, bi,niblocks, 1,   (iblocksize*jblocksize + jblocksize*kblocksize)*sizeof(FloatType), {
	int bk = bjk % nkblocks;
	int bj = bjk / nkblocks;
      
	extern __shared__ char shared[];
	FloatType* shared_a = (FloatType*)shared;
	FloatType* shared_b = (FloatType*)shared + iblocksize*jblocksize;
      
	int ibase = iblocksize * bi;
	int irem = sizei - ibase;
	int icount = irem < iblocksize ? irem : iblocksize;
            
	int jbase = jblocksize * bj;
	int jrem = sizej - jbase;
	int jcount = jrem < jblocksize ? jrem : jblocksize;

	int kbase = kblocksize * bk;
	int krem = sizek - kbase;
	int kcount = krem < kblocksize ? krem : kblocksize;
      
      
	if(kk < icount){
	  int ii = kk;
	  int i = ibase + ii;

	  for(int jj=0;jj<jcount;jj++)
	    shared_a[jj + jblocksize*ii] = a_v(jj+jbase, i);	
	}
	if(kk < kcount){	
	  for(int jj=0;jj<jcount;jj++)
	    shared_b[kk + kblocksize * jj] = b_v(jj+jbase,kk + kbase);
	}
      
	acceleratorSynchronizeBlock();

	if(kk < kcount){	
	  for(int ii=0;ii<icount;ii++){
	    FloatType v = shared_a[jblocksize*ii] * shared_b[kk];
	    for(int jj=1;jj<jcount;jj++){
	      v += shared_a[jj + jblocksize*ii] * shared_b[kk + kblocksize * jj];
	    }  
	    atomicAdd(&c_v(ii+ibase,kk + kbase), v);
	  }		
	}
      });
    return c;
  }
#else
  Matrix<FloatType> c(sizei,sizek);
  autoView(c_v,c,DeviceWrite);
  autoView(a_v,a,DeviceRead);
  autoView(b_v,b,DeviceRead);
    
  accelerator_for2d(k, sizek,i,sizei,1,{
      FloatType v = a_v(0,i) * b_v(0,k);
      for(int j=1;j<sizej;j++)
	v += a_v(j,i) * b_v(j,k);
      c_v(i,k) = v;
    });

  return c;
#endif
  
}


//out(i, b) = above_deriv(i,b) * activation_deriv(i,b)
template<typename FloatType>
Matrix<FloatType> computeThinMatOuterProd(const Matrix<FloatType> &above_deriv, const Matrix<FloatType> &activation_deriv){
  int size0 = above_deriv.size(0);
  int batch_size =  above_deriv.size(1);
  assert(activation_deriv.size(0) == size0 && activation_deriv.size(1) == batch_size);
  
  Matrix<FloatType> activated_above_deriv(size0,batch_size);
  autoView(above_deriv_v,above_deriv,DeviceRead);
  autoView(activation_deriv_v,activation_deriv,DeviceRead);
  autoView(activated_above_deriv_v,activated_above_deriv,DeviceWrite);

  int bblocksize = std::min(128,batch_size);
  int nbblocks = (batch_size + bblocksize - 1) / bblocksize; 
  
  accelerator_for3d(bb,bblocksize,bblock,nbblocks,i,size0,1,{      
      int b = bb + bblocksize*bblock;
      if(b < batch_size){      
	activated_above_deriv_v(i,b) = above_deriv_v(i,b) * activation_deriv_v(i,b);
      }
    });
  return activated_above_deriv;
}


//matrix a * b + c with b having a modest number of columns
template<typename FloatType>
Matrix<FloatType> axpyMatThinMat(const Matrix<FloatType> &a, const Matrix<FloatType> &b, const Vector<FloatType> &c){
  int size0 = a.size(0);
  assert(c.size(0) == size0);
  int size1 = a.size(1);
  assert(b.size(0) == size1);
  int size2 = b.size(1);
    
  Matrix<FloatType> out(size0,size2);
  {
    autoView(c_v,c,DeviceRead);
    autoView(out_v,out,DeviceWrite);
    autoView(b_v,b,DeviceRead);
    autoView(a_v,a,DeviceRead);

    //Basic version where columns are summed over within a thread and rows/batches distributed over threads
    accelerator_for2d(k,size2,i,size0,1,{
	FloatType v = c_v(i);
	for(int j=0;j<size1;j++)
	  v += a_v(i,j)* b_v(j,k);
	out_v(i,k) = v;	  
      });      
  }
  return out;
}


//Contract batched 3-tensors (those for whom the last dimension is the batch index) over some dimension \in [0,1]
//eg  C_{ijb} = \sum_k A_{kib} B_{jkb} * nrm

template<typename FloatType>
Tensor<FloatType,3> batch3tensorContract(const Tensor<FloatType,3> &A, const Tensor<FloatType,3> &B, int contract_dimA, int contract_dimB, FloatType nrm){
  assert(A.size(2) == B.size(2));
  size_t batch_size = A.size(2);

  assert(contract_dimA == 0 || contract_dimA == 1);
  assert(contract_dimB == 0 || contract_dimB == 1);
  assert(A.size(contract_dimA) == B.size(contract_dimB));
  
  int sizes_out[3];
  sizes_out[0] = contract_dimA == 0 ? A.size(1) : A.size(0);
  sizes_out[1] = contract_dimB == 0 ? B.size(1) : B.size(0);
  sizes_out[2] = batch_size;
  Tensor<FloatType,3> out(sizes_out);

  //Let a be the other index for A, and b for B
  int sizek = A.size(contract_dimA);

  size_t Astride[2] = { A.size(1)*batch_size, batch_size };  //(i,j,b) -> b + batch_size*(j + sizej * i)
  size_t Bstride[2] = { B.size(1)*batch_size, batch_size };
  
  size_t kstrideA = Astride[contract_dimA];
  size_t astride = Astride[1-contract_dimA];
  
  size_t kstrideB = Bstride[contract_dimB];
  size_t bstride = Bstride[1-contract_dimB];

  {
    autoView(out_v,out,DeviceWrite);
    autoView(A_v,A,DeviceRead);
    autoView(B_v,B,DeviceRead);
    
    accelerator_for3d(batch, batch_size, a, sizes_out[0], b, sizes_out[1],   1, {
	FloatType* A_p = A_v.data() + batch + a*astride;
	FloatType* B_p = B_v.data() + batch + b*bstride;
	FloatType res = (*A_p) * (*B_p);
	A_p += kstrideA;
	B_p += kstrideB;
	for(int k=1;k<sizek;k++){
	  res += (*A_p) * (*B_p);
	  A_p += kstrideA;
	  B_p += kstrideB;
	}
	out_v(a,b,batch) = res * nrm;
      });
  }
  return out;
}

  
//A_{ij} X_{..., j, ..., b}  + Y_i
template<typename FloatType,int Dim>
Tensor<FloatType,Dim> matrixBatchTensorAxpy(const Matrix<FloatType> &A, const Tensor<FloatType,Dim> &X, const Vector<FloatType> &Y, const int contract_dim){
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
    autoView(Y_v, Y, DeviceRead);
    autoView(A_v, A, DeviceRead);
    autoView(out_v, out, DeviceWrite);
  
#ifdef USE_CUDA
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
  

#else
  
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

#endif
    
  }
  return out;
}
  
//out_jk =  \sum_{b,...} A_{..,j,.., b} B_{..,k,...b}
//Both tensors must have the same dimension, and the sizes of dimensions other that preserve_dim must all be equal
//preserve_dim:  the index of the dimension that is preserved in the output matrix (that of j, k in the above)
//out: a *device* pointer to the output matrix' underlying array, that should be *zero initialized*. Output is stored in the usual lexicographic format, for the above   k+sizek*j
template<typename FloatType, int Dim>
void batchTensorContractToMatrix_p(FloatType* out_p, const Tensor<FloatType,Dim> &A, const Tensor<FloatType,Dim> &B, const int preserve_dim){
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

#ifdef USE_CUDA
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

#else
  
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

#endif
}
//Specialized implementation for matrices for performance
//out_jk =  \sum_b A_{j,b} B_{k,b}
template<typename FloatType>
inline void batchTensorContractToMatrix_p(FloatType* out_p, const Matrix<FloatType> &A, const Matrix<FloatType> &B, const int preserve_dim){
  assert(preserve_dim == 0);
  thinMulMatMatTranspose_p(out_p, A, B);
}

//out_{jb} = \sum_i X_{...,i,...,b}A_{ij} 
template<typename FloatType,int Dim>
Tensor<FloatType,Dim> matrixBatchTensorContractRight(const Tensor<FloatType,Dim> &X, const Matrix<FloatType> &A, const int contract_dim){
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

#ifdef USE_CUDA
    
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

#else
    
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

#endif


    
  }
  return out;
}

//out_{jb} = \sum_i X_{i,b}A_{ij} = \sum_i X_{i,b}A_{ij}
template<typename FloatType>
inline Tensor<FloatType,2> matrixBatchTensorContractRight(const Tensor<FloatType,2> &X, const Matrix<FloatType> &A, const int contract_dim){
  //\sum_i X_{i,b}A_{ij} = \sum_i A_{ij} X_{i,b}
  assert(contract_dim == 0);
  return mulMatTransposeThinMat(A,X);
}
