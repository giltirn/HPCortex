#if defined(USE_GPU) && !defined(USE_BLAS)
//GPU-optimized linalg routines not implemented in BLAS or if BLAS is disabled

// C_jk = \sum_i  A_ji B_ki     for ni small-ish (batch size)  pointer version; pointer must be device writable
template<typename FloatType>
void thinMulMatMatTranspose_p(FloatType* out_p, const Matrix<FloatType> &a, const Matrix<FloatType> &b, FLOPScounter *flops){
  int szj = a.size(0);
  int szi = a.size(1);
  int szk = b.size(0);
  assert(b.size(1) == szi);

  if(flops != nullptr && !flops->locked())
    flops->add(szj*szk*szi*2);
  
  autoView(a_v,a,DeviceRead);
  autoView(b_v,b,DeviceRead);

  int jblocksize = 8;
  int jblocks = (szj + jblocksize-1)/jblocksize;

  int kblocksize = 8;
  int kblocks = (szk + kblocksize-1)/kblocksize;

  int iblocksize = 16;
  int iblocks = (szi + iblocksize-1)/iblocksize;

  assert(jblocksize == kblocksize);
  assert(iblocksize == 2*jblocksize);
  
  accelerator_for3d_shm(jk,jblocksize*kblocksize, bk, kblocks,bj,jblocks,   1, (jblocksize + kblocksize) * iblocksize*sizeof(FloatType),  {
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
}


//C_ik = \sum_j A_ji B_jk for nk small-ish (batch size)
template<typename FloatType>
Matrix<FloatType> mulMatTransposeThinMat(const Matrix<FloatType> &a, const Matrix<FloatType> &b, FLOPScounter *flops){
  int sizei = a.size(1);
  int sizej = a.size(0);
  int sizek = b.size(1);
  assert(b.size(0) == sizej);
  
  if(flops != nullptr && !flops->locked())
    flops->add(sizei*sizek*sizej*2);
       
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

}

//A_{ij} X_{..., j, ..., b}  + Y_i
template<typename FloatType,int Dim>
Tensor<FloatType,Dim> matrixBatchTensorAxpy(const Matrix<FloatType> &A, const Tensor<FloatType,Dim> &X, const Vector<FloatType> &Y, const int contract_dim, FLOPScounter *flops){ 
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

  if(flops != nullptr && !flops->locked())
    flops->add(other_size*_sizei*batch_size*(2 + 2*(_sizej-1)));

  Tensor<FloatType,Dim> out(out_dims); 
  {
    autoView(X_v, X, DeviceRead);
    autoView(Y_v, Y, DeviceRead);
    autoView(A_v, A, DeviceRead);
    autoView(out_v, out, DeviceWrite);
  
    int jblocksz = 64;
    int jblocks = (_sizej + jblocksz -1)/jblocksz;

    constexpr int oblocksz = 4;
    int oblocks = (other_size + oblocksz - 1)/oblocksz;
    
    accelerator_for_3d_gen(1,2,shm( jblocksz*sizeof(FloatType) + 2*oblocksz*sizeof(size_t) ), b, batch_size, i, _sizei, bo, oblocks, {
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

//out_{jb} = \sum_i X_{...,i,...,b}A_{ij} 
template<typename FloatType,int Dim>
Tensor<FloatType,Dim> matrixBatchTensorContractRight(const Tensor<FloatType,Dim> &X, const Matrix<FloatType> &A, const int contract_dim, FLOPScounter *flops){ 
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

  if(flops != nullptr && !flops->locked())
    flops->add(other_size*batch_size*sizej*(1+2*(sizei-1) ));
  
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


//A_{ij} X_{..., j, ..., b}
template<typename FloatType,int Dim>
Tensor<FloatType,Dim> matrixBatchTensorContractLeft(const Matrix<FloatType> &A, const Tensor<FloatType,Dim> &X, const int contract_dim, FLOPScounter *flops){ 
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

  if(flops != nullptr && !flops->locked())
    flops->add(batch_size*_sizei*other_size*(1+ (2*_sizej-1)));
  
  Tensor<FloatType,Dim> out(out_dims); 
  {
    autoView(X_v, X, DeviceRead);
    autoView(A_v, A, DeviceRead);
    autoView(out_v, out, DeviceWrite);
  
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
	  out_v.data()[off_out_o[oo] + b + _stride*i] = out_oib[oo];
	}
	
      });
  }
  return out;
}

#endif
