#ifdef USE_BLAS
//Linalg routines using BLAS

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

  rmGEMM(NoTranspose,Transpose,
	 szj,szk,szi,
	 FloatType(1.0),
	 a_v.data(), szi,
	 b_v.data(), szi,
	 FloatType(0.0),
	 out_p);
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
    
  Matrix<FloatType> c(sizei,sizek);
  autoView(c_v,c,DeviceWrite);
  autoView(a_v,a,DeviceRead);
  autoView(b_v,b,DeviceRead);
  rmGEMM(Transpose,NoTranspose,
	 sizei,sizek,sizej,
	 FloatType(1.0),
	 a_v.data(), sizei,
	 b_v.data(), sizek,
	 FloatType(0.0),
	 c_v.data());
  return c;
}

//matrix a * b + c with b having a modest number of columns
template<typename FloatType>
Matrix<FloatType> axpyMatThinMat(const Matrix<FloatType> &a, const Matrix<FloatType> &b, const Vector<FloatType> &c, FLOPScounter *flops){
  int sizei = a.size(0);
  int sizej = a.size(1);
  int sizek = b.size(1);

  assert(c.size(0) == sizei);
  assert(b.size(0) == sizej);
  if(flops != nullptr && !flops->locked()) //a_ij b_jk + c_i
    flops->add(sizei*sizek*sizej*2);
    
  Matrix<FloatType> out(sizei,sizek);

  autoView(out_v,out,DeviceWrite);
  autoView(c_v,c,DeviceRead);
 
  accelerator_for_2d_gen(1,1,splitBlock<32>(), k,sizek,i,sizei,{
      out_v(i,k) = c_v(i);
    });
  
  autoView(a_v,a,DeviceRead);
  autoView(b_v,b,DeviceRead);
  
  rmGEMM(NoTranspose,NoTranspose,
	 sizei,sizek,sizej,
	 FloatType(1.0),
	 a_v.data(), sizej,
	 b_v.data(), sizek,
	 FloatType(1.0),
	 out_v.data());
  return out;
}

//Contract batched 3-tensors (those for whom the last dimension is the batch index) over some dimension \in [0,1]
//eg  C_{ijb} = \sum_k A_{kib} B_{jkb} * nrm
template<typename FloatType>
Tensor<FloatType,3> batch3tensorContract(const Tensor<FloatType,3> &A, const Tensor<FloatType,3> &B, int contract_dimA, int contract_dimB, FloatType nrm, FLOPScounter *flops){
  Vector<FloatType> Abatch = transformBatchMatrix(contract_dimA, !contract_dimA, A); //A[!contract_dim_A, contract_dim_A] in column major
  Vector<FloatType> Bbatch = transformBatchMatrix(!contract_dimB, contract_dimB, B); //B[contract_dim_B, !contract_dim_B]
  int nbatch = A.size(2);
  
  int m = A.size(!contract_dimA);
  int n = B.size(!contract_dimB);
  int k = A.size(contract_dimA);
  FloatType beta = 0.;
  int lda = A.size(!contract_dimA); //column major,  leading dimension is number of rows
  int ldb = B.size(contract_dimB);

  int omat_sz = A.size(!contract_dimA) * B.size(!contract_dimB);
  int ldc = A.size(!contract_dimA); //C[ A.size(!contract_dim_A), B.size(!contract_dim_B) ] in column major
  Vector<FloatType> Cbatch(omat_sz * nbatch);

  int Astride = A.size(!contract_dimA)*A.size(contract_dimA);
  int Bstride = B.size(!contract_dimB)*B.size(contract_dimB);
  int Cstride = omat_sz;
  {
    autoView(Cbatch_v,Cbatch,DeviceWrite);
    autoView(Abatch_v,Abatch,DeviceRead);
    autoView(Bbatch_v,Bbatch,DeviceRead);
    
    batchedGEMM(NoTranspose, NoTranspose, 
		m, n, k,
		&nrm,
		Abatch_v.data(), lda, Astride,
		Bbatch_v.data(), ldb, Bstride,
		&beta,
		Cbatch_v.data(), ldc, Cstride,
		nbatch);
  }
  //C_{ijb} = \sum_k A_{kib} B_{jkb} * nrm
  Tensor<FloatType,3> C(A.size(!contract_dimA),B.size(!contract_dimB),A.size(2));
  untransformBatchMatrix(1,0,C, Cbatch);

  if(flops != nullptr && !flops->locked())
    flops->add(C.size(0)*C.size(1)*C.size(2) * A.size(contract_dimA)*2 + C.size(0)*C.size(1)*C.size(2));
  
  return C;
}


//A_{ij} X_{..., j, ..., b}  + Y_i
template<typename FloatType,int Dim>
Tensor<FloatType,Dim> matrixBatchTensorAxpy(const Matrix<FloatType> &A, const Tensor<FloatType,Dim> &X, const Vector<FloatType> &Y, const int contract_dim, FLOPScounter *flops){
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
        
  if(flops != nullptr && !flops->locked())
    flops->add(other_size*sizei*(2 + 2*(sizej-1)));
  
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

  
    rmGEMM(NoTranspose,Transpose,
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
}


//out_{jb} = \sum_i X_{...,i,...,b}A_{ij} 
template<typename FloatType,int Dim>
Tensor<FloatType,Dim> matrixBatchTensorContractRight(const Tensor<FloatType,Dim> &X, const Matrix<FloatType> &A, const int contract_dim, FLOPScounter *flops){
  Vector<FloatType> Xvec = transformBatchVector(contract_dim,X);
  
  int out_dims[Dim];
  memcpy(out_dims,X.sizeArray(),Dim*sizeof(int));
  out_dims[contract_dim] = A.size(1);
  
  size_t other_size = 1; //dimensions other than the dimensions along which the vector resides
  for(int d=0;d<Dim;d++)
    if(d!= contract_dim)
      other_size *= X.size(d);
     
  int sizei = A.size(0);  //contract over
  int sizej = A.size(1);
  if(flops != nullptr && !flops->locked())
    flops->add(other_size*sizej*(1+2*(sizei-1) ));
    
  Vector<FloatType> ovec(other_size * sizej);

  {
    autoView(Xvec_v, Xvec, DeviceRead);
    autoView(A_v, A, DeviceRead);
    autoView(ovec_v, ovec, DeviceWrite);

    //x_oi A_ij
    rmGEMM(NoTranspose, NoTranspose,
	   other_size, sizej, sizei,
	   FloatType(1.0),
	   Xvec_v.data(), sizei,
	   A_v.data(), sizej,
	   FloatType(0.0),
	   ovec_v.data());
  }
  
  Tensor<FloatType,Dim> out(out_dims);
  untransformBatchVector(contract_dim, out, ovec);  
  return out;
}

//A_{ij} X_{..., j, ..., b}
template<typename FloatType,int Dim>
Tensor<FloatType,Dim> matrixBatchTensorContractLeft(const Matrix<FloatType> &A, const Tensor<FloatType,Dim> &X, const int contract_dim, FLOPScounter *flops){
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
  
  if(flops != nullptr && !flops->locked())
    flops->add(sizei*other_size*(1+2*(sizej-1)));
  
  Vector<FloatType> ovec(other_size * sizei);

  {
    autoView(Xvec_v, Xvec, DeviceRead);
    autoView(A_v, A, DeviceRead);
    autoView(ovec_v, ovec, DeviceWrite);

    //c_oj = A_ij x_oj = x_oj A_ji^T
    rmGEMM(NoTranspose, Transpose,
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
}

#endif
