#ifndef USE_BLAS
//Non-BLAS implementations suitable for both CPU and GPU

//matrix a * b + c with b having a modest number of columns
template<typename FloatType>
Matrix<FloatType> axpyMatThinMat(const Matrix<FloatType> &a, const Matrix<FloatType> &b, const Vector<FloatType> &c, FLOPScounter *flops){
  int size0 = a.size(0);
  assert(c.size(0) == size0);
  int size1 = a.size(1);
  assert(b.size(0) == size1);
  int size2 = b.size(1);

  if(flops != nullptr && !flops->locked()) //a_ij b_jk + c_i
    flops->add(size0*size2*size1*2);
    
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
Tensor<FloatType,3> batch3tensorContract(const Tensor<FloatType,3> &A, const Tensor<FloatType,3> &B, int contract_dimA, int contract_dimB, FloatType nrm, FLOPScounter *flops){ 
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
  
  if(flops != nullptr && !flops->locked())
    flops->add(sizes_out[0]*sizes_out[1]*batch_size * sizek*2  + sizes_out[0]*sizes_out[1]*batch_size);
    
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


#endif
