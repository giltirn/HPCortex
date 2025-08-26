#ifndef USE_GPU
//Non-optimized versions of linalg routines that don't use block sharing

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
    
  accelerator_for3d(dummy,1, k,szk,j,szj,   1,{
      FloatType v = a_v(j,0) * b_v(k,0);
      for(int i=1;i<szi;i++)
	v += a_v(j,i) * b_v(k,i);
      out_p[k+szk*j] = v;
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

    accelerator_for_3d_gen(1,2,normal(), b, batch_size, i, _sizei, o, other_size, {
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

  accelerator_for3d(dummy,1, jk, sizej*sizek, o, other_size, 1,{ 
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


#endif
