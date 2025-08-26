//C_jk = \sum_i  A_ji B_ki     for ni small-ish (batch size)
template<typename FloatType>
Matrix<FloatType> thinMulMatMatTranspose(const Matrix<FloatType> &a, const Matrix<FloatType> &b, FLOPScounter* flops){
  Matrix<FloatType> out(a.size(0),b.size(0));
  autoView(out_v,out,DeviceWrite);
  thinMulMatMatTranspose_p(out_v.data(),a,b,flops);
  return out;  
}


//out(i, b) = above_deriv(i,b) * activation_deriv(i,b)
template<typename FloatType>
Matrix<FloatType> computeThinMatOuterProd(const Matrix<FloatType> &above_deriv, const Matrix<FloatType> &activation_deriv, FLOPScounter *flops){
  int size0 = above_deriv.size(0);
  int batch_size =  above_deriv.size(1);
  assert(activation_deriv.size(0) == size0 && activation_deriv.size(1) == batch_size);

  if(flops != nullptr && !flops->locked())
    flops->add(size0*batch_size);
  
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

template<typename FloatType>
Matrix<FloatType> matrixBatchTensorAxpy(const Matrix<FloatType> &A, const Matrix<FloatType> &X, const Vector<FloatType> &Y, const int contract_dim, FLOPScounter *flops){
  assert(contract_dim == 0);
  return axpyMatThinMat(A,X,Y,flops);
}

//Specialized implementation for matrices for performance
//out_jk =  \sum_b A_{j,b} B_{k,b}
template<typename FloatType>
inline void batchTensorContractToMatrix_p(FloatType* out_p, const Matrix<FloatType> &A, const Matrix<FloatType> &B, const int preserve_dim, FLOPScounter *flops){
  assert(preserve_dim == 0);
  thinMulMatMatTranspose_p(out_p, A, B, flops);
}

//out_{jb} = \sum_i X_{i,b}A_{ij} = \sum_i X_{i,b}A_{ij}
template<typename FloatType>
inline Tensor<FloatType,2> matrixBatchTensorContractRight(const Tensor<FloatType,2> &X, const Matrix<FloatType> &A, const int contract_dim, FLOPScounter *flops){
  //\sum_i X_{i,b}A_{ij} = \sum_i A_{ij} X_{i,b}
  assert(contract_dim == 0);
  return mulMatTransposeThinMat(A,X,flops);
}
