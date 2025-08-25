template<typename FloatType>
void rmGEMM(BLASop transa, BLASop transb,
	   int m, int n, int k,
	   const FloatType           alpha,
	   const FloatType           *A, int Acols,
	   const FloatType           *B, int Bcols,
	   const FloatType           beta,
	   FloatType           *C){
  //BLAS use column-major format
  //A_cm [col-major] == A_rm^T [row-major]
 
  //!transA,  !transB
  //   C_rm = A_rm * B_rm
  //   C^T_cm = A^T_cm B^T_cm
  //   C_cm = B_cm * A_cm

  //!transA,  transB
  //   C_rm = A_rm * B^T_rm
  //   C_cm^T = A_cm^T * B_cm
  //   C_cm = B_cm^T A_cm

  //etc

  //m' = rows(C_cm) = cols(C_rm) = n
  //n' = cols(C_cm) = rows(C_rm) = m
  //k' = contraction size = k

  //lda = leading dimension of A' = rows(A_cm) = cols(A_rm)
  //ldb = leading dimension of B' = rows(B_cm) = cols(B_rm)
  //ldc = rows(C_cm) = cols(C_rm) = n
  
  GEMM(transb, transa, n,m,k,
       &alpha,
       B, Bcols,
       A, Acols,
       &beta,
       C, n);
}
