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

template<typename FloatType>
void rmBatchedGEMM(BLASop transa,
		   BLASop transb,
		   int m, int n, int k,
		   FloatType           alpha,
		   const FloatType           *A, int Acols,
		   long long int          strideA,
		   const FloatType           *B, int Bcols,
		   long long int          strideB,
		   FloatType           beta,
		   FloatType                 *C,
		   long long int          strideC,
		   int batchCount){
  batchedGEMM(transb, transa, n, m, k,
	      &alpha,
	      B, Bcols, strideB,
	      A, Acols, strideA,
	      &beta,
	      C, n, strideC,
	      batchCount);
}
	       
template<typename FloatType>
void rmBatchedGEMV(BLASop trans,
		   int m, int n,
		   const FloatType           alpha,
		   const FloatType           *A,
		   long long int         strideA,
		   const FloatType           *x, int incx,
		   long long int         stridex,
		   const FloatType           beta,
		   FloatType                 *y, int incy,
		   long long int         stridey,
		   int batchCount){
  //BLAS use column-major format
  //A_cm [col-major] == A_rm^T [row-major]

  //y = a A_rm x_ + b y
  //  = a A_cm^T x + b y
  
  //m=rows of A_cm = cols of A_rm
  //n=cols of A_cm = rows of A_rm
  
  batchedGEMV(trans == Transpose ? NoTranspose : Transpose,
	      n, m,
	      &alpha,
	      A, n, strideA,
	      x, incx, stridex,
	      &beta,
	      y, incy, stridey,
	      batchCount);
}

template<typename FloatType>
void rmBatchedGEMV(BLASop trans,
		   int m, int n,
		   const FloatType           alpha,
		   const FloatType         *const Aarray[],
		   const FloatType         *const xarray[], int incx,
		   const FloatType           beta,
		   FloatType           * yarray[], int incy,
		   int batchCount){
  batchedGEMV(trans == Transpose ? NoTranspose : Transpose,
	      n, m,
	      &alpha,
	      Aarray, n,
	      xarray, incx,
	      &beta,
	      yarray, incy,
	      batchCount);
}
