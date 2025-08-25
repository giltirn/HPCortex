#include <BLAS.hpp>
#include <Accelerator.hpp>

#ifdef USE_CUBLAS
#include <cublas_v2.h>

static inline cublasOperation_t cublasOpLookup(BLASop op){
  switch(op){
  case NoTranspose:
    return CUBLAS_OP_N;
  case Transpose:
    return CUBLAS_OP_T;
  default:
    throw std::runtime_error("Unsupported BLASop");
  }
}

struct cuBLAShandleContainer{
  cublasHandle_t handle;

  cuBLAShandleContainer(){
    assert( cublasCreate(&handle) == CUBLAS_STATUS_SUCCESS );
    assert( cublasSetStream(handle, computeStream) == CUBLAS_STATUS_SUCCESS );
  }
};
static inline cublasHandle_t getcuBLAShandle(){
  static cuBLAShandleContainer con;
  return con.handle;
}

void batchedGEMM(BLASop transa,
                 BLASop transb,
		 int m, int n, int k,
		 const float           *alpha,
		 const float           *A, int lda,
		 long long int          strideA,
		 const float           *B, int ldb,
		 long long int          strideB,
		 const float           *beta,
		 float                 *C, int ldc,
		 long long int          strideC,
		 int batchCount){
  
  assert( cublasSgemmStridedBatched(getcuBLAShandle(), cublasOpLookup(transa), cublasOpLookup(transb), 
				    m, n, k,
				    alpha,
				    A, lda, strideA,
				    B, ldb, strideB,
				    beta,
				    C, ldc, strideC,
				    batchCount) == CUBLAS_STATUS_SUCCESS );
}
void batchedGEMM(BLASop transa,
                 BLASop transb,
		 int m, int n, int k,
		 const double           *alpha,
		 const double           *A, int lda,
		 long long int          strideA,
		 const double           *B, int ldb,
		 long long int          strideB,
		 const double           *beta,
		 double                 *C, int ldc,
		 long long int          strideC,
		 int batchCount){
  assert( cublasDgemmStridedBatched(getcuBLAShandle(), cublasOpLookup(transa), cublasOpLookup(transb), 
				    m, n, k,
				    alpha,
				    A, lda, strideA,
				    B, ldb, strideB,
				    beta,
				    C, ldc, strideC,
				    batchCount) == CUBLAS_STATUS_SUCCESS );
}

void batchedGEMV(BLASop trans,
		 int m, int n,
		 const float           *alpha,
		 const float           *A, int lda,
		 long long int         strideA,
		 const float           *x, int incx,
		 long long int         stridex,
		 const float           *beta,
		 float                 *y, int incy,
		 long long int         stridey,
		 int batchCount){
  assert( cublasSgemvStridedBatched(getcuBLAShandle(),
				    cublasOpLookup(trans),
				    m,n,
                                    alpha,
				    A, lda, strideA,
				    x, incx, stridex,
				    beta,
				    y, incy, stridey,
				    batchCount) == CUBLAS_STATUS_SUCCESS  );
}
void batchedGEMV(BLASop trans,
		 int m, int n,
		 const double           *alpha,
		 const double           *A, int lda,
		 long long int         strideA,
		 const double           *x, int incx,
		 long long int         stridex,
		 const double           *beta,
		 double                 *y, int incy,
		 long long int         stridey,
		 int batchCount){
  assert( cublasDgemvStridedBatched(getcuBLAShandle(),
				    cublasOpLookup(trans),
				    m,n,
                                    alpha,
				    A, lda, strideA,
				    x, incx, stridex,
				    beta,
				    y, incy, stridey,
				    batchCount) == CUBLAS_STATUS_SUCCESS  );
}


void batchedGEMV(BLASop trans,
		 int m, int n,
		 const float           *alpha,
		 const float           *const Aarray[], int lda,
		 const float           *const xarray[], int incx,
		 const float           *beta,
		 float           *const yarray[], int incy,
		 int batchCount){
  assert( cublasSgemvBatched(getcuBLAShandle(), cublasOpLookup(trans),
			     m, n,
			     alpha,
                             Aarray,lda,
			     xarray, incx,
			     beta,
			     yarray, incy,
			     batchCount) == CUBLAS_STATUS_SUCCESS );
}

void batchedGEMV(BLASop trans,
		 int m, int n,
		 const double           *alpha,
		 const double           *const Aarray[], int lda,
		 const double           *const xarray[], int incx,
		 const double           *beta,
		 double           *const yarray[], int incy,
		 int batchCount){
  assert( cublasDgemvBatched(getcuBLAShandle(), cublasOpLookup(trans),
			     m, n,
			     alpha,
                             Aarray,lda,
			     xarray, incx,
			     beta,
			     yarray, incy,
			     batchCount) == CUBLAS_STATUS_SUCCESS );
}


void GEMM(BLASop transa, BLASop transb,
	  int m, int n, int k,
	  const float           *alpha,
	  const float           *A, int lda,
	  const float           *B, int ldb,
	  const float           *beta,
	  float           *C, int ldc){
  assert(  cublasSgemm(getcuBLAShandle(),
		       cublasOpLookup(transa), cublasOpLookup(transb),
		       m, n, k,
		       alpha,
		       A, lda,
		       B, ldb,
		       beta,
		       C, ldc) == CUBLAS_STATUS_SUCCESS );
}

void GEMM(BLASop transa, BLASop transb,
	  int m, int n, int k,
	  const double           *alpha,
	  const double           *A, int lda,
	  const double           *B, int ldb,
	  const double           *beta,
	  double           *C, int ldc){
  assert(  cublasDgemm(getcuBLAShandle(),
		       cublasOpLookup(transa), cublasOpLookup(transb),
		       m, n, k,
		       alpha,
		       A, lda,
		       B, ldb,
		       beta,
		       C, ldc) == CUBLAS_STATUS_SUCCESS );
}


#endif
