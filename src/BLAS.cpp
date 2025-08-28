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
		 float           * yarray[], int incy,
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
		 double           * yarray[], int incy,
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

#elif defined(USE_ONEMKL)

#undef VERSION
#include <oneapi/mkl.hpp>

using namespace oneapi::mkl;
using namespace oneapi::mkl::blas::column_major;

static inline oneapi::mkl::transpose onemklOpLookup(BLASop op){
  switch(op){
  case NoTranspose:
    return oneapi::mkl::transpose::N;
  case Transpose:
    return oneapi::mkl::transpose::T;
  default:
    throw std::runtime_error("Unsupported BLASop");
  }
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
  gemm_batch(*computeQueue,
	     onemklOpLookup(transa), onemklOpLookup(transb),
	     m,n,k,
	     *alpha,
	     A, lda, strideA,
	     B, ldb, strideB,
	     *beta,
	     C, ldc, strideC,
	     batchCount).wait();
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
  gemm_batch(*computeQueue,
	     onemklOpLookup(transa), onemklOpLookup(transb),
	     m,n,k,
	     *alpha,
	     A, lda, strideA,
	     B, ldb, strideB,
	     *beta,
	     C, ldc, strideC,
	     batchCount).wait();
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
  gemv_batch(*computeQueue,
	     onemklOpLookup(trans),
	     m,n,
	     alpha,
	     A,lda,strideA,
	     x,incx,stridex,
	     beta,
	     y, incy, stridey,
	     batchCount).wait();
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
  gemv_batch(*computeQueue,
	     onemklOpLookup(trans),
	     m,n,
	     alpha,
	     A,lda,strideA,
	     x,incx,stridex,
	     beta,
	     y, incy, stridey,
	     batchCount).wait();
}



void batchedGEMV(BLASop trans,
		 int m, int n,
		 const float           *alpha,
		 const float           *const Aarray[], int lda,
		 const float           *const xarray[], int incx,
		 const float           *beta,
		 float           * yarray[], int incy,
		 int batchCount){ 
  int m_a[batchCount], n_a[batchCount], lda_a[batchCount], incx_a[batchCount], incy_a[batchCount], group_size[batchCount];
  float alpha_a[batchCount], beta_a[batchCount];
  oneapi::mkl::transpose trans_a[batchCount];
  oneapi::mkl::transpose trans_t = onemklOpLookup(trans);
  for(int i=0;i<batchCount;i++){
    m_a[i] = m;
    n_a[i] = n;
    lda_a[i] = lda;
    incx_a[i] = incx;
    incy_a[i] = incy;
    group_size[i] = 1;
    alpha_a[i] = *alpha;
    beta_a[i] = *beta;
    trans_a[i] = trans_t;
  }
  
  gemv_batch(*computeQueue,
             trans_a,
	     m_a,n_a,
	     alpha_a,
	     (const float**)Aarray, lda_a,
	     (const float**)xarray, incx_a,
	     beta_a,
	     (float**)yarray, incy_a,
	     batchCount,
	     group_size).wait();

}

void batchedGEMV(BLASop trans,
		 int m, int n,
		 const double           *alpha,
		 const double           *const Aarray[], int lda,
		 const double           *const xarray[], int incx,
		 const double           *beta,
		 double           * yarray[], int incy,
		 int batchCount){
  int m_a[batchCount], n_a[batchCount], lda_a[batchCount], incx_a[batchCount], incy_a[batchCount], group_size[batchCount];
  double alpha_a[batchCount], beta_a[batchCount];
  oneapi::mkl::transpose trans_a[batchCount];
  oneapi::mkl::transpose trans_t = onemklOpLookup(trans);
  for(int i=0;i<batchCount;i++){
    m_a[i] = m;
    n_a[i] = n;
    lda_a[i] = lda;
    incx_a[i] = incx;
    incy_a[i] = incy;
    group_size[i] = 1;
    alpha_a[i] = *alpha;
    beta_a[i] = *beta;
    trans_a[i] = trans_t;
  }
  
  gemv_batch(*computeQueue,
             trans_a,
	     m_a,n_a,
	     alpha_a,
	     (const double**)Aarray, lda_a,
	     (const double**)xarray, incx_a,
	     beta_a,
	     (double**)yarray, incy_a,
	     batchCount,
	     group_size).wait();

}


void GEMM(BLASop transa, BLASop transb,
	  int m, int n, int k,
	  const float           *alpha,
	  const float           *A, int lda,
	  const float           *B, int ldb,
	  const float           *beta,
	  float           *C, int ldc){
  gemm(*computeQueue,
       onemklOpLookup(transa), onemklOpLookup(transb),
       m,n,k,
       *alpha,
       A, lda,
       B, ldb,
       *beta,
       C, ldc).wait();
}


void GEMM(BLASop transa, BLASop transb,
	  int m, int n, int k,
	  const double           *alpha,
	  const double           *A, int lda,
	  const double           *B, int ldb,
	  const double           *beta,
	  double           *C, int ldc){
  gemm(*computeQueue,
       onemklOpLookup(transa), onemklOpLookup(transb),
       m,n,k,
       *alpha,
       A, lda,
       B, ldb,
       *beta,
       C, ldc).wait();
}


#endif
