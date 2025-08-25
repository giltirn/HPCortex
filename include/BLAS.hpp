#pragma once
#include "HPCortexConfig.h"

#ifdef USE_BLAS

enum BLASop { NoTranspose=0, Transpose=1 };

/**
 * @brief Generic wrappers around the (strided) batched GEMM functionality
 *        cf https://docs.nvidia.com/cuda/cublas/#cublas-t-gemmstridedbatched  for arguments
 */
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
		 int batchCount);


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
		 int batchCount);

/**
 * @brief Generic wrappers around the (strided) batched GEMV functionality
 *        cf  https://docs.nvidia.com/cuda/archive/12.6.2/cublas/#cublas-t-gemvstridedbatched
 */
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
		 int batchCount);

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
		 int batchCount);

/**
 * @brief Generic wrappers around the non-strided batched GEMV functionality
 */
void batchedGEMV(BLASop trans,
		 int m, int n,
		 const float           *alpha,
		 const float           *const Aarray[], int lda,
		 const float           *const xarray[], int incx,
		 const float           *beta,
		 float           *const yarray[], int incy,
		 int batchCount);

void batchedGEMV(BLASop trans,
		 int m, int n,
		 const double           *alpha,
		 const double           *const Aarray[], int lda,
		 const double           *const xarray[], int incx,
		 const double           *beta,
		 double           *const yarray[], int incy,
		 int batchCount);

/**
 * @brief Generic wrappers around GEMM functionality
 *        cf https://docs.nvidia.com/cuda/archive/12.6.2/cublas/#cublas-t-gemm for arguments
 */
void GEMM(BLASop transa, BLASop transb,
	  int m, int n, int k,
	  const float           *alpha,
	  const float           *A, int lda,
	  const float           *B, int ldb,
	  const float           *beta,
	  float           *C, int ldc);

void GEMM(BLASop transa, BLASop transb,
	  int m, int n, int k,
	  const double           *alpha,
	  const double           *A, int lda,
	  const double           *B, int ldb,
	  const double           *beta,
	  double           *C, int ldc);

/**
 * @brief A wrapper to handle conversion between standard row-major and the archaic column-major format used by BLAS
 *        Computes   alpha * op(A) * op(B) + beta * C 
 * @param transa operation performed on A
 * @param transb operation performed on B
 * @param m sows of C
 * @param n solumns of C
 * @param k size of contraction dimension
 * @param alpha coefficient
 * @param A pointer to matrix A
 * @param Acols cols columns of matrix A
 * @param B pointer to matrix B
 * @param Bcols columns of matrix B
 * @param beta coefficient
 * @param C pointer to matrix C
 */
template<typename FloatType>
void rmGEMM(BLASop transa, BLASop transb,
	   int m, int n, int k,
	   const FloatType           alpha,
	   const FloatType           *A, int Acols,
	   const FloatType           *B, int Bcols,
	   const FloatType           beta,
	   FloatType           *C);

#include "implementation/BLAS.tcc"
#endif
