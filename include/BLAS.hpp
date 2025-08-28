#pragma once
#include "HPCortexConfig.h"

#ifdef USE_BLAS

enum BLASop { NoTranspose=0, Transpose=1 };

/**
 * @brief Generic wrappers around the (strided) batched GEMM functionality
 *        cf https://docs.nvidia.com/cuda/cublas/#cublas-t-gemmstridedbatched  for arguments
 */
template<typename FloatType>
void batchedGEMM(BLASop transa,
                 BLASop transb,
		 int m, int n, int k,
		 const FloatType           *alpha,
		 const FloatType           *A, int lda,
		 long long int          strideA,
		 const FloatType           *B, int ldb,
		 long long int          strideB,
		 const FloatType           *beta,
		 FloatType                 *C, int ldc,
		 long long int          strideC,
		 int batchCount);


/**
 * @brief Generic wrappers around the (strided) batched GEMV functionality
 *        cf  https://docs.nvidia.com/cuda/archive/12.6.2/cublas/#cublas-t-gemvstridedbatched
 */
template<typename FloatType>
void batchedGEMV(BLASop trans,
		 int m, int n,
		 const FloatType           *alpha,
		 const FloatType           *A, int lda,
		 long long int         strideA,
		 const FloatType           *x, int incx,
		 long long int         stridex,
		 const FloatType           *beta,
		 FloatType                 *y, int incy,
		 long long int         stridey,
		 int batchCount);


/**
 * @brief Generic wrappers around the non-strided batched GEMV functionality
 */
template<typename FloatType>
void batchedGEMV(BLASop trans,
		 int m, int n,
		 const FloatType           *alpha,
		 const FloatType           *const Aarray[], int lda,
		 const FloatType           *const xarray[], int incx,
		 const FloatType           *beta,
		 FloatType           * yarray[], int incy,
		 int batchCount);


/**
 * @brief Generic wrappers around GEMM functionality
 *        cf https://docs.nvidia.com/cuda/archive/12.6.2/cublas/#cublas-t-gemm for arguments
 */
template<typename FloatType>
void GEMM(BLASop transa, BLASop transb,
	  int m, int n, int k,
	  const FloatType           *alpha,
	  const FloatType           *A, int lda,
	  const FloatType           *B, int ldb,
	  const FloatType           *beta,
	  FloatType           *C, int ldc);

/**
 * @brief A wrapper to handle conversion between standard row-major and the archaic column-major format used by BLAS
 *        Computes   alpha * op(A) * op(B) + beta * C 
 * @param transa operation performed on A
 * @param transb operation performed on B
 * @param m rows of C
 * @param n columns of C
 * @param k size of contraction dimension
 * @param alpha coefficient
 * @param A pointer to matrix A
 * @param Acols columns of matrix A
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

/**
 * @brief A wrapper to handle conversion between standard row-major and the archaic column-major format used by BLAS
 *        Computes   alpha * op(A[i]) * op(B[i]) + beta * C[i]    for i in 0...batchCount-1
 * @param transa operation performed on A
 * @param transb operation performed on B
 * @param m rows of C
 * @param n columns of C
 * @param k size of contraction dimension
 * @param alpha coefficient
 * @param A pointer to matrix batch A
 * @param Acols columns of matrix A
 * @param strideA stride between successive A matrices
 * @param B pointer to matrix batch B
 * @param Bcols columns of matrix B
 * @param strideB stride between successive B matrices
 * @param beta coefficient
 * @param C pointer to matrix batch C
 * @param strideC stride between successive C matrices
 * @param batchCount the number of matrices
 */
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
		   int batchCount);

/**
 * @brief A wrapper to handle conversion between standard row-major and the archaic column-major format used by BLAS
 *        Computes   alpha * op(A[i]) * x[i] + beta * y[i]    for i in 0...batchCount-1
 * @param trans operation performed on A
 * @param m rows of A
 * @param n columns of A
 * @param alpha coefficient
 * @param A pointer to matrix batch A
 * @param strideA stride between successive A matrices
 * @param x pointer to vector batch x
 * @param incx increment between elements of x
 * @param stridex stride between successive x vectors
 * @param beta coefficient
 * @param y pointer to vector batch y
 * @param incy increment between elements of y
 * @param stridey stride between successive y vectors
 * @param batchCount the number of matrices
 */
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
		   int batchCount);

/**
 * @brief A wrapper to handle conversion between standard row-major and the archaic column-major format used by BLAS
 *        Computes   alpha * op(A[i]) * x[i] + beta * y[i]    for i in 0...batchCount-1
 * @param trans operation performed on A
 * @param m rows of A
 * @param n columns of A
 * @param alpha coefficient
 * @param Aarray *device* array of *device* pointers to matrices A[i]
 * @param xarray *device* array of *device* pointers to vectors x[i]
 * @param incx increment between elements of x
 * @param beta coefficient
 * @param yarray *device* array of *device* pointers to vectors y[i]
 * @param incy increment between elements of y
 * @param batchCount the number of matrices
 */
template<typename FloatType>
void rmBatchedGEMV(BLASop trans,
		   int m, int n,
		   const FloatType           alpha,
		   const FloatType         *const Aarray[],
		   const FloatType         *const xarray[], int incx,
		   const FloatType           beta,
		   FloatType           * yarray[], int incy,
		   int batchCount);



#include "implementation/BLAS.tcc"
#endif
