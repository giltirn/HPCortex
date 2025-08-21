#pragma once
#include "HPCortexConfig.h"

#ifdef USE_BLAS

enum BLASop { NoTranspose, Transpose };

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



#include "implementation/BLAS.tcc"
#endif
