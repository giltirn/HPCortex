#pragma once
#include <Tensors.hpp>
#include <Accelerator.hpp>
#include <BLAS.hpp>
#include <Performance.hpp>

////////////////////////////// Matrix linalg //////////////////////////////////////////////////

// C_jk = \sum_i  A_ji B_ki     for ni small-ish (batch size)  pointer version; pointer must be device writable
template<typename FloatType>
void thinMulMatMatTranspose_p(FloatType* out_p, const Matrix<FloatType> &a, const Matrix<FloatType> &b, FLOPScounter *flops = nullptr);

//C_jk = \sum_i  A_ji B_ki     for ni small-ish (batch size)
template<typename FloatType>
Matrix<FloatType> thinMulMatMatTranspose(const Matrix<FloatType> &a, const Matrix<FloatType> &b, FLOPScounter *flops = nullptr);

//C_ik = \sum_j A_ji B_jk for nk small-ish (batch size)
template<typename FloatType>
Matrix<FloatType> mulMatTransposeThinMat(const Matrix<FloatType> &a, const Matrix<FloatType> &b, FLOPScounter *flops = nullptr);

//out(i, b) = above_deriv(i,b) * activation_deriv(i,b)
template<typename FloatType>
Matrix<FloatType> computeThinMatOuterProd(const Matrix<FloatType> &above_deriv, const Matrix<FloatType> &activation_deriv, FLOPScounter *flops = nullptr);

//matrix a * b + c with b having a modest number of columns
template<typename FloatType>
Matrix<FloatType> axpyMatThinMat(const Matrix<FloatType> &a, const Matrix<FloatType> &b, const Vector<FloatType> &c, FLOPScounter *flops = nullptr);

////////////////////////////// Batch-tensor linalg (tensors for which the last dimension is the batch index) /////////////////////////////////////////////

//Contract batch 3-tensors over some dimension \in [0,1]
//eg  C_{ijb} = \sum_k A_{kib} B_{jkb} * nrm
template<typename FloatType>
Tensor<FloatType,3> batch3tensorContract(const Tensor<FloatType,3> &A, const Tensor<FloatType,3> &B, int contract_dimA, int contract_dimB, FloatType nrm = 1.0, FLOPScounter *flops = nullptr);

//A_{ij} X_{..., j, ..., b}  + Y_i
template<typename FloatType,int Dim>
Tensor<FloatType,Dim> matrixBatchTensorAxpy(const Matrix<FloatType> &A, const Tensor<FloatType,Dim> &X, const Vector<FloatType> &Y, const int contract_dim, FLOPScounter *flops = nullptr);

//out_jk =  \sum_{b,...} A_{..,j,.., b} B_{..,k,...b}
//Both tensors must have the same dimension, and the sizes of dimensions other that preserve_dim must all be equal
//preserve_dim:  the index of the dimension that is preserved in the output matrix (that of j, k in the above)
//out: a *device* pointer to the output matrix' underlying array, that should be *zero initialized*. Output is stored in the usual lexicographic format, for the above   k+sizek*j
template<typename FloatType, int Dim>
void batchTensorContractToMatrix_p(FloatType* out_p, const Tensor<FloatType,Dim> &A, const Tensor<FloatType,Dim> &B, const int preserve_dim, FLOPScounter *flops = nullptr);

//out_{...,j,...b} = \sum_i X_{...,i,...,b}A_{ij} 
template<typename FloatType,int Dim>
Tensor<FloatType,Dim> matrixBatchTensorContractRight(const Tensor<FloatType,Dim> &X, const Matrix<FloatType> &A, const int contract_dim, FLOPScounter *flops = nullptr);

//out_{...i,...,b} = \sum_j A_{ij} X_{..., j, ..., b}
template<typename FloatType,int Dim>
Tensor<FloatType,Dim> matrixBatchTensorContractLeft(const Matrix<FloatType> &A, const Tensor<FloatType,Dim> &X, const int contract_dim, FLOPScounter *flops = nullptr);

#include "implementation/Linalg_CPU_GPU.tcc"
#include "implementation/Linalg_CPU.tcc"
#include "implementation/Linalg_GPU.tcc"
#include "implementation/Linalg_BLAS.tcc"
#include "implementation/Linalg_GPU_BLAS.tcc"
#include "implementation/Linalg.tcc"
