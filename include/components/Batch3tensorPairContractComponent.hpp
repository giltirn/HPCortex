#pragma once
#include <type_traits>
#include <sstream>
#include <Tensors.hpp>
#include <RingBuffer.hpp>
#include <Linalg.hpp>

//A component implementing the contraction of batched 3-tensors (those for whom the last dimension is the batch index) over some dimension \in [0,1]
//eg  C_{ijb} = \sum_k A_{kib} B_{jkb} * nrm
//it has no trainable parameters
template<typename _FloatType>
class Batch3tensorPairContractComponent{
public:
  typedef _FloatType FloatType;
private:
  FloatType nrm;
  int contract_dim_A;
  int contract_dim_B;
  mutable FLOPScounter value_FLOPS;
  mutable FLOPScounter deriv_FLOPS;
  
  mutable RingBuffer<Tensor<FloatType,3> > A_buf;
  mutable RingBuffer<Tensor<FloatType,3> > B_buf;
public:
  
  Batch3tensorPairContractComponent(int contract_dim_A, int contract_dim_B, FloatType nrm = 1.0): contract_dim_A(contract_dim_A), contract_dim_B(contract_dim_B), nrm(nrm){}
  Batch3tensorPairContractComponent(const Batch3tensorPairContractComponent &r) = delete;
  Batch3tensorPairContractComponent(Batch3tensorPairContractComponent &&r) = default;
  
  //Forward pass
  Tensor<FloatType,3> value(const Tensor<FloatType,3> &A, const Tensor<FloatType,3> &B){   
    A_buf.push(Tensor<FloatType,3>(A));
    B_buf.push(Tensor<FloatType,3>(B));
    auto out = batch3tensorContract(A,B,contract_dim_A,contract_dim_B,nrm,&value_FLOPS);
    value_FLOPS.lock();
    return out;
  }  
  void deriv(Tensor<FloatType,3> &&_dcost_by_dC, Tensor<FloatType,3> &dcost_by_dA, Tensor<FloatType,3> &dcost_by_dB) const{
    Tensor<FloatType,3> dcost_by_dC = std::move(_dcost_by_dC);
    Tensor<FloatType,3> A = A_buf.isFilled() ? A_buf.pop() : Tensor<FloatType,3>(A_buf.latest());
    Tensor<FloatType,3> B = B_buf.isFilled() ? B_buf.pop() : Tensor<FloatType,3>(B_buf.latest());

    //e.g. for 0,0 : C_{ijb} = \sum_k A_{kib} B_{kjb} * nrm   and example cost
    //cost = \sum_{ijb} c_{ijb} C_{ijb} =  \sum_{ijbk} c_{ijb} A_{kib} B_{kjb} * nrm
    //dcost/dC_{i'j'b'} = c_{i'j'b'}
    
    //dcost/dA_{k'i'b'} =  \sum_{j} c_{i'jb'} B_{k'jb'} * nrm = \sum_{j} [ dcost/dC_{i'jb'} ] B_{k'jb'} * nrm
    //thus:
    
    //0 0
    //C_{ijb} = \sum_k A_{kib} B_{kjb} * nrm
    //dcost/dA_{kib} = \sum_j [ dcost/dC_{ijb} ] B_{kjb} * nrm   : 1,1
    //dcost/dB_{kjb} = \sum_i [ dcost/dC_{ijb} ] A_{kib} * nrm   : 0,1
    
    //0 1
    //C_{ijb} = \sum_k A_{kib} B_{jkb} * nrm
    //dcost/dA_{kib} = \sum_j [ dcost/dC_{ijb} ] B_{jkb} * nrm   :  1,0
    //dcost/dB_{jkb} = \sum_i [ dcost/dC_{ijb} ] A_{kib} * nrm   :  0,1
    
    //1 0
    //C_{ijb} = \sum_k A_{ikb} B_{kjb} * nrm
    //dcost/dA_{ikb} = \sum_j [ dcost/dC_{ijb} ] B_{kjb} * nrm   : 1,1
    //dcost/dB_{kjb} = \sum_i [ dcost/dC_{ijb} ] A_{ikb} * nrm   : 0,0

    //1 1
    //C_{ijb} = \sum_k A_{ikb} B_{jkb} * nrm
    //dcost/dA_{ikb} = \sum_j [ dcost/dC_{ijb} ] B_{jkb} * nrm   : 1,0
    //dcost/dB_{jkb} = \sum_i [ dcost/dC_{ijb} ] A_{ikb} * nrm   : 0,0

    //for some we need to swap the order of the arguments to ensure the output indices are in the correct order (could just transpose the output but that would cost us a copy)
    //dimA dimB transdA  transdB
    // 0    0    true     true
    // 0    1    true     false
    // 1    0    false    true
    // 1    1    false    false
    
    bool transpose_dA = !contract_dim_A;
    bool transpose_dB = !contract_dim_B; 
    
    dcost_by_dA = transpose_dA     ?     batch3tensorContract(B, dcost_by_dC, 1-contract_dim_B, 1, nrm, &deriv_FLOPS)   :   batch3tensorContract(dcost_by_dC, B, 1, 1-contract_dim_B, nrm, &deriv_FLOPS);
    dcost_by_dB = transpose_dB     ?     batch3tensorContract(A, dcost_by_dC, 1-contract_dim_A, 0, nrm, &deriv_FLOPS)   :   batch3tensorContract(dcost_by_dC, A, 0, 1-contract_dim_A, nrm, &deriv_FLOPS);
    deriv_FLOPS.lock();
  }
    
  inline int nparams() const{ return 0; }

  size_t FLOPS(int value_or_deriv) const{ return value_or_deriv == 0 ? value_FLOPS.value() : deriv_FLOPS.value(); }
  
  //For pipeliningin
  inline void resizeInputBuffer(size_t to){
    A_buf.resize(to);
    B_buf.resize(to);
  }

};
