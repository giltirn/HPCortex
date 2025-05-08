#include <HPCortex.hpp>
#include <Testing.hpp>

template<typename _FloatType>
struct Batch3tensorPairContractComponentWrapper{
  typedef _FloatType FloatType;
  
  Batch3tensorPairContractComponent<FloatType> &cpt;
  int C_size[3];
  int A_size[3];
  int B_size[3];

  size_t A_lin;
  size_t B_lin;
  size_t C_lin;

  Batch3tensorPairContractComponentWrapper(Batch3tensorPairContractComponent<FloatType> &cpt, int const *A_sz, int const* B_sz, int const *C_sz): cpt(cpt){
    memcpy(A_size,A_sz,3*sizeof(int));
    memcpy(B_size,B_sz,3*sizeof(int));
    memcpy(C_size,C_sz,3*sizeof(int));
    A_lin = size_t(A_size[0])*A_size[1]*A_size[2];
    B_lin = size_t(B_size[0])*B_size[1]*B_size[2];
    C_lin = size_t(C_size[0])*C_size[1]*C_size[2];
  }

  size_t outputLinearSize() const{ return C_lin; }
  size_t inputLinearSize() const{ return A_lin + B_lin; }
  
  Vector<FloatType> value(const Vector<FloatType> &in){
    Tensor<FloatType,3> A(A_size), B(B_size);
    unflatten2(A,B,in);
    Tensor<FloatType,3> C = cpt.value(A,B);
    return flatten(C);
  }
  void deriv(Vector<FloatType> &cost_deriv_params, int off, Vector<FloatType> &&_above_deriv_lin, Vector<FloatType> &cost_deriv_inputs){
    Vector<FloatType> above_deriv_lin = std::move(_above_deriv_lin);
    Tensor<FloatType,3> above_deriv(C_size);
    unflatten(above_deriv,above_deriv_lin);
    Tensor<FloatType,3> dcost_by_dA, dcost_by_dB;
    cpt.deriv(std::move(above_deriv), dcost_by_dA, dcost_by_dB);
    cost_deriv_inputs = flatten2(dcost_by_dA, dcost_by_dB);
    assert(cost_deriv_inputs.size(0) == A_lin + B_lin);
  }
    
  void update(int off, const Vector<FloatType> &new_params){}
  void step(int off, const Vector<FloatType> &derivs, FloatType eps){}
  inline int nparams() const{ return cpt.nparams(); }
  void getParams(Vector<FloatType> &into, int off){}

  std::string inCoord(size_t i) const{
    std::ostringstream ss;
    int coord[3];
    if(i < A_lin){
      tensorOffsetUnmap<3>(coord, A_size, i);
      ss << "A:";
    }else{
      tensorOffsetUnmap<3>(coord, B_size, i-A_lin);
      ss << "B:";   
    }
    ss << "(" << coord[0] << "," << coord[1] << "," << coord[2] << ")";
    return ss.str();
  }

    
    
};
    


void testBatch3tensorPairContractComponent(){
  typedef double FloatType;
  FloatType nrm = 1./3.141;
  
  //0 0
  {
    std::cout << "Contract 0 0" << std::endl;
    int A_sz[3] = {3,4,6};
    int B_sz[3] = {3,5,6};
    int C_sz[3] = {4,5,6};

    Batch3tensorPairContractComponent<FloatType> cpt(0,0,nrm);
    Batch3tensorPairContractComponentWrapper<FloatType> wrp(cpt, A_sz,B_sz,C_sz);
    testComponentDeriv(wrp);
  }
  //0 1
  {
    std::cout << "Contract 0 1" << std::endl;
    int A_sz[3] = {3,4,6};
    int B_sz[3] = {5,3,6};
    int C_sz[3] = {4,5,6};

    Batch3tensorPairContractComponent<FloatType> cpt(0,1,nrm);
    Batch3tensorPairContractComponentWrapper<FloatType> wrp(cpt, A_sz,B_sz,C_sz);
    testComponentDeriv(wrp);
  }
  //1 0
  {
    std::cout << "Contract 1 0" << std::endl;
    int A_sz[3] = {4,3,6};
    int B_sz[3] = {3,5,6};
    int C_sz[3] = {4,5,6};

    Batch3tensorPairContractComponent<FloatType> cpt(1,0,nrm);
    Batch3tensorPairContractComponentWrapper<FloatType> wrp(cpt, A_sz,B_sz,C_sz);
    testComponentDeriv(wrp);
  }
  //1 1
  {   
    int A_sz[3] = {4,3,6};
    int B_sz[3] = {5,3,6};
    int C_sz[3] = {4,5,6};

    std::cout << "Contract 1 1" << std::endl;
    Batch3tensorPairContractComponent<FloatType> cpt(1,1,nrm);
    Batch3tensorPairContractComponentWrapper<FloatType> wrp(cpt, A_sz,B_sz,C_sz);
    testComponentDeriv(wrp);
  }
  
}



int main(int argc, char** argv){
  initialize(argc,argv);
  testBatch3tensorPairContractComponent();
  return 0;
}

