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
  std::cout << "testBatch3tensorPairContractComponent passed" << std::endl;
}




template<typename _FloatType, int TensDim>
struct BatchTensorConcatenateComponentWrapper{
  typedef _FloatType FloatType;
  
  BatchTensorConcatenateComponent<FloatType,TensDim> &cpt;
  std::vector< std::array<int,TensDim> > in_sz;
  size_t lin_sz;
  int N;
  int out_sz[TensDim];
  std::vector<Tensor<FloatType,TensDim>* > tmp;
    
  BatchTensorConcatenateComponentWrapper(BatchTensorConcatenateComponent<FloatType,TensDim> &cpt, const std::vector< std::array<int,TensDim> > &in_sz, int concat_dim): cpt(cpt), in_sz(in_sz), N(in_sz.size()), tmp(N){
    for(int d=0;d<TensDim;d++)      
      out_sz[d] = in_sz[0][d];
    for(int t=1;t<N;t++)
      out_sz[concat_dim] += in_sz[t][concat_dim];

    lin_sz=1;
    for(int d=0;d<TensDim;d++) lin_sz *= out_sz[d];
    
    for(int t=0;t<N;t++) tmp[t] = new Tensor<FloatType,TensDim>(in_sz[t].data());
  }
  ~BatchTensorConcatenateComponentWrapper(){
    for(int t=0;t<N;t++) delete tmp[t];
  }
  
  size_t outputLinearSize() const{ return lin_sz; }
  size_t inputLinearSize() const{ return lin_sz; }
  
  Vector<FloatType> value(const Vector<FloatType> &in){
    unflattenNsameDim(tmp.data(),N,in);    
    Tensor<FloatType,TensDim> out = cpt.value(tmp.data());
    return flatten(out);
  }
  void deriv(Vector<FloatType> &cost_deriv_params, int off, Vector<FloatType> &&_above_deriv_lin, Vector<FloatType> &cost_deriv_inputs){
    Vector<FloatType> above_deriv_lin = std::move(_above_deriv_lin);
    Tensor<FloatType,TensDim> above_deriv(out_sz);
    unflatten(above_deriv,above_deriv_lin);
    cpt.deriv(std::move(above_deriv), tmp.data());
    cost_deriv_inputs = flattenNsameDim(tmp.data(),N);
  }
    
  void update(int off, const Vector<FloatType> &new_params){}
  void step(int off, const Vector<FloatType> &derivs, FloatType eps){}
  inline int nparams() const{ return cpt.nparams(); }
  void getParams(Vector<FloatType> &into, int off){}

  std::string inCoord(size_t i) const{
    return std::to_string(i);
  }

    
    
};


void testBatchTensorConcatenateComponent(){
  typedef double FloatType;
 
  { //contract dim 2
    std::vector< std::array<int,4> > in_sz({
	  {2,3,4,5},
	  {2,3,3,5},
	  {2,3,6,5} });
        
    BatchTensorConcatenateComponent<FloatType,4> cpt(2,  3);
    BatchTensorConcatenateComponentWrapper<FloatType,4> wrp(cpt, in_sz, 2);
    testComponentDeriv(wrp);
  }

  { //contract dim 1
    std::vector< std::array<int,4> > in_sz({
	{2,4,3,5},
	{2,3,3,5},
	{2,6,3,5} });
        
    BatchTensorConcatenateComponent<FloatType,4> cpt(1,  3);
    BatchTensorConcatenateComponentWrapper<FloatType,4> wrp(cpt, in_sz, 1);
    testComponentDeriv(wrp);
  }

  { //contract dim 0
    std::vector< std::array<int,4> > in_sz({
	{4,2,3,5},
	{3,2,3,5},
	{6,2,3,5} });
        
    BatchTensorConcatenateComponent<FloatType,4> cpt(0,  3);
    BatchTensorConcatenateComponentWrapper<FloatType,4> wrp(cpt, in_sz, 0);
    testComponentDeriv(wrp);
  }
  std::cout << "testBatchTensorConcatenateComponent passed" << std::endl;
}


template<typename _FloatType, int TensDim>
struct NormComponentWrapper{
  typedef _FloatType FloatType;
  
  NormComponent<FloatType,TensDim> &cpt;
  int size[TensDim];
  size_t size_lin;

  NormComponentWrapper(NormComponent<FloatType,TensDim> &cpt, int const *sz): cpt(cpt){
    memcpy(size,sz,TensDim*sizeof(int));
    size_lin = 1;
    for(int i=0;i<TensDim;i++)
      size_lin *= sz[i];
  }

  size_t outputLinearSize() const{ return size_lin; }
  size_t inputLinearSize() const{ return size_lin; }
  
  Vector<FloatType> value(const Vector<FloatType> &in){
    Tensor<FloatType,TensDim> T(size);
    unflatten(T,in);
    return flatten(cpt.value(T));
  }
  void deriv(Vector<FloatType> &cost_deriv_params, int off, Vector<FloatType> &&_above_deriv_lin, Vector<FloatType> &cost_deriv_inputs){
    Vector<FloatType> above_deriv_lin = std::move(_above_deriv_lin);
    Tensor<FloatType,TensDim> above_deriv(size);
    unflatten(above_deriv,above_deriv_lin);
    Tensor<FloatType,TensDim> dcost_by_dIn;
    cpt.deriv(cost_deriv_params, off, std::move(above_deriv), dcost_by_dIn);
    cost_deriv_inputs = flatten(dcost_by_dIn);
  }
    
  void update(int off, const Vector<FloatType> &new_params){
    cpt.update(off,new_params);
  }
  void step(int off, const Vector<FloatType> &derivs, FloatType eps){
    cpt.step(off,derivs,eps);
  }
  inline int nparams() const{ return cpt.nparams(); }
  void getParams(Vector<FloatType> &into, int off){
    cpt.getParams(into, off);
  }

  std::string inCoord(size_t i) const{
    std::ostringstream ss;
    int coord[TensDim];
    tensorOffsetUnmap<4>(coord, size, i);
    ss << "(";
    for(int c=0;c<TensDim;c++)
      ss << coord[c] << (c<TensDim-1 ? ", " : "");
    ss << ")";
    return ss.str();
  }

    
    
};

template<typename FloatType>
std::vector<FloatType> norm_lin(const std::vector<FloatType> &v, FloatType eps, FloatType gamma, FloatType beta){
  size_t len = v.size();
  FloatType mean = 0., var = 0.;
  for(auto vv : v){
    mean += vv;
    var += vv*vv;
  }
  mean /= len;
  var = var/len - mean*mean;
  FloatType std = sqrt(var + eps);

  std::vector<FloatType> out(v);
  for(auto &vv : out)
    vv = gamma * (vv - mean)/std + beta;
  return out;
}

void testNormComponent(){
  typedef double FloatType;
  std::mt19937 rng(1234);

  int size[4] = {2,3,4,5};
  
  Tensor<FloatType,4> v(size);
  random(v,rng);
  FloatType eps = 0.55;
  FloatType gamma = 0.234;
  FloatType beta = 0.1234;
  
  {
    //dim 0
    NormComponent<FloatType,4> cpt(0,true,true,gamma,beta,eps);
    Tensor<FloatType,4> got = cpt.value(v);
    Tensor<FloatType,4> expect(size);

    doHost2(v,expect, {
	for(int j=0;j<size[1];j++)
	  for(int k=0;k<size[2];k++)
	    for(int b=0;b<size[3];b++){
	      std::vector<FloatType> l_in(size[0]);
	      for(int i=0;i<size[0];i++)
		l_in[i] = v_v(i,j,k,b);
	      std::vector<FloatType> l_out = norm_lin(l_in, eps, gamma, beta);
	      for(int i=0;i<size[0];i++)
		expect_v(i,j,k,b) = l_out[i];
	    }
      });
    assert(abs_near(got,expect,FloatType(1e-5),true));

    for(int use_gamma=0; use_gamma<2; use_gamma++){
      for(int use_beta=0; use_beta<2; use_beta++){
	std::cout << "use_gamma: " << use_gamma << " use_beta: " << use_beta << std::endl;
	NormComponent<FloatType,4> cpta(0,bool(use_gamma),bool(use_beta),gamma,beta,eps);
	NormComponentWrapper<FloatType,4> wrp(cpta,size);
	testComponentDeriv(wrp, FloatType(1e-7));
      }
    }
    
  }

  {
    //dim 1
    NormComponent<FloatType,4> cpt(1,true,true,gamma,beta,eps);
    Tensor<FloatType,4> got = cpt.value(v);
    Tensor<FloatType,4> expect(size);

    doHost2(v,expect, {
	for(int i=0;i<size[0];i++)
	  for(int k=0;k<size[2];k++)
	    for(int b=0;b<size[3];b++){
	      std::vector<FloatType> l_in(size[1]);
	      for(int j=0;j<size[1];j++)
		l_in[j] = v_v(i,j,k,b);
	      std::vector<FloatType> l_out = norm_lin(l_in, eps, gamma, beta);
	      for(int j=0;j<size[1];j++)
		expect_v(i,j,k,b) = l_out[j];
	    }
      });
    assert(abs_near(got,expect,FloatType(1e-5),true));

    for(int use_gamma=0; use_gamma<2; use_gamma++){
      for(int use_beta=0; use_beta<2; use_beta++){
	std::cout << "use_gamma: " << use_gamma << " use_beta: " << use_beta << std::endl;
	NormComponent<FloatType,4> cpta(1,bool(use_gamma),bool(use_beta),gamma,beta,eps);
	NormComponentWrapper<FloatType,4> wrp(cpta,size);
	testComponentDeriv(wrp, FloatType(1e-7));
      }
    }
    
  }

  {
    //dim 2
    NormComponent<FloatType,4> cpt(2,true,true,gamma,beta,eps);
    Tensor<FloatType,4> got = cpt.value(v);
    Tensor<FloatType,4> expect(size);

    doHost2(v,expect, {
	for(int i=0;i<size[0];i++)
	  for(int j=0;j<size[1];j++)
	    for(int b=0;b<size[3];b++){
	      std::vector<FloatType> l_in(size[2]);
	      for(int k=0;k<size[2];k++)
		l_in[k] = v_v(i,j,k,b);
	      std::vector<FloatType> l_out = norm_lin(l_in, eps, gamma, beta);
	      for(int k=0;k<size[2];k++)
		expect_v(i,j,k,b) = l_out[k];
	    }
      });
    assert(abs_near(got,expect,FloatType(1e-5),true));

    for(int use_gamma=0; use_gamma<2; use_gamma++){
      for(int use_beta=0; use_beta<2; use_beta++){
	std::cout << "use_gamma: " << use_gamma << " use_beta: " << use_beta << std::endl;
	NormComponent<FloatType,4> cpta(2,bool(use_gamma),bool(use_beta),gamma,beta,eps);
	NormComponentWrapper<FloatType,4> wrp(cpta,size);
	testComponentDeriv(wrp, FloatType(1e-7));
      }
    }
    
  }
  
  std::cout << "testNormComponent passed" << std::endl;
}

int main(int argc, char** argv){
  initialize(argc,argv);
  //testBatch3tensorPairContractComponent();
  //testBatchTensorConcatenateComponent();
  testNormComponent();
  
  return 0;
}

