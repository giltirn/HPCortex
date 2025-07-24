#include <HPCortex.hpp>
#include <Testing.hpp>

template<typename Config>
struct Batch3tensorPairContractComponentWrapper{
  EXTRACT_CONFIG_TYPES;
  
  Batch3tensorPairContractComponent<Config> &cpt;
  int C_size[3];
  int A_size[3];
  int B_size[3];

  size_t A_lin;
  size_t B_lin;
  size_t C_lin;

  Batch3tensorPairContractComponentWrapper(Batch3tensorPairContractComponent<Config> &cpt, int const *A_sz, int const* B_sz, int const *C_sz): cpt(cpt){
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
  typedef confDouble Config;
  typedef typename Config::FloatType FloatType;
  
  FloatType nrm = 1./3.141;
  
  //0 0
  {
    std::cout << "Contract 0 0" << std::endl;
    int A_sz[3] = {3,4,6};
    int B_sz[3] = {3,5,6};
    int C_sz[3] = {4,5,6};

    Batch3tensorPairContractComponent<Config> cpt(0,0,nrm);
    Batch3tensorPairContractComponentWrapper<Config> wrp(cpt, A_sz,B_sz,C_sz);
    testComponentDeriv(wrp);
  }
  //0 1
  {
    std::cout << "Contract 0 1" << std::endl;
    int A_sz[3] = {3,4,6};
    int B_sz[3] = {5,3,6};
    int C_sz[3] = {4,5,6};

    Batch3tensorPairContractComponent<Config> cpt(0,1,nrm);
    Batch3tensorPairContractComponentWrapper<Config> wrp(cpt, A_sz,B_sz,C_sz);
    testComponentDeriv(wrp);
  }
  //1 0
  {
    std::cout << "Contract 1 0" << std::endl;
    int A_sz[3] = {4,3,6};
    int B_sz[3] = {3,5,6};
    int C_sz[3] = {4,5,6};

    Batch3tensorPairContractComponent<Config> cpt(1,0,nrm);
    Batch3tensorPairContractComponentWrapper<Config> wrp(cpt, A_sz,B_sz,C_sz);
    testComponentDeriv(wrp);
  }
  //1 1
  {   
    int A_sz[3] = {4,3,6};
    int B_sz[3] = {5,3,6};
    int C_sz[3] = {4,5,6};

    std::cout << "Contract 1 1" << std::endl;
    Batch3tensorPairContractComponent<Config> cpt(1,1,nrm);
    Batch3tensorPairContractComponentWrapper<Config> wrp(cpt, A_sz,B_sz,C_sz);
    testComponentDeriv(wrp);
  }
  std::cout << "testBatch3tensorPairContractComponent passed" << std::endl;
}




template<typename Config, int TensDim>
struct BatchTensorConcatenateComponentWrapper{
  EXTRACT_CONFIG_TYPES;
  
  BatchTensorConcatenateComponent<Config,TensDim> &cpt;
  std::vector< std::array<int,TensDim> > in_sz;
  size_t lin_sz;
  int N;
  int out_sz[TensDim];
  std::vector<Tensor<FloatType,TensDim>* > tmp;
    
  BatchTensorConcatenateComponentWrapper(BatchTensorConcatenateComponent<Config,TensDim> &cpt, const std::vector< std::array<int,TensDim> > &in_sz, int concat_dim): cpt(cpt), in_sz(in_sz), N(in_sz.size()), tmp(N){
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
  typedef confDouble Config;
  typedef typename Config::FloatType FloatType;

  //4-tensor
  
  { //contract dim 2
    std::vector< std::array<int,4> > in_sz({
	  {2,3,4,5},
	  {2,3,3,5},
	  {2,3,6,5} });
        
    BatchTensorConcatenateComponent<Config,4> cpt(2,  3);
    BatchTensorConcatenateComponentWrapper<Config,4> wrp(cpt, in_sz, 2);
    testComponentDeriv(wrp);
  }

  { //contract dim 1
    std::vector< std::array<int,4> > in_sz({
	{2,4,3,5},
	{2,3,3,5},
	{2,6,3,5} });
        
    BatchTensorConcatenateComponent<Config,4> cpt(1,  3);
    BatchTensorConcatenateComponentWrapper<Config,4> wrp(cpt, in_sz, 1);
    testComponentDeriv(wrp);
  }

  { //contract dim 0
    std::vector< std::array<int,4> > in_sz({
	{4,2,3,5},
	{3,2,3,5},
	{6,2,3,5} });
        
    BatchTensorConcatenateComponent<Config,4> cpt(0,  3);
    BatchTensorConcatenateComponentWrapper<Config,4> wrp(cpt, in_sz, 0);
    testComponentDeriv(wrp);
  }


  //3-tensor
  
  { //contract dim 1
    std::vector< std::array<int,3> > in_sz({
	{2,4,5},
	{2,3,5},
	{2,6,5} });
        
    BatchTensorConcatenateComponent<Config,3> cpt(1,  3);
    BatchTensorConcatenateComponentWrapper<Config,3> wrp(cpt, in_sz, 1);
    testComponentDeriv(wrp);
  }

  { //contract dim 0
    std::vector< std::array<int,3> > in_sz({
	{4,2,5},
	{3,2,5},
	{6,2,5} });
        
    BatchTensorConcatenateComponent<Config,3> cpt(0,  3);
    BatchTensorConcatenateComponentWrapper<Config,3> wrp(cpt, in_sz, 0);
    testComponentDeriv(wrp);
  }
  
  std::cout << "testBatchTensorConcatenateComponent passed" << std::endl;
}


template<typename Config, int TensDim>
struct ScaleComponentWrapper{
  EXTRACT_CONFIG_TYPES;
  
  ScaleComponent<Config,TensDim> &cpt;
  int size[TensDim];
  size_t size_lin;

  ScaleComponentWrapper(ScaleComponent<Config,TensDim> &cpt, int const *sz): cpt(cpt){
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
    tensorOffsetUnmap<TensDim>(coord, size, i);
    ss << "(";
    for(int c=0;c<TensDim;c++)
      ss << coord[c] << (c<TensDim-1 ? ", " : "");
    ss << ")";
    return ss.str();
  }

    
    
};

template<typename FloatType>
std::vector<FloatType> scale_lin(const std::vector<FloatType> &v, const Vector<FloatType> &gamma, const Vector<FloatType> &beta){
  std::vector<FloatType> out(v.size());
  doHost2(gamma,beta, {
  for(int i=0;i<v.size();i++)
    out[i] = v[i]*gamma_v(i) + beta_v(i);
    });
  return out;
}

void testScaleComponent(){
  typedef confDouble Config;
  typedef typename Config::FloatType FloatType;
  
  std::mt19937 rng(1234);

  int size[4] = {2,3,4,5};
  
  Tensor<FloatType,4> v(size);
  uniformRandom(v,rng);
 
  {
    //dim 0
    Vector<FloatType> gamma(size[0]), beta(size[0]);
    uniformRandom(gamma,rng); uniformRandom(beta,rng);
    
    ScaleComponent<Config,4> cpt(0,size[0],true,true,gamma,beta);
    Tensor<FloatType,4> got = cpt.value(v);
    Tensor<FloatType,4> expect(size);

    doHost2(v,expect, {
	for(int j=0;j<size[1];j++)
	  for(int k=0;k<size[2];k++)
	    for(int b=0;b<size[3];b++){
	      std::vector<FloatType> l_in(size[0]);
	      for(int i=0;i<size[0];i++)
		l_in[i] = v_v(i,j,k,b);
	      std::vector<FloatType> l_out = scale_lin(l_in, gamma, beta);
	      for(int i=0;i<size[0];i++)
		expect_v(i,j,k,b) = l_out[i];
	    }
      });
    assert(abs_near(got,expect,FloatType(1e-5),true));

    for(int use_gamma=0; use_gamma<2; use_gamma++){
      for(int use_beta=0; use_beta<2; use_beta++){
	std::cout << "use_gamma: " << use_gamma << " use_beta: " << use_beta << std::endl;
	ScaleComponent<Config,4> cpta(0,size[0],bool(use_gamma),bool(use_beta),gamma,beta);
	assert(cpta.nparams() == (use_gamma + use_beta) * size[0]);
	
	ScaleComponentWrapper<Config,4> wrp(cpta,size);
	testComponentDeriv(wrp, FloatType(1e-7));
      }
    }
    
  }

  {
    //dim 1
    Vector<FloatType> gamma(size[1]), beta(size[1]);
    uniformRandom(gamma,rng); uniformRandom(beta,rng);
    
    ScaleComponent<Config,4> cpt(1,size[1],true,true,gamma,beta);
    Tensor<FloatType,4> got = cpt.value(v);
    Tensor<FloatType,4> expect(size);

    doHost2(v,expect, {
	for(int i=0;i<size[0];i++)
	  for(int k=0;k<size[2];k++)
	    for(int b=0;b<size[3];b++){
	      std::vector<FloatType> l_in(size[1]);
	      for(int j=0;j<size[1];j++)
		l_in[j] = v_v(i,j,k,b);
	      std::vector<FloatType> l_out = scale_lin(l_in, gamma, beta);
	      for(int j=0;j<size[1];j++)
		expect_v(i,j,k,b) = l_out[j];
	    }
      });
    assert(abs_near(got,expect,FloatType(1e-5),true));

    for(int use_gamma=0; use_gamma<2; use_gamma++){
      for(int use_beta=0; use_beta<2; use_beta++){
	std::cout << "use_gamma: " << use_gamma << " use_beta: " << use_beta << std::endl;
	ScaleComponent<Config,4> cpta(1,size[1],bool(use_gamma),bool(use_beta),gamma,beta);
	assert(cpta.nparams() == (use_gamma + use_beta) * size[1]);
	
	ScaleComponentWrapper<Config,4> wrp(cpta,size);
	testComponentDeriv(wrp, FloatType(1e-7));
      }
    }
    
  }

  {
    //dim 2
    Vector<FloatType> gamma(size[2]), beta(size[2]);
    uniformRandom(gamma,rng); uniformRandom(beta,rng);
    
    ScaleComponent<Config,4> cpt(2,size[2],true,true,gamma,beta);
    Tensor<FloatType,4> got = cpt.value(v);
    Tensor<FloatType,4> expect(size);

    doHost2(v,expect, {
	for(int i=0;i<size[0];i++)
	  for(int j=0;j<size[1];j++)
	    for(int b=0;b<size[3];b++){
	      std::vector<FloatType> l_in(size[2]);
	      for(int k=0;k<size[2];k++)
		l_in[k] = v_v(i,j,k,b);
	      std::vector<FloatType> l_out = scale_lin(l_in, gamma, beta);
	      for(int k=0;k<size[2];k++)
		expect_v(i,j,k,b) = l_out[k];
	    }
      });
    assert(abs_near(got,expect,FloatType(1e-5),true));

    for(int use_gamma=0; use_gamma<2; use_gamma++){
      for(int use_beta=0; use_beta<2; use_beta++){
	std::cout << "use_gamma: " << use_gamma << " use_beta: " << use_beta << std::endl;
	ScaleComponent<Config,4> cpta(2,size[2],bool(use_gamma),bool(use_beta),gamma,beta);
	assert(cpta.nparams() == (use_gamma + use_beta) * size[2]);
	
	ScaleComponentWrapper<Config,4> wrp(cpta,size);
	testComponentDeriv(wrp, FloatType(1e-7));
      }
    }
    
  }
  
  std::cout << "testScaleComponent passed" << std::endl;
}


template<typename Config, int TensDim>
struct BatchTensorDimensionSliceComponentWrapper{
  EXTRACT_CONFIG_TYPES;
  
  BatchTensorDimensionSliceComponent<Config,TensDim> &cpt;
  int in_sz[TensDim];
  int out_sz[TensDim-1];
  size_t in_lin_sz;
  size_t out_lin_sz;

    
  BatchTensorDimensionSliceComponentWrapper(BatchTensorDimensionSliceComponent<Config,TensDim> &cpt, int const* _in_sz, int const* _out_sz): cpt(cpt){
    memcpy(in_sz,_in_sz,TensDim*sizeof(int));
    memcpy(out_sz,_out_sz,(TensDim-1)*sizeof(int));
    in_lin_sz = 1;
    for(int d=0;d<TensDim;d++)
      in_lin_sz *= in_sz[d];

    out_lin_sz = 1;
    for(int d=0;d<TensDim-1;d++)
      out_lin_sz *= out_sz[d];
  }
  
  size_t outputLinearSize() const{ return out_lin_sz; }
  size_t inputLinearSize() const{ return in_lin_sz; }
  
  Vector<FloatType> value(const Vector<FloatType> &in){
    Tensor<FloatType,TensDim> in_t(in_sz);
    unflatten(in_t,in); 
    Tensor<FloatType,TensDim-1> out = cpt.value(in_t);
    return flatten(out);
  }
  void deriv(Vector<FloatType> &cost_deriv_params, int off, Vector<FloatType> &&_above_deriv_lin, Vector<FloatType> &cost_deriv_inputs){
    Vector<FloatType> above_deriv_lin = std::move(_above_deriv_lin);
    Tensor<FloatType,TensDim-1> above_deriv(out_sz);
    unflatten(above_deriv,above_deriv_lin);
    Tensor<FloatType,TensDim> tmp(in_sz);
    cpt.deriv(std::move(above_deriv), tmp);
    cost_deriv_inputs = flatten(tmp);
  }
    
  void update(int off, const Vector<FloatType> &new_params){}
  void step(int off, const Vector<FloatType> &derivs, FloatType eps){}
  inline int nparams() const{ return cpt.nparams(); }
  void getParams(Vector<FloatType> &into, int off){}

  std::string inCoord(size_t i) const{
    return std::to_string(i);
  }

};



void testBatchTensorDimensionSliceComponent(){
  typedef confDouble Config;
  typedef typename Config::FloatType FloatType;
  
  std::mt19937 rng(1234);

  int size[4] = {2,3,4,5};
  
  Tensor<FloatType,4> v(size);
  uniformRandom(v,rng);

  {
    //dim 0
    int size_out[3] = {size[1],size[2],size[3]};
    for(int slice_idx=0;slice_idx<size[0];slice_idx++){
      BatchTensorDimensionSliceComponent<Config,4> cpt(0,slice_idx);      
      Tensor<FloatType,3> got = cpt.value(v);
      Tensor<FloatType,3> expect(size_out);
      doHost2(expect,v,{     
	  for(int j=0;j<size[1];j++)
	    for(int k=0;k<size[2];k++)
	      for(int b=0;b<size[3];b++)
		expect_v(j,k,b) = v_v(slice_idx,j,k,b);
	});
      assert(equal(got,expect,true));
      BatchTensorDimensionSliceComponentWrapper<Config,4> wrp(cpt, v.sizeArray(),expect.sizeArray());
      testComponentDeriv(wrp);
    }
    
  }
  {
    //dim 1
    int size_out[3] = {size[0],size[2],size[3]};
    for(int slice_idx=0;slice_idx<size[1];slice_idx++){
      BatchTensorDimensionSliceComponent<Config,4> cpt(1,slice_idx);      
      Tensor<FloatType,3> got = cpt.value(v);     
      Tensor<FloatType,3> expect(size_out);
      doHost2(expect,v,{
	  for(int i=0;i<size[0];i++)
	    for(int k=0;k<size[2];k++)
	      for(int b=0;b<size[3];b++)
		expect_v(i,k,b) = v_v(i,slice_idx,k,b);
	});
      assert(equal(got,expect,true));
      BatchTensorDimensionSliceComponentWrapper<Config,4> wrp(cpt, v.sizeArray(),expect.sizeArray());
      testComponentDeriv(wrp);
    }
    
  }
    
  {
    //dim 2
    int size_out[3] = {size[0],size[1],size[3]};
    for(int slice_idx=0;slice_idx<size[2];slice_idx++){
      BatchTensorDimensionSliceComponent<Config,4> cpt(2,slice_idx);      
      Tensor<FloatType,3> got = cpt.value(v);
      Tensor<FloatType,3> expect(size_out);
      doHost2(expect,v,{     
      for(int i=0;i<size[0];i++)
	for(int j=0;j<size[1];j++)
	  for(int b=0;b<size[3];b++)
	    expect_v(i,j,b) = v_v(i,j,slice_idx,b);
	});
      assert(equal(got,expect,true));
      BatchTensorDimensionSliceComponentWrapper<Config,4> wrp(cpt, v.sizeArray(),expect.sizeArray());
      testComponentDeriv(wrp);
    }
    
  }
  
  std::cout << "testBatchTensorDimensionSliceComponent passed" << std::endl;

}


template<typename FloatType>
Tensor<FloatType,4> MatrixTensorContractComponentExpect(const Matrix<FloatType> &weights, const Tensor<FloatType,4> &in){
  int out_sizes[4];
  memcpy(out_sizes,in.sizeArray(),4*sizeof(int));
  int fan_out = weights.size(0);
  out_sizes[2] = fan_out;
  Tensor<FloatType,4> out(out_sizes,0.);
  
  autoView(weights_v,weights,HostRead);
  autoView(in_v,in,HostRead);
  autoView(out_v,out,HostReadWrite);

  for(int i=0;i<in.size(0);i++)
    for(int j=0;j<in.size(1);j++)
      for(int l=0;l<in.size(3);l++)
	for(int k=0;k<fan_out;k++){
	  
	  for(int kk=0;kk<in.size(2);kk++)
	    out_v(i,j,k,l) += weights_v(k,kk) * in_v(i,j,kk,l);
	}
  
  return out;
}


template<typename Config, int Dim>
struct MatrixTensorContractComponentWrapper{
  EXTRACT_CONFIG_TYPES;
  
  MatrixTensorContractComponent<Config,Dim> &cpt;
  int in_size[Dim];
  size_t in_size_lin;
  int out_size[Dim];
  size_t out_size_lin;
  

  MatrixTensorContractComponentWrapper(MatrixTensorContractComponent<Config,Dim> &cpt, int const *in_sz, int const *out_sz): cpt(cpt){
    memcpy(in_size,in_sz,Dim*sizeof(int));
    memcpy(out_size,out_sz,Dim*sizeof(int));
    in_size_lin = out_size_lin = 1;
    for(int d=0;d<Dim;d++){
      in_size_lin *= in_sz[d];
      out_size_lin *= out_sz[d];
    }
  }

  size_t outputLinearSize() const{ return out_size_lin; }
  size_t inputLinearSize() const{ return in_size_lin; }
  
  Vector<FloatType> value(const Vector<FloatType> &in){
    Tensor<FloatType,Dim> A(in_size);
    unflatten(A,in);
    Tensor<FloatType,Dim> C = cpt.value(A);
    return flatten(C);
  }
  void deriv(Vector<FloatType> &cost_deriv_params, int off, Vector<FloatType> &&_above_deriv_lin, Vector<FloatType> &cost_deriv_inputs){
    Vector<FloatType> above_deriv_lin = std::move(_above_deriv_lin);
    Tensor<FloatType,Dim> above_deriv(out_size);
    unflatten(above_deriv,above_deriv_lin);
    Tensor<FloatType,Dim> dcost_by_dIn;
    cpt.deriv(cost_deriv_params, off , std::move(above_deriv), dcost_by_dIn);
    cost_deriv_inputs = flatten(dcost_by_dIn);
  }
    
  void update(int off, const Vector<FloatType> &new_params){  cpt.update(off,new_params); }
  void step(int off, const Vector<FloatType> &derivs, FloatType eps){ cpt.step(off,derivs,eps); }
  inline int nparams() const{ return cpt.nparams(); }
  void getParams(Vector<FloatType> &into, int off){ cpt.getParams(into,off); }

  std::string inCoord(size_t i) const{
    std::ostringstream ss;
    int coord[Dim];
    tensorOffsetUnmap<Dim>(coord, in_size, i);
    ss << "(";
    for(int d=0;d<Dim;d++)
      ss << coord[d] << (d==Dim-1? ")" : ", ");
    return ss.str();
  }       
};

void testMatrixTensorContractComponent(){
  typedef confDouble Config;
  typedef typename Config::FloatType FloatType;
  std::mt19937 rng(1234);
    
  int tens_sizes[4] = {2,3,4,5};
  int out_size = 6;

  Tensor<FloatType,4> x(tens_sizes);
  uniformRandom(x,rng);

  int contract_dim = 2; //fixed
  
  Matrix<FloatType> weights(out_size,tens_sizes[contract_dim]);
  uniformRandom(weights,rng);

  MatrixTensorContractComponent<Config,4> cpt(weights);
    
  int nparam_expect =  out_size*tens_sizes[contract_dim];
  std::cout << "Nparam " << cpt.nparams() << " expect " << nparam_expect << std::endl;
  assert(cpt.nparams() == nparam_expect);
      
  Tensor<FloatType,4> got = cpt.value(x);
  Tensor<FloatType,4> expect = MatrixTensorContractComponentExpect(weights,x);
  assert(abs_near(got,expect, 1e-6, true));

  MatrixTensorContractComponentWrapper<Config,4> wrp(cpt, x.sizeArray(),expect.sizeArray());
  std::cout << "Test component deriv" << std::endl;
  testComponentDeriv(wrp);

  std::cout << "testMatrixTensorContractComponent passed" << std::endl;
}



int main(int argc, char** argv){
  initialize(argc,argv);
  testBatch3tensorPairContractComponent();
  testBatchTensorConcatenateComponent();
  testScaleComponent();
  testBatchTensorDimensionSliceComponent();
  testMatrixTensorContractComponent();
  return 0;
}

