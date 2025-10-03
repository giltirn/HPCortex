#include <HPCortex.hpp>
#include <Testing.hpp>

template<typename Config>
struct Batch3tensorPairContractComponentWrapper{
  EXTRACT_CONFIG_TYPES;
  
  Batch3tensorPairContractComponent<Config> &cpt;
  int C_size[2];
  int A_size[2];
  int B_size[2];

  size_t A_lin;
  size_t B_lin;
  size_t C_lin;

  Batch3tensorPairContractComponentWrapper(Batch3tensorPairContractComponent<Config> &cpt, int const *A_sz, int const* B_sz, int const *C_sz): cpt(cpt){
    memcpy(A_size,A_sz,2*sizeof(int));
    memcpy(B_size,B_sz,2*sizeof(int));
    memcpy(C_size,C_sz,2*sizeof(int));
    A_lin = size_t(A_size[0])*A_size[1];
    B_lin = size_t(B_size[0])*B_size[1];
    C_lin = size_t(C_size[0])*C_size[1];
  }

  size_t outputLinearSize() const{ return C_lin; }
  size_t inputLinearSize() const{ return A_lin + B_lin; }
  
  Matrix<FloatType> value(const Matrix<FloatType> &in, EnableDeriv enable_deriv = DerivNo){
    int batch_size = in.size(1);
    int A_size3[3] = {A_size[0],A_size[1],batch_size};
    int B_size3[3] = {B_size[0],B_size[1],batch_size};
    Tensor<FloatType,3> A(A_size3), B(B_size3);
    unflattenFromBatchVector(A,in,0);
    unflattenFromBatchVector(B,in,A_lin);
    Tensor<FloatType,3> C = cpt.value(A,B,enable_deriv);
    return flattenToBatchVector(C);
  }
  void deriv(Vector<FloatType> &cost_deriv_params, int off, Matrix<FloatType> &&_above_deriv_lin, Matrix<FloatType> &cost_deriv_inputs){
    Matrix<FloatType> above_deriv_lin = std::move(_above_deriv_lin);
    int batch_size = above_deriv_lin.size(1);
    int C_size3[3] = {C_size[0],C_size[1],batch_size};
    Tensor<FloatType,3> above_deriv = unflattenFromBatchVector<3>(above_deriv_lin,C_size3);
    Tensor<FloatType,3> dcost_by_dA, dcost_by_dB;
    cpt.deriv(std::move(above_deriv), dcost_by_dA, dcost_by_dB);
    cost_deriv_inputs = Matrix<FloatType>(A_lin+B_lin,batch_size);
    flattenToBatchVector(cost_deriv_inputs, dcost_by_dA, 0);
    flattenToBatchVector(cost_deriv_inputs, dcost_by_dB, A_lin);
  }
    
  void update(int off, const Vector<FloatType> &new_params){}
  void step(int off, const Vector<FloatType> &derivs, FloatType eps){}
  inline int nparams() const{ return cpt.nparams(); }
  void getParams(Vector<FloatType> &into, int off){}

  std::string inCoord(size_t i, int b, int batch_size) const{
    int A_size3[3] = {A_size[0],A_size[1],batch_size};
    int B_size3[3] = {B_size[0],B_size[1],batch_size};
    
    std::ostringstream ss;
    int coord[3];
    if(i < A_lin){
      tensorOffsetUnmap<3>(coord, A_size3, b+batch_size*i);
      ss << "A:";
    }else{
      tensorOffsetUnmap<3>(coord, B_size3, b+batch_size*(i-A_lin));
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

  int batch_size=6;
  
  //0 0
  {
    std::cout << "Contract 0 0" << std::endl;
    int A_sz[2] = {3,4};
    int B_sz[2] = {3,5};
    int C_sz[2] = {4,5};

    Batch3tensorPairContractComponent<Config> cpt(0,0,nrm);
    Batch3tensorPairContractComponentWrapper<Config> wrp(cpt, A_sz,B_sz,C_sz);
    testComponentDeriv(wrp,batch_size);
    testComponentDiffBatchSizes(wrp);
  }
  //0 1
  {
    std::cout << "Contract 0 1" << std::endl;
    int A_sz[2] = {3,4};
    int B_sz[2] = {5,3};
    int C_sz[2] = {4,5};

    Batch3tensorPairContractComponent<Config> cpt(0,1,nrm);
    Batch3tensorPairContractComponentWrapper<Config> wrp(cpt, A_sz,B_sz,C_sz);
    testComponentDeriv(wrp,batch_size);
    testComponentDiffBatchSizes(wrp);
  }
  //1 0
  {
    std::cout << "Contract 1 0" << std::endl;
    int A_sz[2] = {4,3};
    int B_sz[2] = {3,5};
    int C_sz[2] = {4,5};

    Batch3tensorPairContractComponent<Config> cpt(1,0,nrm);
    Batch3tensorPairContractComponentWrapper<Config> wrp(cpt, A_sz,B_sz,C_sz);
    testComponentDeriv(wrp,batch_size);
    testComponentDiffBatchSizes(wrp);
  }
  //1 1
  {   
    int A_sz[2] = {4,3};
    int B_sz[2] = {5,3};
    int C_sz[2] = {4,5};

    std::cout << "Contract 1 1" << std::endl;
    Batch3tensorPairContractComponent<Config> cpt(1,1,nrm);
    Batch3tensorPairContractComponentWrapper<Config> wrp(cpt, A_sz,B_sz,C_sz);
    testComponentDeriv(wrp,batch_size);
    testComponentDiffBatchSizes(wrp);
  }
  std::cout << "testBatch3tensorPairContractComponent passed" << std::endl;
}




template<typename Config, int TensDim>
struct BatchTensorConcatenateComponentWrapper{
  EXTRACT_CONFIG_TYPES;
  
  BatchTensorConcatenateComponent<Config,TensDim> &cpt;
  std::vector< std::array<int,TensDim-1> > in_sz;
  std::vector<size_t> in_tens_lin_sz;
  size_t lin_sz;
  int N;
  int out_sz[TensDim-1];
  int concat_dim;
  
  BatchTensorConcatenateComponentWrapper(BatchTensorConcatenateComponent<Config,TensDim> &cpt, const std::vector< std::array<int,TensDim-1> > &in_sz, int concat_dim): cpt(cpt), in_sz(in_sz), N(in_sz.size()), concat_dim(concat_dim){
    for(int d=0;d<TensDim-1;d++)      
      out_sz[d] = in_sz[0][d];
    for(int t=1;t<N;t++)
      out_sz[concat_dim] += in_sz[t][concat_dim];

    lin_sz=1;
    for(int d=0;d<TensDim-1;d++) lin_sz *= out_sz[d];

    in_tens_lin_sz.resize(N);
    for(int n=0;n<N;n++){
      in_tens_lin_sz[n]=1;
      for(int d=0;d<TensDim-1;d++)
	in_tens_lin_sz[n] *= in_sz[n][d];
    }
  }

  void inputTensSize(int *into, int n, int batch_size){
    memcpy(into, in_sz[n].data(), (TensDim-1)*sizeof(int));
    into[TensDim-1] = batch_size;
  }
  void outputTensSize(int *into, int batch_size){
    memcpy(into, out_sz, (TensDim-1)*sizeof(int));
    into[TensDim-1] = batch_size;
  }
  std::vector< Tensor<FloatType,TensDim>* > constructInputTensors(int batch_size){
    std::vector< Tensor<FloatType,TensDim>* > tens(N);
    int tsize[TensDim];
    for(int i=0;i<N;i++){
      inputTensSize(tsize, i, batch_size);
      tens[i] = new Tensor<FloatType,TensDim>(tsize);
    }
    return tens;
  }
  
  size_t outputLinearSize() const{ return lin_sz; }
  size_t inputLinearSize() const{ return lin_sz; }
  
  Matrix<FloatType> value(const Matrix<FloatType> &in, EnableDeriv enable_deriv = DerivNo){
    std::vector< Tensor<FloatType,TensDim>* > tens = constructInputTensors(in.size(1));
    int off = 0;
    for(int i=0;i<N;i++){
      unflattenFromBatchVector(*tens[i], in, off);
      off += in_tens_lin_sz[i];
    }
    Tensor<FloatType,TensDim> out = cpt.value(tens.data());
    for(int i=0;i<N;i++) delete tens[i];
    return flattenToBatchVector(out);
  }
  void deriv(Vector<FloatType> &cost_deriv_params, int off, Matrix<FloatType> &&_above_deriv_lin, Matrix<FloatType> &cost_deriv_inputs){
    int batch_size = _above_deriv_lin.size(1);
    int out_sz_b[TensDim]; outputTensSize(out_sz_b, batch_size);
    Tensor<FloatType,TensDim> above_deriv = unflattenFromBatchVector<TensDim>(_above_deriv_lin, out_sz_b);

    std::vector< Tensor<FloatType,TensDim>* > tens = constructInputTensors(batch_size);
    cpt.deriv(std::move(above_deriv), tens.data());
    cost_deriv_inputs = Matrix<FloatType>(lin_sz,batch_size);
    int poff = 0;
    for(int i=0;i<N;i++){
      flattenToBatchVector(cost_deriv_inputs, *tens[i], poff);
      delete tens[i];
      poff += in_tens_lin_sz[i];
    }
  }
    
  void update(int off, const Vector<FloatType> &new_params){}
  void step(int off, const Vector<FloatType> &derivs, FloatType eps){}
  inline int nparams() const{ return cpt.nparams(); }
  void getParams(Vector<FloatType> &into, int off){}

  std::string inCoord(size_t i, int b, int batch_size) const{
    return std::to_string(i)+" "+std::to_string(b);
  }

    
    
};


void testBatchTensorConcatenateComponent(){
  typedef confDouble Config;
  typedef typename Config::FloatType FloatType;

  //4-tensor
  int batch_size = 5;
  
  { //contract dim 2
    std::vector< std::array<int,3> > in_sz({
	  {2,3,4},
	  {2,3,3},
	  {2,3,6} });
        
    BatchTensorConcatenateComponent<Config,4> cpt(2,  3);
    BatchTensorConcatenateComponentWrapper<Config,4> wrp(cpt, in_sz, 2);
    testComponentDeriv(wrp,batch_size);
    testComponentDiffBatchSizes(wrp);
  }

  { //contract dim 1
    std::vector< std::array<int,3> > in_sz({
	{2,4,3},
	{2,3,3},
	{2,6,3} });
        
    BatchTensorConcatenateComponent<Config,4> cpt(1,  3);
    BatchTensorConcatenateComponentWrapper<Config,4> wrp(cpt, in_sz, 1);
    testComponentDeriv(wrp,batch_size);
    testComponentDiffBatchSizes(wrp);
  }

  { //contract dim 0
    std::vector< std::array<int,3> > in_sz({
	{4,2,3},
	{3,2,3},
	{6,2,3} });
        
    BatchTensorConcatenateComponent<Config,4> cpt(0,  3);
    BatchTensorConcatenateComponentWrapper<Config,4> wrp(cpt, in_sz, 0);
    testComponentDeriv(wrp,batch_size);
    testComponentDiffBatchSizes(wrp);
  }


  //3-tensor
  
  { //contract dim 1
    std::vector< std::array<int,2> > in_sz({
	{2,4},
	{2,3},
	{2,6} });
        
    BatchTensorConcatenateComponent<Config,3> cpt(1,  3);
    BatchTensorConcatenateComponentWrapper<Config,3> wrp(cpt, in_sz, 1);
    testComponentDeriv(wrp,batch_size);
    testComponentDiffBatchSizes(wrp);
  }

  { //contract dim 0
    std::vector< std::array<int,2> > in_sz({
	{4,2},
	{3,2},
	{6,2} });
        
    BatchTensorConcatenateComponent<Config,3> cpt(0,  3);
    BatchTensorConcatenateComponentWrapper<Config,3> wrp(cpt, in_sz, 0);
    testComponentDeriv(wrp,batch_size);
    testComponentDiffBatchSizes(wrp);
  }
  
  std::cout << "testBatchTensorConcatenateComponent passed" << std::endl;
}


template<typename Config, int TensDim>
struct ScaleComponentWrapper{
  EXTRACT_CONFIG_TYPES;
  
  ScaleComponent<Config,TensDim> &cpt;
  int size[TensDim-1];
  size_t size_lin;

  ScaleComponentWrapper(ScaleComponent<Config,TensDim> &cpt, int const *sz): cpt(cpt){
    memcpy(size,sz,(TensDim-1)*sizeof(int));
    size_lin = 1;
    for(int i=0;i<TensDim-1;i++)
      size_lin *= sz[i];
  }

  size_t outputLinearSize() const{ return size_lin; }
  size_t inputLinearSize() const{ return size_lin; }
  
  Matrix<FloatType> value(const Matrix<FloatType> &in, EnableDeriv enable_deriv = DerivNo){
    int size_d[TensDim]; memcpy(size_d,size,(TensDim-1)*sizeof(int)); size_d[TensDim-1] = in.size(1);
    Tensor<FloatType,TensDim> T = unflattenFromBatchVector<TensDim>(in,size_d);
    return flattenToBatchVector(cpt.value(T,enable_deriv));
  }
  void deriv(Vector<FloatType> &cost_deriv_params, int off, Matrix<FloatType> &&_above_deriv_lin, Matrix<FloatType> &cost_deriv_inputs){
    int size_d[TensDim]; memcpy(size_d,size,(TensDim-1)*sizeof(int)); size_d[TensDim-1] = _above_deriv_lin.size(1);
    Tensor<FloatType,TensDim> above_deriv = unflattenFromBatchVector<TensDim>(_above_deriv_lin,size_d);
    Tensor<FloatType,TensDim> dcost_by_dIn;
    cpt.deriv(cost_deriv_params, off, std::move(above_deriv), dcost_by_dIn);
    cost_deriv_inputs = flattenToBatchVector(dcost_by_dIn);
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

  std::string inCoord(size_t i, int b, int batch_size) const{
    std::ostringstream ss;
    batchTensorSize(size_b, TensDim-1, size, batch_size);
    int coord[TensDim];
    tensorOffsetUnmap<TensDim>(coord, size_b, b+batch_size*i);
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
  int osize[3] = {size[0],size[1],size[2]};
  int batch_size = size[3];
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

	ScaleComponentWrapper<Config,4> wrp(cpta,osize);
	testComponentDeriv(wrp,batch_size,FloatType(1e-7));
	testComponentDiffBatchSizes(wrp);
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

	ScaleComponentWrapper<Config,4> wrp(cpta,osize);
	testComponentDeriv(wrp,batch_size,FloatType(1e-7));
	testComponentDiffBatchSizes(wrp);
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

	ScaleComponentWrapper<Config,4> wrp(cpta,osize);
	testComponentDeriv(wrp,batch_size,FloatType(1e-7));
	testComponentDiffBatchSizes(wrp);
      }
    }
    
  }
  
  std::cout << "testScaleComponent passed" << std::endl;
}


template<typename Config, int TensDim>
struct BatchTensorDimensionSliceComponentWrapper{
  EXTRACT_CONFIG_TYPES;
  
  BatchTensorDimensionSliceComponent<Config,TensDim> &cpt;
  int in_sz[TensDim-1];
  int out_sz[TensDim-2];
  size_t in_lin_sz;
  size_t out_lin_sz;

    
  BatchTensorDimensionSliceComponentWrapper(BatchTensorDimensionSliceComponent<Config,TensDim> &cpt, int const* _in_sz, int const* _out_sz): cpt(cpt){
    memcpy(in_sz,_in_sz,(TensDim-1)*sizeof(int));
    memcpy(out_sz,_out_sz,(TensDim-2)*sizeof(int));
    in_lin_sz = 1;
    for(int d=0;d<TensDim-1;d++)
      in_lin_sz *= in_sz[d];

    out_lin_sz = 1;
    for(int d=0;d<TensDim-2;d++)
      out_lin_sz *= out_sz[d];
  }
  
  size_t outputLinearSize() const{ return out_lin_sz; }
  size_t inputLinearSize() const{ return in_lin_sz; }
  
  Matrix<FloatType> value(const Matrix<FloatType> &in, EnableDeriv enable_deriv = DerivNo){
    batchTensorSize(in_sz_b, TensDim, in_sz, in.size(1));     
    Tensor<FloatType,TensDim> in_t = unflattenFromBatchVector<TensDim>(in, in_sz_b);
    return flattenToBatchVector(cpt.value(in_t));
  }
  void deriv(Vector<FloatType> &cost_deriv_params, int off, Matrix<FloatType> &&_above_deriv_lin, Matrix<FloatType> &cost_deriv_inputs){
    int batch_size = _above_deriv_lin.size(1);
    batchTensorSize(out_sz_b, TensDim-1, out_sz, batch_size); 
    batchTensorSize(in_sz_b, TensDim, in_sz, batch_size);     
    Tensor<FloatType,TensDim-1> above_deriv = unflattenFromBatchVector<TensDim-1>(_above_deriv_lin, out_sz_b);
    Tensor<FloatType,TensDim> tmp(in_sz_b);
    cpt.deriv(std::move(above_deriv), tmp);
    cost_deriv_inputs = flattenToBatchVector(tmp);
  }
    
  void update(int off, const Vector<FloatType> &new_params){}
  void step(int off, const Vector<FloatType> &derivs, FloatType eps){}
  inline int nparams() const{ return cpt.nparams(); }
  void getParams(Vector<FloatType> &into, int off){}

  std::string inCoord(size_t i, int b, int batch_size) const{
    return std::to_string(i)+","+std::to_string(b);
  }

};



void testBatchTensorDimensionSliceComponent(){
  typedef confDouble Config;
  typedef typename Config::FloatType FloatType;
  
  std::mt19937 rng(1234);

  int size[4] = {2,3,4,5};
  int batch_size = size[3];
  
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
      testComponentDeriv(wrp,batch_size);
      testComponentDiffBatchSizes(wrp);
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
      testComponentDeriv(wrp,batch_size);
      testComponentDiffBatchSizes(wrp);
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
      testComponentDeriv(wrp,batch_size);
      testComponentDiffBatchSizes(wrp);
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
  int in_size[Dim-1];
  size_t in_size_lin;
  int out_size[Dim-1];
  size_t out_size_lin;
  

  MatrixTensorContractComponentWrapper(MatrixTensorContractComponent<Config,Dim> &cpt, int const *in_sz, int const *out_sz): cpt(cpt){
    memcpy(in_size,in_sz,(Dim-1)*sizeof(int));
    memcpy(out_size,out_sz,(Dim-1)*sizeof(int));
    in_size_lin = out_size_lin = 1;
    for(int d=0;d<Dim-1;d++){
      in_size_lin *= in_sz[d];
      out_size_lin *= out_sz[d];
    }
  }

  size_t outputLinearSize() const{ return out_size_lin; }
  size_t inputLinearSize() const{ return in_size_lin; }
  
  Matrix<FloatType> value(const Matrix<FloatType> &in, EnableDeriv enable_deriv = DerivNo){
    batchTensorSize(in_size_b, Dim, in_size, in.size(1));
    Tensor<FloatType,Dim> A = unflattenFromBatchVector<Dim>(in, in_size_b);
    return flattenToBatchVector(cpt.value(A,enable_deriv));
  }
  void deriv(Vector<FloatType> &cost_deriv_params, int off, Matrix<FloatType> &&_above_deriv_lin, Matrix<FloatType> &cost_deriv_inputs){
    batchTensorSize(out_size_b, Dim, out_size, _above_deriv_lin.size(1));
    Tensor<FloatType,Dim> above_deriv = unflattenFromBatchVector<Dim>(_above_deriv_lin, out_size_b);
    Tensor<FloatType,Dim> dcost_by_dIn;
    cpt.deriv(cost_deriv_params, off , std::move(above_deriv), dcost_by_dIn);
    cost_deriv_inputs = flattenToBatchVector(dcost_by_dIn);
  }
    
  void update(int off, const Vector<FloatType> &new_params){  cpt.update(off,new_params); }
  void step(int off, const Vector<FloatType> &derivs, FloatType eps){ cpt.step(off,derivs,eps); }
  inline int nparams() const{ return cpt.nparams(); }
  void getParams(Vector<FloatType> &into, int off){ cpt.getParams(into,off); }

  std::string inCoord(size_t i, int b, int batch_size) const{
    batchTensorSize(in_size_b, Dim, in_size, batch_size);
    std::ostringstream ss;
    int coord[Dim];
    tensorOffsetUnmap<Dim>(coord, in_size_b, b+i*batch_size);
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
  int batch_size = tens_sizes[3];

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
  testComponentDeriv(wrp,batch_size);
  testComponentDiffBatchSizes(wrp);
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

