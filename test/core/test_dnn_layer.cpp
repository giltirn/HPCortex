#include <HPCortex.hpp>
#include <Testing.hpp>
typedef confDouble Config;
typedef typename Config::FloatType FloatType;

template<typename ActivationFunc>
Tensor<FloatType,4> expectContract(int contract_dim, const Matrix<FloatType> &weights, const Vector<FloatType> &bias, const Tensor<FloatType,4> &in, bool use_bias, ActivationFunc &activation){
  int out_sizes[4];
  memcpy(out_sizes,in.sizeArray(),4*sizeof(int));
  out_sizes[contract_dim] = bias.size(0);
  Tensor<FloatType,4> out(out_sizes,0.);
  
  autoView(weights_v,weights,HostRead);
  autoView(bias_v,bias,HostRead);
  autoView(in_v,in,HostRead);
  autoView(out_v,out,HostReadWrite);

  if(contract_dim == 0){
  
    for(int j=0;j<in.size(1);j++)
      for(int k=0;k<in.size(2);k++)
	for(int l=0;l<in.size(3);l++)
	  for(int i=0;i<bias.size(0);i++){

	    for(int ii=0;ii<in.size(0);ii++)
	      out_v(i,j,k,l) += weights_v(i,ii) * in_v(ii,j,k,l);
	    out_v(i,j,k,l) += use_bias ? bias_v(i): 0.;
	  }

  }else if(contract_dim == 1){
  
    for(int i=0;i<in.size(0);i++)
      for(int k=0;k<in.size(2);k++)
	for(int l=0;l<in.size(3);l++)
	  for(int j=0;j<bias.size(0);j++){

	    for(int jj=0;jj<in.size(1);jj++)
	      out_v(i,j,k,l) += weights_v(j,jj) * in_v(i,jj,k,l);
	    out_v(i,j,k,l) += use_bias ? bias_v(j): 0.;
	  }

  }else if(contract_dim == 2){
  
    for(int i=0;i<in.size(0);i++)
      for(int j=0;j<in.size(1);j++)
	for(int l=0;l<in.size(3);l++)
	  for(int k=0;k<bias.size(0);k++){

	    for(int kk=0;kk<in.size(2);kk++)
	      out_v(i,j,k,l) += weights_v(k,kk) * in_v(i,j,kk,l);
	    out_v(i,j,k,l) += use_bias ? bias_v(k): 0.;
	  }

  }else assert(0);
  
  activation(out);
  
  return out;
}


template<typename Config, int Dim, typename ActivationFunc>
struct BatchTensorDNNcomponentWrapper{
  EXTRACT_CONFIG_TYPES;
  
  BatchTensorDNNcomponent<Config,Dim, ActivationFunc> &cpt;
  int in_size[Dim];
  size_t in_size_lin;
  int out_size[Dim];
  size_t out_size_lin;
  

  BatchTensorDNNcomponentWrapper(BatchTensorDNNcomponent<Config,Dim, ActivationFunc> &cpt, int const *in_sz, int const *out_sz): cpt(cpt){
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
  
  Vector<FloatType> value(const Vector<FloatType> &in, EnableDeriv enable_deriv = DerivNo){
    Tensor<FloatType,Dim> A(in_size);
    unflatten(A,in);
    Tensor<FloatType,Dim> C = cpt.value(A,enable_deriv);
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

void testBatchTensorDNNcomponentAndLayer(){
  std::mt19937 rng(1234);
    
  int tens_sizes[4] = {2,3,4,5};
  int out_size = 6;

  Vector<FloatType> bias(out_size);
  uniformRandom(bias,rng);

  //typedef noActivation<FloatType> ActivationFunc;
  typedef ReLU<FloatType> ActivationFunc;
  ActivationFunc activation;

  Tensor<FloatType,4> x(tens_sizes);
  uniformRandom(x,rng);
 
  for(int contract_dim=0;contract_dim < 3;contract_dim++){
    std::cout << "Contract dim " << contract_dim << std::endl;
    Matrix<FloatType> weights(out_size,tens_sizes[contract_dim]);
    uniformRandom(weights,rng);

    for(int use_bias = 0; use_bias < 1; use_bias++){
      std::cout << (use_bias ? "WITH" : "WITHOUT") << " bias" << std::endl;
      
      std::unique_ptr< BatchTensorDNNcomponent<Config,4, ActivationFunc>  > cpt;
      if(use_bias)
	cpt.reset(new BatchTensorDNNcomponent<Config,4, ActivationFunc>(weights,bias,contract_dim,activation));
      else
	cpt.reset(new BatchTensorDNNcomponent<Config,4, ActivationFunc >(weights,contract_dim,activation));

      int nparam_expect =  out_size*tens_sizes[contract_dim] + (use_bias ? out_size : 0);
      std::cout << "Nparam " << cpt->nparams() << " expect " << nparam_expect << std::endl;
      assert(cpt->nparams() == nparam_expect);
      
      Tensor<FloatType,4> got = cpt->value(x);
      Tensor<FloatType,4> expect = expectContract(contract_dim, weights,bias,x,use_bias,activation);
      assert(abs_near(got,expect, 1e-6, true));

      BatchTensorDNNcomponentWrapper<Config,4, ActivationFunc> wrp(*cpt, x.sizeArray(),expect.sizeArray());
      std::cout << "Test component deriv" << std::endl;
      testComponentDeriv(wrp);

      std::cout << "Test layer deriv" << std::endl;
      if(use_bias){
	auto m = batch_tensor_dnn_layer<4>(weights, bias, contract_dim, activation, input_layer<Config,Tensor<FloatType,4> >());
	Tensor<FloatType,4> got2 = m.value(x);
	assert(abs_near(got2,expect, 1e-6, true));
	testDeriv(m, x.sizeArray(), expect.sizeArray(), 1e-8);
      }else{
	auto m = batch_tensor_dnn_layer<4>(weights, contract_dim, activation, input_layer<Config,Tensor<FloatType,4> >());
	Tensor<FloatType,4> got2 = m.value(x);
	assert(abs_near(got2,expect, 1e-6, true));
	testDeriv(m, x.sizeArray(), expect.sizeArray(), 1e-8);
      }
      
    }
  }


  std::cout << "testBatchTensorDNNcomponentAndLayer passed" << std::endl;
}


template<typename ActivationFunc>
Matrix<FloatType> expectContract2D(const Matrix<FloatType> &weights, const Vector<FloatType> &bias, const Matrix<FloatType> &in, bool use_bias, ActivationFunc &activation){
  int out_sizes[2]; 
  memcpy(out_sizes,in.sizeArray(),2*sizeof(int));
  out_sizes[0] = bias.size(0);
  Matrix<FloatType> out(out_sizes,0.);
  
  autoView(weights_v,weights,HostRead);
  autoView(bias_v,bias,HostRead);
  autoView(in_v,in,HostRead);
  autoView(out_v,out,HostReadWrite);

  for(int i=0;i<out_sizes[0];i++){
    for(int b=0;b<in.size(1);b++){
      for(int ii=0;ii<in.size(0);ii++)
	out_v(i,b) += weights_v(i,ii) * in_v(ii,b);
      out_v(i,b) += use_bias ? bias_v(i): 0.;
    }
  }
  
  activation(out);
  
  return out;
}


void testBatchTensorDNNcomponentAndLayer2D(){
  std::mt19937 rng(1234);
    
  int tens_sizes[2] = {4,5};
  int out_size = 6;

  Vector<FloatType> bias(out_size);
  uniformRandom(bias,rng);

  typedef ReLU<FloatType> ActivationFunc;
  ActivationFunc activation;

  Matrix<FloatType> x(tens_sizes);
  uniformRandom(x,rng);

  int contract_dim = 0;
  
  Matrix<FloatType> weights(out_size,tens_sizes[contract_dim]);
  uniformRandom(weights,rng);

  for(int use_bias = 0; use_bias < 1; use_bias++){
    std::cout << (use_bias ? "WITH" : "WITHOUT") << " bias" << std::endl;
    
    std::unique_ptr< BatchTensorDNNcomponent<Config,2, ActivationFunc>  > cpt;
    if(use_bias)
      cpt.reset(new BatchTensorDNNcomponent<Config,2, ActivationFunc>(weights,bias,contract_dim,activation));
    else
      cpt.reset(new BatchTensorDNNcomponent<Config,2, ActivationFunc >(weights,contract_dim,activation));

    int nparam_expect =  out_size*tens_sizes[0] + (use_bias ? out_size : 0);
    std::cout << "Nparam " << cpt->nparams() << " expect " << nparam_expect << std::endl;
    assert(cpt->nparams() == nparam_expect);
      
    Matrix<FloatType> got = cpt->value(x);
    Matrix<FloatType> expect = expectContract2D(weights,bias,x,use_bias,activation);
    assert(abs_near(got,expect, 1e-6, true));

    BatchTensorDNNcomponentWrapper<Config,2, ActivationFunc> wrp(*cpt, x.sizeArray(),expect.sizeArray());
    std::cout << "Test component deriv" << std::endl;
    testComponentDeriv(wrp);

    std::cout << "Test layer deriv" << std::endl;
    if(use_bias){
      auto m = batch_tensor_dnn_layer<2>(weights, bias, contract_dim, activation, input_layer<Config>());
      Matrix<FloatType> got2 = m.value(x);
      assert(abs_near(got2,expect, 1e-6, true));
      testDeriv(m, x.sizeArray(), expect.sizeArray(), 1e-8);
    }else{
      auto m = batch_tensor_dnn_layer<2>(weights, contract_dim, activation, input_layer<Config>());
      Matrix<FloatType> got2 = m.value(x);
      assert(abs_near(got2,expect, 1e-6, true));
      testDeriv(m, x.sizeArray(), expect.sizeArray(), 1e-8);
    }
  }      

  std::cout << "testBatchTensorDNNcomponentAndLayer2D passed" << std::endl;
}




int main(int argc, char** argv){
  initialize(argc,argv);
  testBatchTensorDNNcomponentAndLayer();
  testBatchTensorDNNcomponentAndLayer2D();

  return 0;
}


