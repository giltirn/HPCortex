#include <HPCortex.hpp>
#include <Testing.hpp>

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
    cpt.deriv(std::move(above_deriv), dcost_by_dIn);
    cost_deriv_inputs = flatten(dcost_by_dIn);
  }
    
  void update(int off, const Vector<FloatType> &new_params){}
  void step(int off, const Vector<FloatType> &derivs, FloatType eps){}
  inline int nparams() const{ return cpt.nparams(); }
  void getParams(Vector<FloatType> &into, int off){}

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
std::vector<FloatType> norm_lin(const std::vector<FloatType> &v, FloatType eps){
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
    vv = (vv - mean)/std;
  return out;
}

void testNormComponent(){
  typedef double FloatType;
  std::mt19937 rng(1234);

  int size[4] = {2,3,4,5};
  
  Tensor<FloatType,4> v(size);
  uniformRandom(v,rng);
  FloatType eps = 0.55;
 
  {
    //dim 0
    NormComponent<FloatType,4> cpt(0,eps);
    Tensor<FloatType,4> got = cpt.value(v);
    Tensor<FloatType,4> expect(size);

    doHost2(v,expect, {
	for(int j=0;j<size[1];j++)
	  for(int k=0;k<size[2];k++)
	    for(int b=0;b<size[3];b++){
	      std::vector<FloatType> l_in(size[0]);
	      for(int i=0;i<size[0];i++)
		l_in[i] = v_v(i,j,k,b);
	      std::vector<FloatType> l_out = norm_lin(l_in, eps);
	      for(int i=0;i<size[0];i++)
		expect_v(i,j,k,b) = l_out[i];
	    }
      });
    assert(abs_near(got,expect,FloatType(1e-5),true));

    NormComponent<FloatType,4> cpta(0,eps);
    NormComponentWrapper<FloatType,4> wrp(cpta,size);
    testComponentDeriv(wrp, FloatType(1e-7));
  }

  {
    //dim 1
    NormComponent<FloatType,4> cpt(1,eps);
    Tensor<FloatType,4> got = cpt.value(v);
    Tensor<FloatType,4> expect(size);

    doHost2(v,expect, {
	for(int i=0;i<size[0];i++)
	  for(int k=0;k<size[2];k++)
	    for(int b=0;b<size[3];b++){
	      std::vector<FloatType> l_in(size[1]);
	      for(int j=0;j<size[1];j++)
		l_in[j] = v_v(i,j,k,b);
	      std::vector<FloatType> l_out = norm_lin(l_in, eps);
	      for(int j=0;j<size[1];j++)
		expect_v(i,j,k,b) = l_out[j];
	    }
      });
    assert(abs_near(got,expect,FloatType(1e-5),true));

    NormComponent<FloatType,4> cpta(1,eps);
    NormComponentWrapper<FloatType,4> wrp(cpta,size);
    testComponentDeriv(wrp, FloatType(1e-7));
  }

  {
    //dim 2
    NormComponent<FloatType,4> cpt(2,eps);
    Tensor<FloatType,4> got = cpt.value(v);
    Tensor<FloatType,4> expect(size);

    doHost2(v,expect, {
	for(int i=0;i<size[0];i++)
	  for(int j=0;j<size[1];j++)
	    for(int b=0;b<size[3];b++){
	      std::vector<FloatType> l_in(size[2]);
	      for(int k=0;k<size[2];k++)
		l_in[k] = v_v(i,j,k,b);
	      std::vector<FloatType> l_out = norm_lin(l_in, eps);
	      for(int k=0;k<size[2];k++)
		expect_v(i,j,k,b) = l_out[k];
	    }
      });
    assert(abs_near(got,expect,FloatType(1e-5),true));

    NormComponent<FloatType,4> cpta(2,eps);
    NormComponentWrapper<FloatType,4> wrp(cpta,size);
    testComponentDeriv(wrp, FloatType(1e-7));
  }
  
  std::cout << "testNormComponent passed" << std::endl;
}

template<typename FloatType>
std::vector<FloatType> norm_and_scale_lin(const std::vector<FloatType> &v, FloatType eps, const Vector<FloatType> &gamma, const Vector<FloatType> &beta){
  size_t len = v.size();
  FloatType mean = 0., var = 0.;
  for(auto vv : v){
    mean += vv;
    var += vv*vv;
  }
  mean /= len;
  var = var/len - mean*mean;
  FloatType std = sqrt(var + eps);

  std::vector<FloatType> out(v.size());
  doHost2(gamma,beta, {
      for(int i=0;i<out.size();i++)
	out[i] = gamma_v(i) * ( v[i] - mean )/ std + beta_v(i);      
    });
  return out;
}

void testNormLayer(){
  typedef double FloatType;
  std::mt19937 rng(1234);

  int size[4] = {2,3,4,5};
  
  Tensor<FloatType,4> v(size);
  uniformRandom(v,rng);
  FloatType eps = 0.55;
  
  {
    //dim 0
    Vector<FloatType> gamma(size[0]), beta(size[0]);
    uniformRandom(gamma,rng); uniformRandom(beta,rng);

    auto m = norm_layer<4>(0, size[0], true, true, gamma, beta, eps,
			   input_layer<FloatType,Tensor<FloatType,4> >()
			   );

    Tensor<FloatType,4> got = m.value(v);
    Tensor<FloatType,4> expect(size);

    doHost2(v,expect, {
	for(int j=0;j<size[1];j++)
	  for(int k=0;k<size[2];k++)
	    for(int b=0;b<size[3];b++){
	      std::vector<FloatType> l_in(size[0]);
	      for(int i=0;i<size[0];i++)
		l_in[i] = v_v(i,j,k,b);
	      std::vector<FloatType> l_out = norm_and_scale_lin(l_in, eps, gamma, beta);
	      for(int i=0;i<size[0];i++)
		expect_v(i,j,k,b) = l_out[i];
	    }
      });
    assert(abs_near(got,expect,FloatType(1e-5),true));

    for(int use_gamma=0; use_gamma<2; use_gamma++){
      for(int use_beta=0; use_beta<2; use_beta++){
	std::cout << "use_gamma: " << use_gamma << " use_beta: " << use_beta << std::endl;
	auto mm = norm_layer<4>(0, size[0], bool(use_gamma), bool(use_beta), gamma, beta, eps,
				input_layer<FloatType,Tensor<FloatType,4> >()
				);
	testDeriv(mm,size,size, FloatType(1e-7));
      }
    }
    
  }

  {
    //dim 1
    Vector<FloatType> gamma(size[1]), beta(size[1]);
    uniformRandom(gamma,rng); uniformRandom(beta,rng);

    auto m = norm_layer<4>(1, size[1], true, true, gamma, beta, eps,
			   input_layer<FloatType,Tensor<FloatType,4> >()
			   );

    Tensor<FloatType,4> got = m.value(v);
    Tensor<FloatType,4> expect(size);

    doHost2(v,expect, {
	for(int i=0;i<size[0];i++)
	  for(int k=0;k<size[2];k++)
	    for(int b=0;b<size[3];b++){
	      std::vector<FloatType> l_in(size[1]);
	      for(int j=0;j<size[1];j++)
		l_in[j] = v_v(i,j,k,b);
	      std::vector<FloatType> l_out = norm_and_scale_lin(l_in, eps, gamma, beta);
	      for(int j=0;j<size[1];j++)
		expect_v(i,j,k,b) = l_out[j];
	    }
      });
    assert(abs_near(got,expect,FloatType(1e-5),true));

    for(int use_gamma=0; use_gamma<2; use_gamma++){
      for(int use_beta=0; use_beta<2; use_beta++){
	std::cout << "use_gamma: " << use_gamma << " use_beta: " << use_beta << std::endl;
	auto mm = norm_layer<4>(1, size[1], bool(use_gamma), bool(use_beta), gamma, beta, eps,
				input_layer<FloatType,Tensor<FloatType,4> >()
				);
	testDeriv(mm,size,size, FloatType(1e-7));
      }
    }
    
  }

  {
    //dim 2
    Vector<FloatType> gamma(size[2]), beta(size[2]);
    uniformRandom(gamma,rng); uniformRandom(beta,rng);

    auto m = norm_layer<4>(2, size[2], true, true, gamma, beta, eps,
			   input_layer<FloatType,Tensor<FloatType,4> >()
			   );
    
    Tensor<FloatType,4> got = m.value(v);
    Tensor<FloatType,4> expect(size);

    doHost2(v,expect, {
	for(int i=0;i<size[0];i++)
	  for(int j=0;j<size[1];j++)
	    for(int b=0;b<size[3];b++){
	      std::vector<FloatType> l_in(size[2]);
	      for(int k=0;k<size[2];k++)
		l_in[k] = v_v(i,j,k,b);
	      std::vector<FloatType> l_out = norm_and_scale_lin(l_in, eps, gamma, beta);
	      for(int k=0;k<size[2];k++)
		expect_v(i,j,k,b) = l_out[k];
	    }
      });
    assert(abs_near(got,expect,FloatType(1e-5),true));

    for(int use_gamma=0; use_gamma<2; use_gamma++){
      for(int use_beta=0; use_beta<2; use_beta++){
	std::cout << "use_gamma: " << use_gamma << " use_beta: " << use_beta << std::endl;
	auto mm = norm_layer<4>(2, size[2], bool(use_gamma), bool(use_beta), gamma, beta, eps,
				input_layer<FloatType,Tensor<FloatType,4> >()
				);
	testDeriv(mm,size,size, FloatType(1e-7));
      }
    }
    
  }
  
  std::cout << "testNormLayer passed" << std::endl;
}



int main(int argc, char** argv){
  initialize(argc,argv);
  testNormComponent();
  testNormLayer();
  return 0;
}
