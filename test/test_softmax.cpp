#include <HPCortex.hpp>
#include <Testing.hpp>

template<typename _FloatType, int Dim>
struct SoftMaxComponentWrapper{
  typedef _FloatType FloatType;
  
  SoftMaxComponent<FloatType,Dim> &cpt;
  int size[Dim];
  size_t size_lin;

  SoftMaxComponentWrapper(SoftMaxComponent<FloatType,Dim> &cpt, int const *sz): cpt(cpt){
    memcpy(size,sz,Dim*sizeof(int));
    size_lin = 1;
    for(int d=0;d<Dim;d++) size_lin *= sz[d];
  }

  size_t outputLinearSize() const{ return size_lin; }
  size_t inputLinearSize() const{ return size_lin; }
  
  Vector<FloatType> value(const Vector<FloatType> &in){
    Tensor<FloatType,Dim> A(size);
    unflatten(A,in);
    Tensor<FloatType,Dim> C = cpt.value(A);
    return flatten(C);
  }
  void deriv(Vector<FloatType> &cost_deriv_params, int off, Vector<FloatType> &&_above_deriv_lin, Vector<FloatType> &cost_deriv_inputs){
    Vector<FloatType> above_deriv_lin = std::move(_above_deriv_lin);
    Tensor<FloatType,Dim> above_deriv(size);
    unflatten(above_deriv,above_deriv_lin);
    Tensor<FloatType,Dim> dcost_by_dIn;
    cpt.deriv(std::move(above_deriv), dcost_by_dIn);
    cost_deriv_inputs = flatten(dcost_by_dIn);
  }
    
  void update(int off, const Vector<FloatType> &new_params){}
  void step(int off, const Vector<FloatType> &derivs, FloatType eps){}
  inline int nparams() const{ return cpt.nparams(); }
  void getParams(Vector<FloatType> &into, int off){}

  std::string inCoord(size_t i) const{
    std::ostringstream ss;
    int coord[Dim];
    tensorOffsetUnmap<Dim>(coord, size, i);
    ss << "(";
    for(int d=0;d<Dim;d++)
      ss << coord[d] << (d==Dim-1? ")" : ", ");
    return ss.str();
  }       
};

void testSoftMaxComponent(){
  typedef double FloatType;
  FloatType delta = 1e-4;
  std::mt19937 rng(1234);
   
  typedef std::vector<FloatType> vecD;

  FloatType beta = 0.3;

  {
    int size[2] = {4,5};
    Matrix<FloatType> logp(size);
    random(logp,rng);

    SoftMaxComponent<FloatType,2> cpt(0, beta);

    Matrix<FloatType> vgot = cpt.value(logp);
    Matrix<FloatType> vexpect(size);

    doHost3(vexpect, vgot, logp, {
	std::vector<FloatType> logp_pencil(size[0]);
	for(int b=0;b<size[1];b++){
	  for(int i=0;i<size[0];i++)
	    logp_pencil[i] = logp_v(i,b);
	  std::vector<FloatType> expect_pencil = softMaxVector(logp_pencil, beta);
	  for(int i=0;i<size[0];i++){
	    vexpect_v(i,b) = expect_pencil[i];
	    std::cout << i << " " << b << " got " << vgot_v(i,b) << " expect " << vexpect_v(i,b) << std::endl;
	  }
	}
      });
    
    assert(abs_near(vgot,vexpect,FloatType(1e-4),true));
  }
  {
    int size[3] = {3,4,5};
    Tensor<FloatType,3> logp(size);
    random(logp,rng);

    {//dim 0
      SoftMaxComponent<FloatType,3> cpt(0, beta);

      Tensor<FloatType,3> vgot = cpt.value(logp);
      Tensor<FloatType,3> vexpect(size);

      doHost3(vexpect, vgot, logp, {
	  std::vector<FloatType> logp_pencil(size[0]);
	  for(int j=0;j<size[1];j++){
	    for(int b=0;b<size[2];b++){
	      
	      for(int i=0;i<size[0];i++)
		logp_pencil[i] = logp_v(i,j,b);
	      std::vector<FloatType> expect_pencil = softMaxVector(logp_pencil, beta);
	      for(int i=0;i<size[0];i++){
		vexpect_v(i,j,b) = expect_pencil[i];
		std::cout << i << " " << j << " " << b << " got " << vgot_v(i,j,b) << " expect " << vexpect_v(i,j,b) << std::endl;
	      }
	      
	    }
	  }
	});
    
      assert(abs_near(vgot,vexpect,FloatType(1e-4),true));
    }
    {//dim 1
      SoftMaxComponent<FloatType,3> cpt(1, beta);

      Tensor<FloatType,3> vgot = cpt.value(logp);
      Tensor<FloatType,3> vexpect(size);

      doHost3(vexpect, vgot, logp, {
	  std::vector<FloatType> logp_pencil(size[1]);
	  for(int i=0;i<size[0];i++){
	    for(int b=0;b<size[2];b++){
	      
	      for(int j=0;j<size[1];j++)
		logp_pencil[j] = logp_v(i,j,b);
	      std::vector<FloatType> expect_pencil = softMaxVector(logp_pencil, beta);
	      for(int j=0;j<size[1];j++){
		vexpect_v(i,j,b) = expect_pencil[j];
		std::cout << i << " " << j << " " << b << " got " << vgot_v(i,j,b) << " expect " << vexpect_v(i,j,b) << std::endl;
	      }
	      
	    }
	  }
	});
    
      assert(abs_near(vgot,vexpect,FloatType(1e-4),true));
    }

      
  }

  //Check derivatives
  for(int d=0;d<3;d++){
    std::cout << "Testing derivs for softmax on dim " << d << std::endl;
    int size[4] = {2,3,4,5};
    SoftMaxComponent<FloatType,4> cpt(d, beta);
    SoftMaxComponentWrapper<FloatType,4> wrp(cpt,size);
    testComponentDeriv(wrp);
  }
    
}


void testSoftMaxLayer(){
  typedef float FloatType;
  FloatType delta = 1e-4;
  std::mt19937 rng(1234);
   
  typedef std::vector<FloatType> vecD;

  FloatType beta = 0.3;
  auto m = softmax_layer(input_layer<FloatType>(), beta);

  int np = 20;
  int batch_size = 5;
  
  Matrix<FloatType> logp(np,batch_size);
  random(logp,rng);

  ///value
  Matrix<FloatType> vgot = m.value(logp);

  Matrix<FloatType> vexpect(np,batch_size);
  doHost3(vexpect, vgot, logp, {
      for(int b=0;b<batch_size;b++){
	FloatType max = logp_v(0,b);
	for(int i=1;i<np;i++)
	  max = std::max(max, logp_v(i,b));

	FloatType norm = exp(beta*(logp_v(0,b)-max));
	for(int i=1;i<np;i++)
	  norm += exp(beta*(logp_v(i,b)-max));
	
	for(int i=0;i<np;i++){
	  vexpect_v(i,b) = exp(beta*(logp_v(i,b)-max)) / norm;
	  std::cout << i << " " << b << " got " << vgot_v(i,b) << " expect " << vexpect_v(i,b) << std::endl;
	}
      }
    });
    
  assert(abs_near(vgot,vexpect,FloatType(1e-4),true));

  //deriv
  //it has no parameters so we only need to test the input derivatives
  Matrix<FloatType> in_deriv; //dcost/din_j
  Vector<FloatType> cost_deriv_dummy;

  //let  cost = \sum_i c_i * out_i
  //above_deriv = dcost/dout_i = c_i
  Vector<FloatType> c(np);
  random(c,rng);
    
  Matrix<FloatType> above_deriv(np,batch_size); //dcost/dout_i
  doHost2(above_deriv, c, {
      for(int i=0;i<np;i++)
	for(int b=0;b<batch_size;b++)
	  above_deriv_v(i,b) = c_v(i);
    });
  m.deriv(cost_deriv_dummy, 0, std::move(above_deriv), &in_deriv);

  Matrix<FloatType> expect_deriv(np,batch_size, 0.);
  
  for(int j=0;j<np;j++){
    Matrix<FloatType> logp_dj = logp;
    doHost(logp_dj, {
	for(int b=0;b<batch_size;b++)
	  logp_dj_v(j,b) += delta;
      });
    Matrix<FloatType> vp = m.value(logp_dj); //out_i( ... in_j+delta ... )
    Matrix<FloatType> dout_dinj = (FloatType(1.)/delta) * (vp - vgot);
    doHost3(expect_deriv,dout_dinj,c,
	    {
	      for(int i=0;i<np;i++)
		for(int b=0;b<batch_size;b++)
		  expect_deriv_v(j,b) += c_v(i)*dout_dinj_v(i,b);
	    });
  }
  assert(abs_near(in_deriv, expect_deriv, FloatType(1e-3), true));
  
  std::cout << "Tests passed" << std::endl;
}


template<typename _FloatType>
struct BatchedMatrixRowSoftMaxComponentWrapper{
  typedef _FloatType FloatType;
  
  BatchedMatrixRowSoftMaxComponent<FloatType> &cpt;
  int size[3];
  size_t size_lin;

  BatchedMatrixRowSoftMaxComponentWrapper(BatchedMatrixRowSoftMaxComponent<FloatType> &cpt, int const *sz): cpt(cpt){
    memcpy(size,sz,3*sizeof(int));
    size_lin = 1;
    for(int d=0;d<3;d++) size_lin *= sz[d];
  }

  size_t outputLinearSize() const{ return size_lin; }
  size_t inputLinearSize() const{ return size_lin; }
  
  Vector<FloatType> value(const Vector<FloatType> &in){
    Tensor<FloatType,3> A(size);
    unflatten(A,in);
    Tensor<FloatType,3> C = cpt.value(A);
    return flatten(C);
  }
  void deriv(Vector<FloatType> &cost_deriv_params, int off, Vector<FloatType> &&_above_deriv_lin, Vector<FloatType> &cost_deriv_inputs){
    Vector<FloatType> above_deriv_lin = std::move(_above_deriv_lin);
    Tensor<FloatType,3> above_deriv(size);
    unflatten(above_deriv,above_deriv_lin);
    Tensor<FloatType,3> dcost_by_dIn;
    cpt.deriv(std::move(above_deriv), dcost_by_dIn);
    cost_deriv_inputs = flatten(dcost_by_dIn);
  }
    
  void update(int off, const Vector<FloatType> &new_params){}
  void step(int off, const Vector<FloatType> &derivs, FloatType eps){}
  inline int nparams() const{ return cpt.nparams(); }
  void getParams(Vector<FloatType> &into, int off){}

  std::string inCoord(size_t i) const{
    std::ostringstream ss;
    int coord[3];
    tensorOffsetUnmap<3>(coord, size, i);
    ss << "(";
    for(int d=0;d<3;d++)
      ss << coord[d] << (d==3-1? ")" : ", ");
    return ss.str();
  }       
};

void testBatchedMatrixRowSoftMaxComponent(){
  typedef double FloatType;
  FloatType delta = 1e-4;
  std::mt19937 rng(1234);
   
  typedef std::vector<FloatType> vecD;

  FloatType beta = 0.3;

  int size[3] = {4,4,5};
  Tensor<FloatType,3> logp(size);
  random(logp,rng);
  
  for(int use_mask = 0; use_mask < 2; use_mask++){
    std::cout << "Testing " << (use_mask ? "WITH" : "WITHOUT") << " mask" << std::endl;
    
    BatchedMatrixRowSoftMaxComponent<FloatType> cpt((bool)use_mask, beta);
    SoftMaxComponent<FloatType,3> compare(1, beta); //tested above

    Tensor<FloatType,3> vgot = cpt.value(logp);

    Tensor<FloatType,3> logp_cmp(logp);
    if(use_mask){
      assert(size[0]==size[1]);

      autoView(logp_cmp_v,logp_cmp,HostReadWrite);
      for(int b=0;b<size[2];b++)
	for(int r=0;r<size[1];r++)
	  for(int c=r+1;c<size[1];c++) //softmax(In + M)  where M is *strictly* upper triangular (diagonal elements also zero) with nonzero elements =-inf
	    logp_cmp_v(r,c,b) = -1000000;
    }  
	
    Tensor<FloatType,3> vexpect = compare.value(logp_cmp);
    
    assert(abs_near(vgot,vexpect,FloatType(1e-4),true));
 
    BatchedMatrixRowSoftMaxComponentWrapper<FloatType> wrp(cpt,size);
    testComponentDeriv(wrp);
  }
}



int main(int argc, char** argv){
  initialize(argc,argv);
  testSoftMaxComponent();
  testSoftMaxLayer();
  testBatchedMatrixRowSoftMaxComponent();
  return 0;
}
