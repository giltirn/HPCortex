#include <HPCortex.hpp>
#include <Testing.hpp>

void testSoftMax(){
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

int main(int argc, char** argv){
  initialize(argc,argv);
  
  testSoftMax();

  return 0;
}
