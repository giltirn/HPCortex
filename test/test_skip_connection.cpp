#include <HPCortex.hpp>
#include <Testing.hpp>

void testSkipConnection(){
  typedef double FloatType; //more precise derivatives
  FloatType delta = 1e-6;
  typedef std::vector<FloatType> vecD;
 
  Matrix<FloatType> w1_init(3,3, vecD({0.1,0.2,-0.1,
                	              -0.1,-0.2,0.2,
	                               0.7,0.7,-0.3}));
  Vector<FloatType> b1_init( vecD({-0.5,0.7,-0.9}));		    


  Matrix<FloatType> w2_init(3,3, vecD({0.4,-0.2,-0.3,
	                              -0.05,0.1,0.3,
	                               0.25,-0.6,-0.7}));
  Vector<FloatType> b2_init( vecD({0.3,-0.35,0.75}));		    

  
  {
    //Test value for one layer  
    auto skip1 = skip_connection( dnn_layer(input_layer<FloatType>(), w1_init, b1_init),  input_layer<FloatType>() );

    auto internal1 = dnn_layer(input_layer<FloatType>(), w1_init, b1_init);

    Matrix<FloatType> x1(3,2,vecD({1.3, 0.6,
	  -0.3, -1.7,
	  0.7, 0.35}));
    Matrix<FloatType> expect_val = x1 + internal1.value(x1);
    Matrix<FloatType> got_val = skip1.value(x1);

    assert(near(expect_val,got_val,1e-5,true));
  }
  {
    //Test value for two layer
    auto skip1 = skip_connection( dnn_layer(input_layer<FloatType>(), w1_init, b1_init),  input_layer<FloatType>() );
    auto skip2 = skip_connection( dnn_layer(input_layer<FloatType>(), w2_init, b2_init),  skip1 );
    
    auto internal1 = dnn_layer(input_layer<FloatType>(), w1_init, b1_init);
    auto internal2 = dnn_layer(input_layer<FloatType>(), w2_init, b2_init);

    Matrix<FloatType> x1(3,2,vecD({1.3, 0.6,
	  -0.3, -1.7,
	  0.7, 0.35}));
    Matrix<FloatType> l1 = internal1.value(x1) + x1;
    Matrix<FloatType> expect_val = internal2.value(l1) + l1;
    
    Matrix<FloatType> got_val = skip2.value(x1);
	
    assert(near(expect_val,got_val,1e-5,true));

    //Test getparams
    int nparams = skip2.nparams();
    std::cout << "nparams got " << nparams << " expect " << 9+3+9+3 << std::endl;
    assert(nparams == 9+3+9+3);
    
    Vector<FloatType> params_got(nparams);
    skip2.getParams(params_got,0);
    Vector<FloatType> params_expect(nparams);
    {
      autoView(params_expect_v,params_expect,HostWrite);
      autoView(w2_init_v,w2_init,HostRead);
      autoView(b2_init_v,b2_init,HostRead);
      autoView(w1_init_v,w1_init,HostRead);
      autoView(b1_init_v,b1_init,HostRead);
      
      int p =0;
      for(int i=0;i<3;i++)
	for(int j=0;j<3;j++)
	  params_expect_v(p++) = w2_init_v(i,j);
      for(int j=0;j<3;j++)
	params_expect_v(p++) = b2_init_v(j);

      for(int i=0;i<3;i++)
	for(int j=0;j<3;j++)
	  params_expect_v(p++) = w1_init_v(i,j);
      for(int j=0;j<3;j++)
	params_expect_v(p++) = b1_init_v(j);
    }
      
    assert(near(params_got,params_expect,1e-7,true));

    //Test update
    Vector<FloatType> params_new(params_got);
    doHost(params_new, { for(int i=0;i<nparams;i++) params_new_v(i) *= 1.05; });
    skip2.update(0, params_new);

    skip2.getParams(params_got, 0);
    assert(near(params_got,params_new,1e-7,true));

    //Test step
    Vector<FloatType> dir(nparams,0.05);
    skip2.step(0,dir,2.15);

    skip2.getParams(params_got, 0);
    params_expect = params_new;
    doHost(params_expect, { for(int i=0;i<nparams;i++) params_expect_v(i) -= 2.15 * 0.05; });

    assert(near(params_got,params_expect,1e-7,true));

    //Test derivs
    Vector<FloatType> params_base = params_new;
    skip2.update(0, params_base);

    
    Matrix<FloatType> y1(3,2,vecD({0.4, 3.1,
	                           2.8, -3.1,
	                           1.2, -2.1}));
    auto cost = mse_cost(skip2);
    FloatType base_cost = cost.loss(x1,y1);

    Vector<FloatType> dgot = cost.deriv();
    Vector<FloatType> dexpect(nparams);
    for(int p=0;p<nparams;p++){
      Vector<FloatType> params_shift(params_base);
      doHost(params_shift, { params_shift_v(p) += delta; });
      skip2.update(0, params_shift);
      doHost(dexpect, { dexpect_v(p) = (cost.loss(x1,y1) - base_cost)/delta; });
    }
    std::cout << "Derivatives got " << dgot << " expect " << dexpect << std::endl;    
    
    assert(near(dgot,dexpect,FloatType(1e-3),true));
  }
}

void testSkipConnectionTensor(){
  typedef double FloatType; 
  FloatType delta = 1e-6;
  typedef std::vector<FloatType> vecD;
  std::mt19937 rng(1234);
  
  typedef Tensor<FloatType,3> TensorType;

  int tens_size[3] = {3,4,5};
  
  Matrix<FloatType> w1_init(4,4);
  Vector<FloatType> b1_init(4);
  uniformRandom(w1_init,rng);
  uniformRandom(b1_init,rng);
  
  auto skip_over = batch_tensor_dnn_layer<3>(input_layer<FloatType,TensorType>(), w1_init, b1_init, 1, ReLU<FloatType>());
  auto skip = skip_connection( skip_over, input_layer<FloatType,TensorType>() );

  TensorType x(tens_size);
  uniformRandom(x,rng);

  TensorType got = skip.value(x);
  TensorType expect = x + skip_over.value(x);
  assert(abs_near(got,expect,1e-5,true));

  testDeriv(skip, tens_size,tens_size);
}

void testSkipConnectionTensorSplitJoin(){
  typedef double FloatType; 
  FloatType delta = 1e-6;
  typedef std::vector<FloatType> vecD;
  std::mt19937 rng(1234);
  
  typedef Tensor<FloatType,3> TensorType;

  int tens_size[3] = {3,4,5};
  
  Matrix<FloatType> w1_init(4,4);
  Vector<FloatType> b1_init(4);
  uniformRandom(w1_init,rng);
  uniformRandom(b1_init,rng);

  auto repl = replicate_layer(input_layer<FloatType,TensorType>(),2);
  auto chn = batch_tensor_dnn_layer<3>(*repl[0], w1_init, b1_init, 1, ReLU<FloatType>());
  auto join = sum_join_layer(chn,*repl[1]);
  
  auto skip_over = batch_tensor_dnn_layer<3>(input_layer<FloatType,TensorType>(), w1_init, b1_init, 1, ReLU<FloatType>());
  
  TensorType x(tens_size);
  uniformRandom(x,rng);

  TensorType got = join.value(x);
  TensorType expect = x + skip_over.value(x);
  assert(abs_near(got,expect,1e-5,true));

  testDeriv(join, tens_size,tens_size);
}


int main(int argc, char** argv){
  initialize(argc,argv);
  testSkipConnection();
  testSkipConnectionTensor();
  testSkipConnectionTensorSplitJoin();  
  return 0;
}
