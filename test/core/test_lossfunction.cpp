#include <HPCortex.hpp>
#include <Testing.hpp>

void testMSEcost(){
  typedef confDouble Config;
  typedef typename Config::FloatType FloatType;
  typedef std::vector<FloatType> vecD;
  
  Matrix<FloatType> w1_init(3,2, vecD({0.1,0.2,
   	                             -0.1,-0.2,
			              0.7,0.7}));
  Vector<FloatType> b1_init( vecD({0.5,0.7,0.9}));		    

  doHost(w1_init, {  assert(w1_init_v(0,0) == FloatType(0.1) && w1_init_v(1,0) == FloatType(-0.1) ); });
  {
    auto c = peekColumn(w1_init,0);
    doHost(c, { assert(c_v(0) == FloatType(0.1) && c_v(1) == FloatType(-0.1) && c_v(2) == FloatType(0.7) ); });
  }
  
  auto f = mse_cost( dnn_layer(w1_init, b1_init, input_layer<Config>()) );

  //NB batch size 2, batches in different *columns*
  Matrix<FloatType> x1(2,2,vecD({1.3, 0.6,
	             -0.3, -1.7}));
  
  Matrix<FloatType> y1(3,2, vecD({-0.5, -0.5,
        	                   1.7, 1.3,
			          -0.7, -0.5}));
  //test the MSE loss calculation
  FloatType expect = 0.;
  for(int i=0;i<2;i++){  
    Vector<FloatType> y1pred = w1_init * peekColumn(x1,i) + b1_init;
    Vector<FloatType> y1_b = peekColumn(y1,i);
    std::cout << y1pred << " " << y1_b << std::endl;

    doHost2(y1pred,y1_b,{
    expect += pow(y1pred_v(0)-y1_b_v(0),2)/3. + pow(y1pred_v(1)-y1_b_v(1),2)/3. + pow(y1pred_v(2)-y1_b_v(2),2)/3.;
      });
  }
  expect /= 2.;
    
  FloatType got=  f.loss(x1,y1);
  std::cout << "Test loss : got " << got << " expect " << expect << std::endl;
  assert(abs_near(got,expect,FloatType(1e-8)));

  //test predict
  {
    Vector<FloatType> x_test(vecD({-0.3,0.3}));
    Vector<FloatType> y_test_expect = w1_init * x_test + b1_init;
    Vector<FloatType> y_test_got = f.predict(x_test, 2);
    assert(abs_near(y_test_expect,y_test_got,FloatType(1e-8)));
  }
  
  //test the derivatives
  Vector<FloatType> dexpect(9);
  FloatType delta =  1e-8;
  {
    autoView(dexpect_v,dexpect,HostWrite); 
    int p=0;
    for(int i=0;i<3;i++){
      for(int j=0;j<2;j++){
	Matrix<FloatType> w1_p = w1_init;
	doHost(w1_p, { w1_p_v(i,j) += delta; });
	auto f2 = mse_cost( dnn_layer(w1_p, b1_init, input_layer<Config>()) );
	dexpect_v(p++) = (f2.loss(x1,y1) - got)/delta;
      }
    }
    for(int i=0;i<3;i++){
      Vector<FloatType> b1_p = b1_init;
      doHost(b1_p, { b1_p_v(i) += delta; });      
      auto f2 = mse_cost( dnn_layer(w1_init, b1_p, input_layer<Config>()) );
      dexpect_v(p++) = (f2.loss(x1,y1) - got)/delta;    
    }
  }
  f.loss(x1,y1);
  Vector<FloatType> dgot = f.deriv();
  doHost2(dgot,dexpect,{
    for(int i=0;i<9;i++){
      std::cout << "Test deriv wrt param " << i <<  ": got " << dgot_v(i) << " expect " << dexpect_v(i) << std::endl;
    }
    });
  assert(abs_near(dgot,dexpect,FloatType(1e-5),true));
  
  //test update
  Matrix<FloatType> w1_new(3,2,    vecD({-0.5,0.4,
					  0.8,1.2,
					  2.1,-3.0}));
  Vector<FloatType> b1_new( vecD({-0.5,0.7,-1.1}));	

  auto ftest = mse_cost( dnn_layer(w1_new, b1_new, input_layer<Config>()) );
  f.update(ftest.getParams());

  FloatType expect_l = ftest.loss(x1,y1);
  FloatType got_l =  f.loss(x1,y1);
  std::cout << "Update check : expect " << expect_l  << " got " << got_l << std::endl;
  assert(abs_near(got_l,expect_l,FloatType(1e-6)));
  
  std::cout << "testMSEcost passed" << std::endl;
}


int main(int argc, char** argv){
  initialize(argc,argv);
  
  testMSEcost();
  return 0;
}
