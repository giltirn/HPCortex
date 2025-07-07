#include <HPCortex.hpp>
#include <Testing.hpp>

//This test aims to ensure we are correctly treating the derivatives of the activation function

//A non-linear activation function
template<typename FloatType>
class TestActivation{
public: 
  void operator()(Matrix<FloatType> &x, Matrix<FloatType> *deriv = nullptr) const{
    int dim = x.size(0);
    int batch_size = x.size(1);
  
    //f(x)_i = 0.75*x_i^2
    if(deriv == nullptr){
      autoView(x_v,x,DeviceReadWrite);

      accelerator_for2d(b,batch_size,i,dim,1,{
	  x_v(i,b) = 0.75 * x_v(i,b) * x_v(i,b);
	});
    }else{
      *deriv = Matrix<FloatType>(dim,batch_size);

      autoView(deriv_v, (*deriv), DeviceWrite);
      autoView(x_v,x,DeviceReadWrite);
      accelerator_for2d(b,batch_size,i,dim,1,{
	  deriv_v(i,b) = 0.75 * 2 * x_v(i,b);
	  x_v(i,b) = 0.75 * x_v(i,b) * x_v(i,b);
	});
    }
  }
};

template<typename ActivationFunc>
struct testActivationFunc{};

template<typename FloatType>
struct testActivationFunc<TestActivation<FloatType> >{
  static void doit(Vector<FloatType> &y1pred){
    doHost(y1pred, {
	for(int i=0;i<3;i++)
	  y1pred_v(i) = 0.75 * pow(y1pred_v(i),2);
      });
  }

};

template<typename FloatType>
struct testActivationFunc<ReLU<FloatType> >{
  static void doit(Vector<FloatType> &y1pred){
    doHost(y1pred, {
	for(int i=0;i<3;i++)
	  y1pred_v(i) = y1pred_v(i) <= 0. ? 0. : y1pred_v(i);
      });
  }

};


template<typename FloatType>
struct testActivationFunc<noActivation<FloatType> >{
  static void doit(Vector<FloatType> &y1pred){ }

};


template<template<typename> class ActivationFunc>
void testActivation(){
  typedef double FloatType; //more precise derivatives
  FloatType delta = 1e-6;
  typedef std::vector<FloatType> vecD;
 
  Matrix<FloatType> w1_init(3,2, vecD({0.1,0.2,
   	                             -0.1,-0.2,
			              0.7,0.7}));
  Vector<FloatType> b1_init( vecD({-0.5,0.7,-0.9}));		    
 
  auto f = mse_cost( dnn_layer(w1_init, b1_init, ActivationFunc<FloatType>(), input_layer<FloatType>()) );

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
    
    testActivationFunc< ActivationFunc<FloatType> >::doit(y1pred);
      
    Vector<FloatType> y1_b = peekColumn(y1,i);
    std::cout << y1pred << " " << y1_b << std::endl;

    doHost2(y1pred,y1_b,{
    expect += pow(y1pred_v(0)-y1_b_v(0),2)/3. + pow(y1pred_v(1)-y1_b_v(1),2)/3. + pow(y1pred_v(2)-y1_b_v(2),2)/3.;
      });
  }
  expect /= 2.;
    
  FloatType got=  f.loss(x1,y1);
  std::cout << "Test loss : got " << got << " expect " << expect << std::endl;
  assert(near(got,expect,FloatType(1e-6)));

  //test the derivatives
  Vector<FloatType> dexpect(9);
  {
    autoView(dexpect_v,dexpect,HostWrite); 
    int p=0;
    for(int i=0;i<3;i++){
      for(int j=0;j<2;j++){
	Matrix<FloatType> w1_p = w1_init;
	doHost(w1_p, { w1_p_v(i,j) += delta; });
	auto f2 = mse_cost( dnn_layer(w1_p, b1_init, ActivationFunc<FloatType>(), input_layer<FloatType>()) );
	dexpect_v(p++) = (f2.loss(x1,y1) - got)/delta;
      }
    }
    for(int i=0;i<3;i++){
      Vector<FloatType> b1_p = b1_init;
      doHost(b1_p, { b1_p_v(i) += delta; });      
      auto f2 = mse_cost( dnn_layer(w1_init, b1_p, ActivationFunc<FloatType>(), input_layer<FloatType>()) );
      dexpect_v(p++) = (f2.loss(x1,y1) - got)/delta;    
    }
  }
  
  Vector<FloatType> dgot = f.deriv();
  doHost2(dgot,dexpect,{
    for(int i=0;i<9;i++){
      std::cout << "Test deriv wrt param " << i <<  ": got " << dgot_v(i) << " expect " << dexpect_v(i) << std::endl;
    }
    });
  assert(near(dgot,dexpect,FloatType(1e-3),true));
  std::cout << "testActivation passed" << std::endl;
}

int main(int argc, char** argv){
  initialize(argc,argv);
  
  testActivation<TestActivation>();
  testActivation<ReLU>();
  testActivation<noActivation>();

  return 0;
}
