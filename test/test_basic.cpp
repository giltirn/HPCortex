#include <Layers.hpp>
#include <LossFunctions.hpp>
#include <Optimizers.hpp>  
#include <Comms.hpp>

void basicTests(){
  typedef std::vector<double> vecD;
  
  Matrix w1_init(3,2, vecD({0.1,0.2,
	                   -0.1,-0.2,
			    0.7,0.7}));
  Vector b1_init( vecD({0.5,0.7,0.9}));		    
  
  auto f = mse_cost( dnn_layer(input_layer(), w1_init, b1_init) );

  //NB batch size 2, batches in different *columns*
  Matrix x1(2,2,vecD({1.3, 0.6,
	             -0.3, -1.7}));
  
  Matrix y1(3,2,std::vector<double>({-0.5, -0.5,
	                             1.7, 1.3
				     -0.7, -0.5}));

  double expect = 0.;
  for(int i=0;i<2;i++){  
    Vector y1pred = w1_init * x1.peekColumn(i) + b1_init;
    Vector y1_b = y1.peekColumn(i);
    expect += pow(y1pred(0)-y1_b(0),2)/3. + pow(y1pred(1)-y1_b(1),2)/3. + pow(y1pred(2)-y1_b(2),2)/3.;
  }
  expect /= 2.;
    
  double got=  f.loss(x1,y1);
  std::cout << "Test loss : got " << got << " expect " << expect << std::endl;


  Vector dexpect(9);
  int p=0;
  for(int i=0;i<3;i++){
    for(int j=0;j<2;j++){
      Matrix w1_p = w1_init;
      w1_p(i,j) += 1e-7;
      auto f2 = mse_cost( dnn_layer(input_layer(), w1_p, b1_init) );
      dexpect(p++) = (f2.loss(x1,y1) - got)/1e-7;
    }
  }
  for(int i=0;i<3;i++){
    Vector b1_p = b1_init;
    b1_p(i) += 1e-7;
    auto f2 = mse_cost( dnn_layer(input_layer(), w1_init, b1_p) );
    dexpect(p++) = (f2.loss(x1,y1) - got)/1e-7;    
  }

  Vector dgot = f.deriv();
  for(int i=0;i<9;i++){
    std::cout << "Test deriv wrt param " << i <<  ": got " << dgot(i) << " expect " << dexpect(i) << std::endl;
  }
    
  //test update
  Matrix w1_new(3,2, std::vector<double>({-0.5,0.4,
					  0.8,1.2,
					  2.1,-3.0}));
  Vector b1_new( std::vector<double>({-0.5,0.7,-1.1}));	

  auto ftest = mse_cost( dnn_layer(input_layer(), w1_new, b1_new) );
  f.update(ftest.getParams());
  
  std::cout << "Update check : expect " << ftest.loss(x1,y1) << " got " <<  f.loss(x1,y1) << std::endl;

}

int main(int argc, char** argv){
  initialize(argc,argv);
  
  basicTests();

  return 0;
}
