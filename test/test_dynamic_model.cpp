#include <HPCortex.hpp>
#include <Testing.hpp>

void testDynamicModel(){
  typedef float FloatType;
  
  FloatType B=0;
  FloatType A=2;
  Matrix<FloatType> winit(1,1,A);
  Vector<FloatType> binit(1,B);

  auto model = dnn_layer(
			 dnn_layer(
				   dnn_layer(
      				             input_layer<FloatType>(),
					     winit,binit),
				   winit, binit),
			 winit,binit);
 
  auto composed = enwrap( input_layer<FloatType>() );
  for(int i=0;i<3;i++)
     composed = enwrap( dnn_layer(std::move(composed), winit,binit) );
 
  
  int iters=10;
  for(int i=0;i<iters;i++){
    Matrix<FloatType> x(1,1, i+1);
    Matrix<FloatType> vexpect = model.value(x);
    Matrix<FloatType> vgot = composed.value(x);
    doHost2(vexpect,vgot,{    
	std::cout << i << " got " << vgot_v(0,0) << " expect " << vexpect_v(0,0) << std::endl;
	assert(near(vgot_v(0,0),vexpect_v(0,0),FloatType(1e-8)));
      });
  }
}

int main(int argc, char** argv){
  initialize(argc, argv);
  
  testDynamicModel();
  
  return 0;
}
