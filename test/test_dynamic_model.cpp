#include<DynamicModel.hpp>
#include <Comms.hpp>

void testDynamicModel(){
  double B=0;
  double A=2;
  Matrix winit(1,1,A);
  Vector binit(1,B);

  auto model = dnn_layer(
			 dnn_layer(
				   dnn_layer(
      				             input_layer(),
					     winit,binit),
				   winit, binit),
			 winit,binit);
 
  LayerWrapper composed = enwrap( input_layer() );
  for(int i=0;i<3;i++)
     composed = enwrap( dnn_layer(std::move(composed), winit,binit) );
 
  
  int iters=10;
  for(int i=0;i<iters;i++){
      Matrix x(1,1, i+1);
      Matrix vexpect = model.value(x);

      Matrix vgot = composed.value(x);

      std::cout << i << " got " << vgot(0,0) << " expect " << vexpect(0,0) << std::endl;
  }
}

int main(int argc, char** argv){
  initialize(argc, argv);
  
  testDynamicModel();
  
  return 0;
}
