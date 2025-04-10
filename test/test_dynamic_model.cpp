#include <HPCortex.hpp>
#include <Testing.hpp>

void testDynamicModel(){
  typedef double FloatType;
  std::mt19937 rng(1234);
  
  {
  
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
  {
    int kernel_size = 3;
    int stride = 1;
    int input_size = 10;

    typedef Tensor<FloatType,3> Tens3;
    Tens3 filter(1,1,kernel_size);
    random(filter,rng);
    NoPadding<FloatType> padding;
    int conv_out_len = padding.layerOutputLength(input_size,kernel_size,stride);

    auto conv_layer_w = enwrap( conv1d_layer(input_layer<FloatType,Tens3 >(),
					     filter, ReLU<FloatType>(), padding, stride) );
    auto flatten_layer_w = enwrap(flatten_layer(conv_layer_w));
    Matrix<FloatType> weights(2,conv_out_len);
    Vector<FloatType> bias(2);
    random(weights,rng);
    random(bias,rng);
    auto dnn_layer_w = enwrap( dnn_layer( flatten_layer_w, weights, bias ) );

    int batch_size = 3;
    Tens3 input(1,input_size,batch_size);
    random(input,rng);
    
    Matrix<FloatType> got = dnn_layer_w.value(input);
    assert(got.size(0) == 2 && got.size(1) == batch_size);
    
    
    auto full_model = dnn_layer(
				flatten_layer(
					      conv1d_layer(
							   input_layer<FloatType,Tens3>(),
							   filter, ReLU<FloatType>(), padding, stride
							   )
					      )
				, weights, bias);
    Matrix<FloatType> expect = full_model.value(input);
    assert( abs_near(got,expect,1e-8,true) );
  }
  std::cout << "Tests passed" << std::endl;
}

int main(int argc, char** argv){
  initialize(argc, argv);
  
  testDynamicModel();
  
  return 0;
}
