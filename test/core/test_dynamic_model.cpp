#include <HPCortex.hpp>
#include <Testing.hpp>

void testDynamicModel(){
  typedef confDouble Config;
  typedef typename Config::FloatType FloatType;

  std::mt19937 rng(1234);
  
  {
  
    FloatType B=0;
    FloatType A=2;
    Matrix<FloatType> winit(1,1,A);
    Vector<FloatType> binit(1,B);

    auto model = dnn_layer(winit,binit,
			   dnn_layer(winit, binit,
				     dnn_layer(winit,binit,
					       input_layer<Config>()
					       )
				     )
			   );
 
    auto composed = enwrap( input_layer<Config>() );
    for(int i=0;i<3;i++)
      composed = enwrap( dnn_layer(winit,binit, std::move(composed) ) );
 
  
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
    uniformRandom(filter,rng);
    NoPadding<FloatType> padding;
    int conv_out_len = padding.layerOutputLength(input_size,kernel_size,stride);

    auto conv_layer_w = enwrap( conv1d_layer(filter, ReLU<FloatType>(), padding, stride,
					     input_layer<Config,Tens3 >()
					     )
				);
    auto flatten_layer_w = enwrap(flatten_layer(conv_layer_w));
    Matrix<FloatType> weights(2,conv_out_len);
    Vector<FloatType> bias(2);
    uniformRandom(weights,rng);
    uniformRandom(bias,rng);
    auto dnn_layer_w = enwrap( dnn_layer( weights, bias, flatten_layer_w ) );

    int batch_size = 3;
    Tens3 input(1,input_size,batch_size);
    uniformRandom(input,rng);
    
    Matrix<FloatType> got = dnn_layer_w.value(input);
    assert(got.size(0) == 2 && got.size(1) == batch_size);
    
    
    auto full_model = dnn_layer(weights, bias,
				flatten_layer(
					      conv1d_layer(filter, ReLU<FloatType>(), padding, stride,
							   input_layer<Config,Tens3>()							   
							   )
					      )
				);
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
