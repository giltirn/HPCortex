#include <HPCortex.hpp>
#include <Testing.hpp>

void testFlattenLayer(){
  typedef double FloatType; //more precise derivatives
  FloatType delta = 1e-6;

  int dims[3] = {2,3,4};
  typedef std::vector<FloatType> vecD;
  vecD in_lin(2*3*4);
  for(int x=0;x<2;x++)
    for(int y=0;y<3;y++)
      for(int z=0;z<4;z++)
	in_lin[ z + 4*(y + 3*x) ] = 0.1 + 0.01 * (z + 4*(y + 3*x));

  typedef Tensor<FloatType,3> TensType;
  Matrix<FloatType> expect(2*3, 4, in_lin); //will use z as batch_size  
  TensType in(dims, in_lin);

  auto m = flatten_layer( input_layer<FloatType, TensType>() );
  
  Matrix<FloatType> got = m.value(in);
  std::cout << got << std::endl;
  
  assert( abs_near(got, expect, 1e-12, true) );

  //To test the deriv we can just pass in the flattened matrix and should get the tensor back
  //i.e.     dcost/dout_{(i,j), k} = tensflat_{(i,j),k}
  //         dcost/din_{i'j'k'} = dcost/dout_{(i,j), k} dout_{(i,j), k}/din_{i'j'k'}=  tensflat_{(i',j'),k'} = tens_{i'j'k'}
  Vector<FloatType> cost_deriv(0);
  TensType deriv_got;
  m.deriv(cost_deriv,0,std::move(expect), &deriv_got);

  doHost2(in,deriv_got,{
  for(int x=0;x<2;x++)
    for(int y=0;y<3;y++)
      for(int z=0;z<4;z++)
	assert( abs_near(in_v(x,y,z), deriv_got_v(x,y,z), 1e-9 ) );
    });

  std::cout << "Tests passed" << std::endl;
}

int main(int argc, char** argv){
  initialize(argc,argv);
  testFlattenLayer();
  return 0;
}
