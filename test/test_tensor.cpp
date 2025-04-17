#include <HPCortex.hpp>
#include <Testing.hpp>

void testTensor(){
  typedef double FloatType; 
  typedef std::vector<FloatType> vecD;
  std::mt19937 rng(1234);
  //Test some basic functionality
  {
    int dims[3] = {2,3,4};
    size_t size = 2*3*4;
    assert( tensorSize<3>(dims) == size );
            
    vecD input(size);
    vecD expect(size);
    for(int i=0;i<2;i++)
      for(int j=0;j<3;j++)
	for(int k=0;k<4;k++){
	  size_t off = k + 4*(j + 3*i);
	  int coord[3] = {i,j,k};
	  
	  assert(tensorOffset<3>(coord,dims) == off);

	  int test_coord[3];
	  tensorOffsetUnmap<3>(test_coord,dims,off);
	  for(int ii=0;ii<3;ii++) assert(test_coord[i] == coord[i]);	  
	  
	  input[off] = 3.141 * off;

	  expect[off] = input[off] + 0.15*off*off;
	}
    
    //Try a host-side kernel
    Tensor<FloatType,3> tens_host(dims, input);
    for(int d=0;d<3;d++)
      assert(tens_host.size(d) == dims[d]);
    
    {
      autoView(tens_host_v,tens_host,HostReadWrite);
    
      thread_for3d(i,2,j,3,k,4,{
	  int coord[3] = {(int)i,(int)j,(int)k};
	  size_t off = tensorOffset<3>(coord,dims);
	  tens_host_v(coord) = tens_host_v(coord) + 0.15 * off * off;
	});
    }
    {  
      autoView(tens_host_v,tens_host,HostRead);
      for(int i=0;i<2;i++)
	for(int j=0;j<3;j++)
	  for(int k=0;k<4;k++){
	    size_t off = k + 4*(j + 3*i);
	    int coord[3] = {(int)i,(int)j,(int)k};
	    assert(abs_near(tens_host_v(coord), expect[off],1e-8 ));
	  }
    }

    //Try a device-side kernel
    Tensor<FloatType,3> tens_device(dims, input);
    {
      autoView(tens_device_v,tens_device,DeviceReadWrite);
    
      accelerator_for3d(i,2,j,3,k,4,  1,{
	  int coord[3] = {(int)i,(int)j,(int)k};
	  size_t off = tensorOffset<3>(coord,dims);
	  tens_device_v(coord) = tens_device_v(coord) + 0.15 * off * off;
	});
    }
    {  
      autoView(tens_device_v,tens_device,HostRead);
      for(int i=0;i<2;i++)
	for(int j=0;j<3;j++)
	  for(int k=0;k<4;k++){
	    size_t off = k + 4*(j + 3*i);
	    int coord[3] = {(int)i,(int)j,(int)k};
	    assert(abs_near(tens_device_v(coord), expect[off], 1e-8) );
	  }
    }    
    
  }

  //Test fixed dim accessors
  {
    {
      vecD init({1.1,2.2,3.3});
      int sz = 3;
      Tensor<FloatType,1> tens_1(&sz, init);      
      doHost(tens_1, { assert(tens_1_v(0) == 1.1 && tens_1_v(1) == 2.2 && tens_1_v(2) == 3.3); });
    }
    {
      vecD init({
	  1.1,2.2,3.3,
	  4.4,5.5,6.6
	});
      int sz[2] = {2,3};
      Tensor<FloatType,2> tens_2(sz, init);
      doHost(tens_2, {
	  for(int i=0;i<2;i++)
	    for(int j=0;j<3;j++)
	      assert(tens_2_v(i,j) == init[j+3*i]);
	});
    }

    {
      int sz[3] = {2,3,4};
      vecD init(2*3*4);
      for(int i=0;i<2;i++)
	for(int j=0;j<3;j++)
	  for(int k=0;k<4;k++)
	    init[k+4*(j+3*i)] = (k+4*(j+3*i))*1.1;

      Tensor<FloatType,3> tens_3(sz, init);
      doHost(tens_3, {
	  for(int i=0;i<2;i++)
	    for(int j=0;j<3;j++)
	      for(int k=0;k<4;k++)
		assert(tens_3_v(i,j,k) == init[k+4*(j+3*i)]);
	});
    }

    {
      int sz[4] = {2,3,4,5};
      vecD init(2*3*4*5);
      for(int i=0;i<2;i++)
	for(int j=0;j<3;j++)
	  for(int k=0;k<4;k++)
	    for(int l=0;l>5;l++)
	      init[l+5*(k+4*(j+3*i))] = (l+5*(k+4*(j+3*i)))*1.1;

      Tensor<FloatType,4> tens_4(sz, init);
      doHost(tens_4, {
	  for(int i=0;i<2;i++)
	    for(int j=0;j<3;j++)
	      for(int k=0;k<4;k++)
		for(int l=0;l<5;l++)
		  assert(tens_4_v(i,j,k,l) == init[l+5*(k+4*(j+3*i))]);
	});
    }

    { //test pokeLastDimension
      int sz[3] = {2,3,4};
      Tensor<FloatType,3> orig(sz);
      random(orig,rng);

      int szp[2] = {2,3};
      Tensor<FloatType,2> topoke(szp);
      Tensor<FloatType,2> topoke2(szp);
      random(topoke,rng);
      random(topoke2,rng);

      Tensor<FloatType,3> result(orig);
      result.pokeLastDimension(topoke,0);
      doHost2(result, topoke, {
	  for(int i=0;i<2;i++)
	    for(int j=0;j<3;j++)
		assert(result_v(i,j,0) == topoke_v(i,j));
	});
      result.pokeLastDimension(topoke2,2);
      
      doHost3(result, topoke, topoke2, {
	  for(int i=0;i<2;i++)
	    for(int j=0;j<3;j++){
		assert(result_v(i,j,0) == topoke_v(i,j));
		assert(result_v(i,j,2) == topoke2_v(i,j));
	      }		
	});
    }

    

  }

  
  std::cout << "Test passed" << std::endl;
}

int main(int argc, char** argv){
  initialize(argc,argv);
  
  testTensor();

  return 0;
}
