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

    { //test peek/pokeLastDimension
      int sz[3] = {2,3,4};
      Tensor<FloatType,3> orig(sz);
      uniformRandom(orig,rng);

      int szp[2] = {2,3};
      Tensor<FloatType,2> topoke(szp);
      Tensor<FloatType,2> topoke2(szp);
      uniformRandom(topoke,rng);
      uniformRandom(topoke2,rng);

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

      Tensor<FloatType,2> r0 = result.peekLastDimension(0);
      Tensor<FloatType,2> r2 = result.peekLastDimension(2);
      assert(equal(r0,topoke));
      assert(equal(r2,topoke2));
    }

  }

  
  std::cout << "testTensor passed" << std::endl;
}


void testDimensionIteration(){
  typedef double FloatType;
  std::mt19937 rng(1234);
  {
    int size[2] = {3,4};
    Tensor<FloatType,2> tens(size);
    uniformRandom(tens,rng);

    size_t stride0 = tensorDimensionStride<2>(0,size);
    size_t stride1 = tensorDimensionStride<2>(1,size);
        
    doHost(tens, {
	//iter_dim=0
	for(int d1=0;d1<4;d1++){
	  size_t base = tensorDimensionBase<2>(0, &d1, size);
	  for(int d0=0;d0<3;d0++)      
	    assert( tens_v.data()[base + d0*stride0] == tens_v(d0,d1) );
	}

	//iter_dim=1
	for(int d0=0;d0<3;d0++){
	  size_t base = tensorDimensionBase<2>(1, &d0, size);
	  for(int d1=0;d1<4;d1++)      
	    assert( tens_v.data()[base + d1*stride1] == tens_v(d0,d1) );
	}
      });
  }

  {
    int size[3] = {3,4,5};
    Tensor<FloatType,3> tens(size);
    uniformRandom(tens,rng);
    
    size_t stride0 = tensorDimensionStride<3>(0,size);
    size_t stride1 = tensorDimensionStride<3>(1,size);
    size_t stride2 = tensorDimensionStride<3>(2,size);

    doHost(tens, {
	//iter_dim=0
	int other_coord[2];
	for(int d1=0;d1<size[1];d1++){
	  for(int d2=0;d2<size[2];d2++){
	    other_coord[0] = d1;
	    other_coord[1] = d2;
	    size_t base = tensorDimensionBase<3>(0, other_coord, size);
	    for(int d0=0;d0<size[0];d0++)      
	      assert( tens_v.data()[base + d0*stride0] == tens_v(d0,d1,d2) );
	  }
	}
	//iter_dim=1
	for(int d0=0;d0<size[0];d0++){
	  for(int d2=0;d2<size[2];d2++){
	    other_coord[0] = d0;
	    other_coord[1] = d2;
	    size_t base = tensorDimensionBase<3>(1, other_coord, size);
	    for(int d1=0;d1<size[1];d1++)      
	      assert( tens_v.data()[base + d1*stride1] == tens_v(d0,d1,d2) );
	  }
	}
	//iter_dim=2
	for(int d0=0;d0<size[0];d0++){
	  for(int d1=0;d1<size[1];d1++){
	    other_coord[0] = d0;
	    other_coord[1] = d1;
	    size_t base = tensorDimensionBase<3>(2, other_coord, size);
	    for(int d2=0;d2<size[2];d2++)      
	      assert( tens_v.data()[base + d2*stride2] == tens_v(d0,d1,d2) );
	  }
	}
      });
  }


  {
    int size[4] = {2,3,4,5};
    Tensor<FloatType,4> tens(size);
    uniformRandom(tens,rng);
    
    size_t stride0 = tensorDimensionStride<4>(0,size);
    size_t stride1 = tensorDimensionStride<4>(1,size);
    size_t stride2 = tensorDimensionStride<4>(2,size);
    size_t stride3 = tensorDimensionStride<4>(3,size);
    
    doHost(tens, {
	//iter_dim=0
	for(int d1=0;d1<size[1];d1++){
	  for(int d2=0;d2<size[2];d2++){
	    for(int b=0; b<size[3]; b++){
	      size_t o = d2 + size[2]*d1;
	      size_t base = batchTensorDimensionBaseLin<4>(0, b, o, size);

	      for(int d0=0;d0<size[0];d0++)      
		assert( tens_v.data()[base + d0*stride0] == tens_v(d0,d1,d2,b) );
	    }
	  }
	}

	//iter_dim=1
	for(int d0=0;d0<size[0];d0++){
	  for(int d2=0;d2<size[2];d2++){
	    for(int b=0; b<size[3]; b++){
	      size_t o = d2 + size[2]*d0;
	      size_t base = batchTensorDimensionBaseLin<4>(1, b, o, size);

	      for(int d1=0;d1<size[1];d1++)      
		assert( tens_v.data()[base + d1*stride1] == tens_v(d0,d1,d2,b) );
	    }
	  }
	}

	//iter_dim=2
	for(int d0=0;d0<size[0];d0++){
	  for(int d1=0;d1<size[1];d1++){
	    for(int b=0; b<size[3]; b++){
	      size_t o = d1 + size[1]*d0;
	      size_t base = batchTensorDimensionBaseLin<4>(2, b, o, size);

	      for(int d2=0;d2<size[2];d2++)      
		assert( tens_v.data()[base + d2*stride2] == tens_v(d0,d1,d2,b) );
	    }
	  }
	}
	
      });
  }

      
  std::cout << "testDimensionIteration passed" << std::endl;
}

	
void testConcatenateSplit(){
  typedef double FloatType;
  std::mt19937 rng(1234);
  
  { //contract dim 2
    int size1[4] = {2,3,4,5};
    int size2[4] = {2,3,3,5};
    int size3[4] = {2,3,6,5};

    std::vector<Tensor<FloatType,4>* > tens({ new Tensor<FloatType,4>(size1), new Tensor<FloatType,4>(size2), new Tensor<FloatType,4>(size3) });
    for(int i=0;i<3;i++)
      uniformRandom(*tens[i],rng);

    Tensor<FloatType,4> got = batchTensorConcatenate(tens.data(), 3,  2);
    
    int osize[4] = {2,3,4+3+6,5};
    Tensor<FloatType,4> expect(osize, 0.);
    int off = 0;
    for(int t=0;t<3;t++){
      autoView(out_v,expect,HostReadWrite);
      autoView(in_v, (*tens[t]), HostRead);
      int csz = tens[t]->size(2);
      
      for(int i=0;i<2;i++)
	for(int j=0;j<3;j++)
	  for(int b=0;b<5;b++)
	    for(int k=0;k<csz;k++)
	      out_v(i,j,off + k,b) = in_v(i,j,k,b);
      off += csz;
    }

    assert(equal(got,expect,true));

    std::vector<Tensor<FloatType,4>* > split_tens({ new Tensor<FloatType,4>(size1), new Tensor<FloatType,4>(size2), new Tensor<FloatType,4>(size3) });
    
    batchTensorSplit(split_tens.data(), 3, got, 2);
    
    for(int t=0;t<3;t++){
      assert( equal(*split_tens[t], *tens[t], true) );
      
      delete tens[t];
      delete split_tens[t];
    }
  }

  { //contract dim 1
    int size1[4] = {2,4,3,5};
    int size2[4] = {2,3,3,5};
    int size3[4] = {2,6,3,5};

    std::vector<Tensor<FloatType,4>* > tens({ new Tensor<FloatType,4>(size1), new Tensor<FloatType,4>(size2), new Tensor<FloatType,4>(size3) });
    for(int i=0;i<3;i++)
      uniformRandom(*tens[i],rng);

    Tensor<FloatType,4> got = batchTensorConcatenate<4,FloatType>(tens.data(), 3,  1);
    
    int osize[4] = {2,4+3+6,3,5};
    Tensor<FloatType,4> expect(osize, 0.);
    int off = 0;
    for(int t=0;t<3;t++){
      autoView(out_v,expect,HostReadWrite);
      autoView(in_v, (*tens[t]), HostRead);
      int csz = tens[t]->size(1);
      
      for(int i=0;i<2;i++)	  
	for(int k=0;k<3;k++)
	  for(int b=0;b<5;b++)
	    for(int j=0;j<csz;j++)
	      out_v(i,off + j,k,b) = in_v(i,j,k,b);
      off += csz;
    }

    assert(equal(got,expect,true));

    std::vector<Tensor<FloatType,4>* > split_tens({ new Tensor<FloatType,4>(size1), new Tensor<FloatType,4>(size2), new Tensor<FloatType,4>(size3) });
    
    batchTensorSplit(split_tens.data(), 3, got, 1);
    
    for(int t=0;t<3;t++){
      assert( equal(*split_tens[t], *tens[t], true) );
      
      delete tens[t];
      delete split_tens[t];
    }
  }

  { //contract dim 0
    int size1[4] = {4,2,3,5};
    int size2[4] = {3,2,3,5};
    int size3[4] = {6,2,3,5};

    std::vector<Tensor<FloatType,4>* > tens({ new Tensor<FloatType,4>(size1), new Tensor<FloatType,4>(size2), new Tensor<FloatType,4>(size3) });
    for(int i=0;i<3;i++)
      uniformRandom(*tens[i],rng);

    Tensor<FloatType,4> got = batchTensorConcatenate<4,FloatType>(tens.data(), 3,  0);
    
    int osize[4] = {4+3+6,2,3,5};
    Tensor<FloatType,4> expect(osize, 0.);
    int off = 0;
    for(int t=0;t<3;t++){
      autoView(out_v,expect,HostReadWrite);
      autoView(in_v, (*tens[t]), HostRead);
      int csz = tens[t]->size(0);
      
      for(int j=0;j<2;j++)	  
	for(int k=0;k<3;k++)
	  for(int b=0;b<5;b++)
	    for(int i=0;i<csz;i++)
	      out_v(off + i,j,k,b) = in_v(i,j,k,b);
      off += csz;
    }

    assert(equal(got,expect,true));

    std::vector<Tensor<FloatType,4>* > split_tens({ new Tensor<FloatType,4>(size1), new Tensor<FloatType,4>(size2), new Tensor<FloatType,4>(size3) });
    
    batchTensorSplit(split_tens.data(), 3, got, 0);
    
    for(int t=0;t<3;t++){
      assert( equal(*split_tens[t], *tens[t], true) );
      
      delete tens[t];
      delete split_tens[t];
    }
    
  }
  std::cout << "testConcatenate passed" << std::endl;
}

  
int main(int argc, char** argv){
  initialize(argc,argv);
  
  testTensor();
  testDimensionIteration();
  testConcatenateSplit();
  return 0;
}
