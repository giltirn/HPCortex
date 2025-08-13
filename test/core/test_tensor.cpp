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
	  size_t off = tensorOffset<3>(coord,tens_device_v.sizeArray());
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

    { //test sliceLastDimension
      int sz[3] = {2,3,6};
      Tensor<FloatType,3> orig(sz);
      uniformRandom(orig,rng);

      Tensor<FloatType,3> slice = orig.sliceLastDimension(1,4);
      assert(slice.size(2) == 4);
      int slice_sz[3] = {2,3,4};
      Tensor<FloatType,3> slice_expect(slice_sz);
      doHost2(slice_expect,orig,{
	  for(int i=0;i<2;i++)
	    for(int j=0;j<3;j++)
	      for(int k=1; k<=4; k++)
		slice_expect_v(i,j,k-1) = orig_v(i,j,k);
	});
      assert(equal(slice_expect,slice,true));

      Tensor<FloatType,3> orig2(sz);
      uniformRandom(orig2,rng);

      Tensor<FloatType,3> ins = orig2;
      ins.insertSliceLastDimension(slice, 1,4);

      Tensor<FloatType,3> ins_expect = orig2;
      doHost2(ins_expect,slice,{
	  for(int i=0;i<2;i++)
	    for(int j=0;j<3;j++)
	      for(int k=1; k<=4; k++)
		ins_expect_v(i,j,k) = slice_v(i,j,k-1);
	});
      assert(equal(ins_expect,ins,true));
    }

    
  }

  //Test norm2
  {
    int sz[3] = {3,4,5};
    Tensor<FloatType, 3> T(sz);
    uniformRandom(T,rng);
    
    double expect = 0.;
    {
      autoView(T_v,T,HostRead);
      for(size_t i=0; i<T.data_len(); i++){
	FloatType v = T_v.data()[i];
	expect += v*v;
      }
    }

    double got = norm2(T);
    std::cout << "norm2 test got " << got << " expect " << expect << " diff " << got-expect << std::endl;
    assert(abs_near(expect,got,FloatType(1e-6)));
  }
    
  std::cout << "testTensor passed" << std::endl;
}


void testMatrix(){
  typedef float FloatType;
  
  typedef std::vector<FloatType> vecD;

  //Test peekColumns
  {
    Matrix<FloatType> m(2,3, vecD({ 1.,2.,3.,
	                            4.,5.,6.  }));
    auto mv = peekColumns(m,1,2);
    assert(mv.size(0) == 2 && mv.size(1) == 2);    
    doHost(mv, { assert(mv_v(0,0) == FloatType(2.) && mv_v(0,1) == FloatType(3.) && mv_v(1,0) == FloatType(5.) && mv_v(1,1) == FloatType(6.)); });
  }   
  //Test pokeColumns
  {
    Matrix<FloatType> m(2,3, vecD({ 1.,2.,3.,
	                            4.,5.,6.  }));
    Matrix<FloatType> v(2,2, vecD({5.,6.,
	                           7.,8.}));   
    pokeColumns(m,1,2,v);
    
    doHost(m, { assert(m_v(0,1) == FloatType(5.) && m_v(0,2) == FloatType(6.) && m_v(1,1) == FloatType(7.) && m_v(1,2) == FloatType(8.)); });
  }
  //Test matrix-vector linalg
  {
    Matrix<FloatType> m(2,3, vecD({ -0.1, 0.1, 0.3,
              	                    0.7, -0.3, 0.25 }));
    Vector<FloatType> v(vecD({ 4.56, 3.14, -2.56 }));
    Vector<FloatType> v2(vecD({ 7.14, -4.13, -5.66 }));

    //m*v
    {
      Vector<FloatType> got = m * v;
      Vector<FloatType> expect(2, 0., MemoryManager::Pool::HostPool);
      {
	autoView(e_v,expect,HostReadWrite);
	autoView(m_v,m,HostRead);
	autoView(v_v,v,HostRead);
	for(int i=0;i<2;i++)
	  for(int j=0;j<3;j++)
	    e_v(i) += m_v(i,j) * v_v(j);
      }
      assert(near(expect,got,FloatType(1e-5)));
    }
    //v + v2,  v += v2
    {
      Vector<FloatType> got = v + v2;
      Vector<FloatType> expect(3, MemoryManager::Pool::HostPool);
      {
	autoView(e_v,expect,HostWrite);
	autoView(v_v,v,HostRead);
	autoView(v2_v,v2,HostRead);
	for(int i=0;i<3;i++)
	  e_v(i) = v_v(i) + v2_v(i);
      }
      assert(near(expect,got,FloatType(1e-5)));

      got = v;
      got += v2;
      assert(near(expect,got,FloatType(1e-5)));
    }
    //v - v2
    {
      Vector<FloatType> got = v - v2;
      Vector<FloatType> expect(3, MemoryManager::Pool::HostPool);
      {
	autoView(e_v,expect,HostWrite);
	autoView(v_v,v,HostRead);
	autoView(v2_v,v2,HostRead);
	for(int i=0;i<3;i++)
	  e_v(i) = v_v(i) - v2_v(i);
      }
      assert(near(expect,got,FloatType(1e-5)));
    }
    //v * eps, v*= eps
    {
      FloatType eps = 0.123;
      Vector<FloatType> got = eps * v;
      Vector<FloatType> expect(3, MemoryManager::Pool::HostPool);
      {
	autoView(e_v,expect,HostWrite);
	autoView(v_v,v,HostRead);
	for(int i=0;i<3;i++)
	  e_v(i) = v_v(i) * eps;
      }
      assert(near(expect,got,FloatType(1e-5)));

      got = v;
      got *= eps;

      assert(near(expect,got,FloatType(1e-5)));
    }    
  }
  std::cout << "testMatrix passed" << std::endl;
}




template<int Dim>
accelerator_inline size_t batchTensorDimensionBaseLinOrig(int iter_dim, int batch_idx, size_t other_dim_lin, int const *size){
  int coord[Dim];
  coord[iter_dim]=0;
  coord[Dim-1] = batch_idx;
  size_t rem = other_dim_lin;

  //other_dim_lin for, eg 3 dims, mapped as     z + dim3*( y + dim2 * x )
  for(int d=Dim-2;d>=0;d--)
    if(d!=iter_dim){
      coord[d] = rem % size[d];
      rem /= size[d];
    }
  return tensorOffset<Dim>(coord, size);
}


void testTensorOffset(){

  {
    int size[4] = {2,3,4,5};
    for(int iter_dim=0;iter_dim<3;iter_dim++){
      size_t other_dim_sz = 1;
      for(int d=0;d<3;d++)
	if(d!=iter_dim)
	  other_dim_sz *= size[d];
      for(size_t o=0; o<other_dim_sz; o++)
	for(int b=0;b< size[3]; b++)
	  assert(batchTensorDimensionBaseLin<4>(iter_dim,b,o,size) == batchTensorDimensionBaseLinOrig<4>(iter_dim,b,o,size) );
    }
  }
  
  {
    int size[3] = {2,4,5};
    for(int iter_dim=0;iter_dim<2;iter_dim++){
      size_t other_dim_sz = iter_dim == 0 ? size[1] : size[0];

      for(size_t o=0; o<other_dim_sz; o++)
	for(int b=0;b< size[2]; b++)
	  assert(batchTensorDimensionBaseLin<3>(iter_dim,b,o,size) == batchTensorDimensionBaseLinOrig<3>(iter_dim,b,o,size) );
    }
  }

  {
    int size[2] = {3,5};
    int iter_dim = 0;
    int other_dim_sz = 1;

    for(size_t o=0; o<other_dim_sz; o++)
      for(int b=0;b< size[1]; b++)
	assert(batchTensorDimensionBaseLin<2>(iter_dim,b,o,size) == batchTensorDimensionBaseLinOrig<2>(iter_dim,b,o,size) );    
  }
  std::cout << "testTensorOffset passed" << std::endl;
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

void testDimensionSlice(){
  std::mt19937 rng(1234);
  int tens_sz[3] = {5,6,7};  
  Tensor<double,3> tens(tens_sz);
  uniformRandom(tens, rng);
  
  {
    //dim 0
    std::vector<int> sub_idx({2,0,3});
    Tensor<double,3> got = dimensionSlice(tens, sub_idx, 0, Host);

    int expect_sz[3] = {3,6,7};
    Tensor<double,3> expect(expect_sz);
    {
      autoView(e_v,expect,HostWrite);
      autoView(f_v,tens,HostRead);
      for(int ii=0;ii<3;ii++){
	int idx = sub_idx[ii];
	for(int j=0;j<6;j++)
	  for(int k=0;k<7;k++)
	    e_v(ii,j,k) = f_v(idx,j,k);
      }
    }
    assert(equal(got,expect,true));

    got = dimensionSlice(tens, sub_idx, 0, Device);
    assert(equal(got,expect,true));
  }
  {
    //dim 1
    std::vector<int> sub_idx({1,4,2,5});
    Tensor<double,3> got = dimensionSlice(tens, sub_idx, 1, Host);

    int expect_sz[3] = {5,4,7};
    Tensor<double,3> expect(expect_sz);
    {
      autoView(e_v,expect,HostWrite);
      autoView(f_v,tens,HostRead);
      for(int i=0;i<5;i++){
	for(int jj=0;jj<4;jj++){
	  int idx = sub_idx[jj];
	  for(int k=0;k<7;k++)
	    e_v(i,jj,k) = f_v(i,idx,k);
	}
      }
    }
    assert(equal(got,expect,true));

    got = dimensionSlice(tens, sub_idx, 1, Device);
    assert(equal(got,expect,true));
  }
  {
    //dim 2
    std::vector<int> sub_idx({6,5,4,3});
    Tensor<double,3> got = dimensionSlice(tens, sub_idx, 2, Host);

    int expect_sz[3] = {5,6,4};
    Tensor<double,3> expect(expect_sz);
    {
      autoView(e_v,expect,HostWrite);
      autoView(f_v,tens,HostRead);
      for(int i=0;i<5;i++){
	for(int j=0;j<6;j++){
	  for(int kk=0;kk<4;kk++){
	  int idx = sub_idx[kk];
	  e_v(i,j,kk) = f_v(i,j,idx);
	  }
	}
      }
    }
    assert(equal(got,expect,true));
    
    got = dimensionSlice(tens, sub_idx, 2, Device);
    assert(equal(got,expect,true));
  }
  std::cout << "testDimensionSlice passed" << std::endl;
}

void testNormalize(){
  std::mt19937 rng(1234);
  int tens_sz[3] = {5,6,7};  
  Tensor<double,3> tens(tens_sz);
  uniformRandom(tens, rng);
  
  {
    //dim 0
    Tensor<double,3> got(tens);     
    auto nrm = normalize(got, 0, Host);

    Tensor<double,3> expect(tens_sz);
    {
      autoView(e_v,expect,HostWrite);
      autoView(t_v,tens,HostRead);
      for(int j=0;j<6;j++){
	for(int k=0;k<7;k++){
	  double mu =0, std=0;
	  for(int i=0;i<5;i++){
	    mu += t_v(i,j,k);
	    std += pow(t_v(i,j,k),2);
	  }
	  mu /= 5;
	  std = sqrt(std/5 - mu*mu);
	  
	  for(int i=0;i<5;i++)
	    e_v(i,j,k) = (t_v(i,j,k) - mu)/std;
	}
      }
    }

    assert(abs_near(got,expect,1e-7,true));

    unnormalize(got, 0, nrm, Host);
    assert(abs_near(got,tens,1e-7,true));
    
    got = tens;
    nrm = normalize(got, 0, Device);
    assert(abs_near(got,expect,1e-7,true));

    unnormalize(got, 0, nrm, Device);
    assert(abs_near(got,tens,1e-7,true));    
  }

  {
    //dim 1
    Tensor<double,3> got(tens);     
    auto nrm = normalize(got, 1, Host);

    Tensor<double,3> expect(tens_sz);
    {
      autoView(e_v,expect,HostWrite);
      autoView(t_v,tens,HostRead);
      for(int i=0;i<5;i++){
	for(int k=0;k<7;k++){
	  double mu =0, std=0;
	  for(int j=0;j<6;j++){
	    mu += t_v(i,j,k);
	    std += pow(t_v(i,j,k),2);
	  }
	  mu /= 6;
	  std = sqrt(std/6 - mu*mu);
	  
	  for(int j=0;j<6;j++)
	    e_v(i,j,k) = (t_v(i,j,k) - mu)/std;
	}
      }
    }

    assert(abs_near(got,expect,1e-7,true));

    unnormalize(got, 1, nrm, Host);
    assert(abs_near(got,tens,1e-7,true));
    
    got = tens;
    nrm = normalize(got, 1, Device);
    assert(abs_near(got,expect,1e-7,true));

    unnormalize(got, 1, nrm, Device);
    assert(abs_near(got,tens,1e-7,true));    
  }


  {
    //dim 2
    Tensor<double,3> got(tens);     
    auto nrm = normalize(got, 2, Host);

    Tensor<double,3> expect(tens_sz);
    {
      autoView(e_v,expect,HostWrite);
      autoView(t_v,tens,HostRead);
      for(int i=0;i<5;i++){
	for(int j=0;j<6;j++){
	  double mu =0, std=0;
	  for(int k=0;k<7;k++){
	    mu += t_v(i,j,k);
	    std += pow(t_v(i,j,k),2);
	  }
	  mu /= 7;
	  std = sqrt(std/7 - mu*mu);
	  
	  for(int k=0;k<7;k++)
	    e_v(i,j,k) = (t_v(i,j,k) - mu)/std;
	}
      }
    }

    assert(abs_near(got,expect,1e-7,true));

    unnormalize(got, 2, nrm, Host);
    assert(abs_near(got,tens,1e-7,true));
    
    got = tens;
    nrm = normalize(got, 2, Device);
    assert(abs_near(got,expect,1e-7,true));

    unnormalize(got, 2, nrm, Device);
    assert(abs_near(got,tens,1e-7,true));    
  }
  
  std::cout << "testNormalize passed" << std::endl;
}
  
  


int main(int argc, char** argv){
  initialize(argc,argv);
  
  testTensor();
  testMatrix();
  testTensorOffset();
  testDimensionIteration();
  testConcatenateSplit();
  testDimensionSlice();
  testNormalize();
  return 0;
}
