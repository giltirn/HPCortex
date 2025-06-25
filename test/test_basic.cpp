#include <HPCortex.hpp>
#include <Testing.hpp>

void basicTests(){
  typedef float FloatType;
  FloatType delta = 1e-4;
  
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
    

  
  
  Matrix<FloatType> w1_init(3,2, vecD({0.1,0.2,
   	                             -0.1,-0.2,
			              0.7,0.7}));
  Vector<FloatType> b1_init( vecD({0.5,0.7,0.9}));		    

  doHost(w1_init, {  assert(w1_init_v(0,0) == FloatType(0.1) && w1_init_v(1,0) == FloatType(-0.1) ); });
  {
    auto c = peekColumn(w1_init,0);
    doHost(c, { assert(c_v(0) == FloatType(0.1) && c_v(1) == FloatType(-0.1) && c_v(2) == FloatType(0.7) ); });
  }
  
  auto f = mse_cost( dnn_layer(w1_init, b1_init, input_layer<FloatType>()) );

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
    Vector<FloatType> y1_b = peekColumn(y1,i);
    std::cout << y1pred << " " << y1_b << std::endl;

    doHost2(y1pred,y1_b,{
    expect += pow(y1pred_v(0)-y1_b_v(0),2)/3. + pow(y1pred_v(1)-y1_b_v(1),2)/3. + pow(y1pred_v(2)-y1_b_v(2),2)/3.;
      });
  }
  expect /= 2.;
    
  FloatType got=  f.loss(x1,y1);
  std::cout << "Test loss : got " << got << " expect " << expect << std::endl;
  assert(near(got,expect,FloatType(1e-4)));

  //test the derivatives
  Vector<FloatType> dexpect(9);
  {
    autoView(dexpect_v,dexpect,HostWrite); 
    int p=0;
    for(int i=0;i<3;i++){
      for(int j=0;j<2;j++){
	Matrix<FloatType> w1_p = w1_init;
	doHost(w1_p, { w1_p_v(i,j) += delta; });
	auto f2 = mse_cost( dnn_layer(w1_p, b1_init, input_layer<FloatType>()) );
	dexpect_v(p++) = (f2.loss(x1,y1) - got)/delta;
      }
    }
    for(int i=0;i<3;i++){
      Vector<FloatType> b1_p = b1_init;
      doHost(b1_p, { b1_p_v(i) += delta; });      
      auto f2 = mse_cost( dnn_layer(w1_init, b1_p, input_layer<FloatType>()) );
      dexpect_v(p++) = (f2.loss(x1,y1) - got)/delta;    
    }
  }
  
  Vector<FloatType> dgot = f.deriv();
  doHost2(dgot,dexpect,{
    for(int i=0;i<9;i++){
      std::cout << "Test deriv wrt param " << i <<  ": got " << dgot_v(i) << " expect " << dexpect_v(i) << std::endl;
    }
    });
  assert(near(dgot,dexpect,FloatType(5e-3),true));
  
  //test update
  Matrix<FloatType> w1_new(3,2,    vecD({-0.5,0.4,
					  0.8,1.2,
					  2.1,-3.0}));
  Vector<FloatType> b1_new( vecD({-0.5,0.7,-1.1}));	

  auto ftest = mse_cost( dnn_layer(w1_new, b1_new, input_layer<FloatType>()) );
  f.update(ftest.getParams());

  FloatType expect_l = ftest.loss(x1,y1);
  FloatType got_l =  f.loss(x1,y1);
  std::cout << "Update check : expect " << expect_l  << " got " << got_l << std::endl;
  assert(near(got_l,expect_l,FloatType(1e-4)));
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
  
}


int main(int argc, char** argv){
  initialize(argc,argv);
  
  basicTests();
  testTensorOffset();
  return 0;
}
