#include <HPCortex.hpp>
#include <Testing.hpp>

#ifdef USE_BLAS

void testrmGEMM(){
  std::mt19937 rng(1234);

  {
    //N, N  
    Matrix<double> A(2,3), B(3,4), Cexpect(2,4,0.);
    doHost3(A,B,Cexpect,
	    {
	      for(int m=0;m<2;m++)
		for(int n=0;n<4;n++)
		  for(int k=0;k<3;k++)
		    Cexpect_v(m,n) += A_v(m,k)*B_v(k,n);
	    });
    Matrix<double> Cgot(2,4);
    {
      autoView(Cgot_v,Cgot,DeviceWrite);
      autoView(A_v,A,DeviceRead);
      autoView(B_v,B,DeviceRead);
      rmGEMM(NoTranspose,NoTranspose,
	    2,4,3,
	    1.0,
	    A_v.data(), 3,
	    B_v.data(), 4,
	    0.0,
	    Cgot_v.data());
    }
    assert(abs_near(Cgot,Cexpect,1e-6,true));
  }
  {
    //N, T
    Matrix<double> A(2,3), B(4,3), Cexpect(2,4,0.);
    doHost3(A,B,Cexpect,
	    {
	      for(int m=0;m<2;m++)
		for(int n=0;n<4;n++)
		  for(int k=0;k<3;k++)
		    Cexpect_v(m,n) += A_v(m,k)*B_v(n,k);
	    });
    Matrix<double> Cgot(2,4);
    {
      autoView(Cgot_v,Cgot,DeviceWrite);
      autoView(A_v,A,DeviceRead);
      autoView(B_v,B,DeviceRead);
      rmGEMM(NoTranspose,Transpose,
	    2,4,3,
	    1.0,
	    A_v.data(), 3,
	    B_v.data(), 3,
	    0.0,
	    Cgot_v.data());
    }
    assert(abs_near(Cgot,Cexpect,1e-6,true));
  }
  {
    //T, N
    Matrix<double> A(3,2), B(3,4), Cexpect(2,4,0.);
    doHost3(A,B,Cexpect,
	    {
	      for(int m=0;m<2;m++)
		for(int n=0;n<4;n++)
		  for(int k=0;k<3;k++)
		    Cexpect_v(m,n) += A_v(k,m)*B_v(k,n);
	    });
    Matrix<double> Cgot(2,4);
    {
      autoView(Cgot_v,Cgot,DeviceWrite);
      autoView(A_v,A,DeviceRead);
      autoView(B_v,B,DeviceRead);
      rmGEMM(Transpose,NoTranspose,
	    2,4,3,
	    1.0,
	    A_v.data(), 3,
	    B_v.data(), 4,
	    0.0,
	    Cgot_v.data());
    }
    assert(abs_near(Cgot,Cexpect,1e-6,true));
  }
  {
    //T, T
    Matrix<double> A(3,2), B(4,3), Cexpect(2,4,0.);
    doHost3(A,B,Cexpect,
	    {
	      for(int m=0;m<2;m++)
		for(int n=0;n<4;n++)
		  for(int k=0;k<3;k++)
		    Cexpect_v(m,n) += A_v(k,m)*B_v(n,k);
	    });
    Matrix<double> Cgot(2,4);
    {
      autoView(Cgot_v,Cgot,DeviceWrite);
      autoView(A_v,A,DeviceRead);
      autoView(B_v,B,DeviceRead);
      rmGEMM(Transpose,Transpose,
	    2,4,3,
	    1.0,
	    A_v.data(), 2,
	    B_v.data(), 3,
	    0.0,
	    Cgot_v.data());
    }
    assert(abs_near(Cgot,Cexpect,1e-6,true));
  }
  std::cout << "testrmGEMM passed" << std::endl;
}

void testrmBatchedGEMM(){
  std::mt19937 rng(1234);

  {
    //N, N
    Tensor<double,3> A(3,   2,3);
    Tensor<double,3> B(3,   3,4);
    Tensor<double,3> Cexpect(3,   2,4, 0.);
    Tensor<double,3> C(3,   2,4);
    uniformRandom(A,rng);
    uniformRandom(B,rng);
        
    doHost3(A,B,Cexpect,
	    {
	      for(int i=0;i<3;i++)		
		for(int m=0;m<2;m++)
		  for(int n=0;n<4;n++)
		    for(int k=0;k<3;k++)
		      Cexpect_v(i,m,n) += A_v(i,m,k)*B_v(i,k,n);
	    });

    {
      autoView(A_v,A,DeviceRead);
      autoView(B_v,B,DeviceRead);
      autoView(C_v,C,DeviceWrite);
      
      rmBatchedGEMM(NoTranspose, NoTranspose,
		    2,4,3,
		    1.0,
		    A_v.data(),A.size(2),A.size(1)*A.size(2),
		    B_v.data(),B.size(2),B.size(1)*B.size(2),
		    0.0,
		    C_v.data(),C.size(1)*C.size(2),
		    3);
    }    
    assert(abs_near(C,Cexpect,1e-6,true));
  }

  {
    //N, T
    Tensor<double,3> A(3,   2,3);
    Tensor<double,3> B(3,   4,3);
    Tensor<double,3> Cexpect(3,   2,4, 0.);
    Tensor<double,3> C(3,   2,4);
    uniformRandom(A,rng);
    uniformRandom(B,rng);
        
    doHost3(A,B,Cexpect,
	    {
	      for(int i=0;i<3;i++)		
		for(int m=0;m<2;m++)
		  for(int n=0;n<4;n++)
		    for(int k=0;k<3;k++)
		      Cexpect_v(i,m,n) += A_v(i,m,k)*B_v(i,n,k);
	    });

    {
      autoView(A_v,A,DeviceRead);
      autoView(B_v,B,DeviceRead);
      autoView(C_v,C,DeviceWrite);
      
      rmBatchedGEMM(NoTranspose, Transpose,
		    2,4,3,
		    1.0,
		    A_v.data(),A.size(2),A.size(1)*A.size(2),
		    B_v.data(),B.size(2),B.size(1)*B.size(2),
		    0.0,
		    C_v.data(),C.size(1)*C.size(2),
		    3);
    }    
    assert(abs_near(C,Cexpect,1e-6,true));
  }

  {
    //T, N
    Tensor<double,3> A(3,   3,2);
    Tensor<double,3> B(3,   3,4);
    Tensor<double,3> Cexpect(3,   2,4, 0.);
    Tensor<double,3> C(3,   2,4);
    uniformRandom(A,rng);
    uniformRandom(B,rng);
        
    doHost3(A,B,Cexpect,
	    {
	      for(int i=0;i<3;i++)		
		for(int m=0;m<2;m++)
		  for(int n=0;n<4;n++)
		    for(int k=0;k<3;k++)
		      Cexpect_v(i,m,n) += A_v(i,k,m)*B_v(i,k,n);
	    });

    {
      autoView(A_v,A,DeviceRead);
      autoView(B_v,B,DeviceRead);
      autoView(C_v,C,DeviceWrite);
      
      rmBatchedGEMM(Transpose, NoTranspose,
		    2,4,3,
		    1.0,
		    A_v.data(),A.size(2),A.size(1)*A.size(2),
		    B_v.data(),B.size(2),B.size(1)*B.size(2),
		    0.0,
		    C_v.data(),C.size(1)*C.size(2),
		    3);
    }    
    assert(abs_near(C,Cexpect,1e-6,true));
  }

  {
    //T, T
    Tensor<double,3> A(3,   3,2);
    Tensor<double,3> B(3,   4,3);
    Tensor<double,3> Cexpect(3,   2,4, 0.);
    Tensor<double,3> C(3,   2,4);
    uniformRandom(A,rng);
    uniformRandom(B,rng);
        
    doHost3(A,B,Cexpect,
	    {
	      for(int i=0;i<3;i++)		
		for(int m=0;m<2;m++)
		  for(int n=0;n<4;n++)
		    for(int k=0;k<3;k++)
		      Cexpect_v(i,m,n) += A_v(i,k,m)*B_v(i,n,k);
	    });

    {
      autoView(A_v,A,DeviceRead);
      autoView(B_v,B,DeviceRead);
      autoView(C_v,C,DeviceWrite);
      
      rmBatchedGEMM(Transpose, Transpose,
		    2,4,3,
		    1.0,
		    A_v.data(),A.size(2),A.size(1)*A.size(2),
		    B_v.data(),B.size(2),B.size(1)*B.size(2),
		    0.0,
		    C_v.data(),C.size(1)*C.size(2),
		    3);
    }    
    assert(abs_near(C,Cexpect,1e-6,true));
  }

  std::cout << "testrmBatchedGEMM passed" << std::endl;
}


void testrmBatchedGEMV(){
  std::mt19937 rng(1234);

 
  {
    //N   
    Tensor<double,3> A(3,   2,3);
    Tensor<double,2> x(3,   3);
    Tensor<double,2> y(3,   2);
    uniformRandom(A,rng);
    uniformRandom(x,rng);
    uniformRandom(y,rng);

    Tensor<double,2> yexpect(y);
    
    doHost4(A,x,y, yexpect,
	    {
	      for(int i=0;i<3;i++)		
		for(int m=0;m<2;m++)
		    for(int k=0;k<3;k++)
		      yexpect_v(i, m) += A_v(i,  m,k)*x_v(i,  k);
	    });

    Tensor<double,2> ygot(y);

    //strided
    {
      
      autoView(A_v,A,DeviceRead);
      autoView(x_v,x,DeviceRead);
      autoView(ygot_v,ygot,DeviceReadWrite);


      rmBatchedGEMV(NoTranspose,
		    A.size(1),A.size(2),
		    1.0,
		    A_v.data(),A.size(1)*A.size(2),
		    x_v.data(), 1, x.size(1),
		    1.0,
		    ygot_v.data(), 1, ygot.size(1),
		    3);
    }    
    assert(abs_near(ygot,yexpect,1e-6,true));

    Tensor<double,2> ygot2(y);
    //group
    {
      
      autoView(A_v,A,DeviceRead);
      autoView(x_v,x,DeviceRead);
      autoView(ygot2_v,ygot2,DeviceReadWrite);
     
      size_t Astride = A.size(1)*A.size(2);
      size_t xstride = x.size(1);
      size_t ystride = y.size(1);

      ManagedArray<double*> Aarray(3), xarray(3), ygot2_array(3);
    
      autoView(Aarray_v,Aarray,DeviceWrite);
      autoView(xarray_v,xarray,DeviceWrite);
      autoView(ygot2_array_v,ygot2_array,DeviceWrite);      
      accelerator_for_gen(0,1,normal(),o,3,{
	  Aarray_v[o] = A_v.data() + o*Astride;
	  xarray_v[o] = x_v.data() + o*xstride;
	  ygot2_array_v[o] = ygot2_v.data() + o*ystride;
      });

      
      rmBatchedGEMV(NoTranspose,
		    A.size(1),A.size(2),
		    1.0,
		    Aarray_v.data(),
		    xarray_v.data(), 1,
		    1.0,
		    ygot2_array_v.data(), 1,
		    3);
      
    }
    assert(abs_near(ygot2,yexpect,1e-6,true));
    
  }


  {
    //T
    Tensor<double,3> A(3,   3,2);
    Tensor<double,2> x(3,   3);
    Tensor<double,2> y(3,   2);
    uniformRandom(A,rng);
    uniformRandom(x,rng);
    uniformRandom(y,rng);

    Tensor<double,2> yexpect(y);
    
    doHost4(A,x,y, yexpect,
	    {
	      for(int i=0;i<3;i++)		
		for(int m=0;m<2;m++)
		    for(int k=0;k<3;k++)
		      yexpect_v(i, m) += A_v(i,  k,m)*x_v(i,  k);
	    });

    Tensor<double,2> ygot(y);

    //strided
    {
      
      autoView(A_v,A,DeviceRead);
      autoView(x_v,x,DeviceRead);
      autoView(ygot_v,ygot,DeviceReadWrite);


      rmBatchedGEMV(Transpose,
		    A.size(1),A.size(2),
		    1.0,
		    A_v.data(),A.size(1)*A.size(2),
		    x_v.data(), 1, x.size(1),
		    1.0,
		    ygot_v.data(), 1, ygot.size(1),
		    3);
    }    
    assert(abs_near(ygot,yexpect,1e-6,true));

    Tensor<double,2> ygot2(y);
    //group
    {
      
      autoView(A_v,A,DeviceRead);
      autoView(x_v,x,DeviceRead);
      autoView(ygot2_v,ygot2,DeviceReadWrite);
     
      size_t Astride = A.size(1)*A.size(2);
      size_t xstride = x.size(1);
      size_t ystride = y.size(1);

      ManagedArray<double*> Aarray(3), xarray(3), ygot2_array(3);
    
      autoView(Aarray_v,Aarray,DeviceWrite);
      autoView(xarray_v,xarray,DeviceWrite);
      autoView(ygot2_array_v,ygot2_array,DeviceWrite);      
      accelerator_for_gen(0,1,normal(),o,3,{
	  Aarray_v[o] = A_v.data() + o*Astride;
	  xarray_v[o] = x_v.data() + o*xstride;
	  ygot2_array_v[o] = ygot2_v.data() + o*ystride;
      });

      
      rmBatchedGEMV(Transpose,
		    A.size(1),A.size(2),
		    1.0,
		    Aarray_v.data(),
		    xarray_v.data(), 1,
		    1.0,
		    ygot2_array_v.data(), 1,
		    3);
      
    }
    assert(abs_near(ygot2,yexpect,1e-6,true));
    
  }

  
  std::cout << "testrmBatchedGEMV passed" << std::endl;
}

int main(int argc, char** argv){
  initialize(argc,argv);
  testrmGEMM();
  testrmBatchedGEMM();
  testrmBatchedGEMV();
  return 0;
}
  
#else
int main(void){
  return 0;
}
#endif
