#include <HPCortex.hpp>
#include <Testing.hpp>

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
  
int main(int argc, char** argv){
  initialize(argc,argv);
  testrmGEMM();
  return 0;
}
  
