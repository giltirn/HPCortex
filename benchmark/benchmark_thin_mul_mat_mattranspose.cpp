#include<HPCortex.hpp>
#include<Testing.hpp>

//338.19us 15.0451us
// C_jk = \sum_i  A_ji B_ki
template<typename FloatType>
Matrix<FloatType> thinMulMatMatTransposeBase(const Matrix<FloatType> &a, const Matrix<FloatType> &b){
  int szj = a.size(0);
  int szi = a.size(1);
  int szk = b.size(0);
  assert(b.size(1) == szi);
  
  Matrix<FloatType> out(szj,szk);
  autoView(out_v,out,DeviceWrite);
  autoView(a_v,a,DeviceRead);
  autoView(b_v,b,DeviceRead);
    
  accelerator_for3d(dummy,1, k,szk,j,szj,   64,{
      FloatType v = a_v(j,0) * b_v(k,0);
      for(int i=1;i<szi;i++)
	v += a_v(j,i) * b_v(k,i);
      out_v(j,k) = v;
    });
  
  return out;  
}

//131.56us 15.9946us
template<typename FloatType>
Matrix<FloatType> thinMulMatMatTranspose_v2(const Matrix<FloatType> &a, const Matrix<FloatType> &b){
  int szj = a.size(0);
  int szi = a.size(1);
  int szk = b.size(0);
  assert(b.size(1) == szi);
  
  Matrix<FloatType> out(szj,szk);
  autoView(out_v,out,DeviceWrite);
  autoView(a_v,a,DeviceRead);
  autoView(b_v,b,DeviceRead);

  int jblocksize = 8;
  int jblocks = (szj + jblocksize-1)/jblocksize;

  int kblocksize = 8;
  int kblocks = (szk + kblocksize-1)/kblocksize;
  
  accelerator_for3d(jk,jblocksize*kblocksize, bk, kblocks,bj,jblocks,   1,{
      //jk = kk+kblocksize*jj
      int kk = jk % kblocksize;
      int jj = jk / kblocksize;

      int j = jj + jblocksize*bj;
      int k = kk + kblocksize*bk;
      
      if(j < szj && k < szk){
	FloatType v = a_v(j,0) * b_v(k,0);
	for(int i=1;i<szi;i++)
	  v += a_v(j,i) * b_v(k,i);
	out_v(j,k) = v;
      }
    });
  
  return out;  
}

//value: 139.23us 17.4579us
template<typename FloatType>
Matrix<FloatType> thinMulMatMatTranspose_v3(const Matrix<FloatType> &a, const Matrix<FloatType> &b){
  int szj = a.size(0);
  int szi = a.size(1);
  int szk = b.size(0);
  assert(b.size(1) == szi);
  
  Matrix<FloatType> out(szj,szk);
  autoView(out_v,out,DeviceWrite);
  autoView(a_v,a,DeviceRead);
  autoView(b_v,b,DeviceRead);

  int jblocksize = 8;
  int jblocks = (szj + jblocksize-1)/jblocksize;

  int kblocksize = 8;
  int kblocks = (szk + kblocksize-1)/kblocksize;

  int iblocksize = 8;
  int iblocks = (szi + iblocksize-1)/iblocksize;
  
  accelerator_for3d(jk,jblocksize*kblocksize, bk, kblocks,bj,jblocks,   1,{
      //jk = kk+kblocksize*jj
      int kk = jk % kblocksize;
      int jj = jk / kblocksize;

      int j = jj + jblocksize*bj;
      int k = kk + kblocksize*bk;
      
      if(j < szj && k < szk){
	FloatType v = 0.;
	for(int bi=0;bi<iblocks;bi++){
	  int istart = bi * iblocksize;
	  int ilessthan = istart + iblocksize;
	  if(ilessthan > szi) ilessthan = szi;
	  for(int i=istart;i<ilessthan;i++){	
	    v += a_v(j,i) * b_v(k,i);
	  }
	  acceleratorSynchronizeBlock();
	}
	out_v(j,k) = v;
      }
    });
  
  return out;  
}

//97.81us 16.3479us
template<typename FloatType>
Matrix<FloatType> thinMulMatMatTranspose_v4(const Matrix<FloatType> &a, const Matrix<FloatType> &b){
  int szj = a.size(0);
  int szi = a.size(1);
  int szk = b.size(0);
  assert(b.size(1) == szi);
  
  Matrix<FloatType> out(szj,szk);
  autoView(out_v,out,DeviceWrite);
  autoView(a_v,a,DeviceRead);
  autoView(b_v,b,DeviceRead);

  int jblocksize = 8;
  int jblocks = (szj + jblocksize-1)/jblocksize;

  int kblocksize = 8;
  int kblocks = (szk + kblocksize-1)/kblocksize;

  int iblocksize = 16;
  int iblocks = (szi + iblocksize-1)/iblocksize;

  assert(iblocksize % kblocksize == 0 && iblocksize/kblocksize == 2);
  
  accelerator_for3d_shm(jk,jblocksize*kblocksize, bk, kblocks,bj,jblocks,   1, (jblocksize + kblocksize) * iblocksize*sizeof(FloatType),  {
      extern __shared__ FloatType shared[];
      FloatType *abuf = shared;
      FloatType *bbuf = shared + jblocksize * iblocksize;
      
      //jk = kk+kblocksize*jj
      int kk = jk % kblocksize;
      int jj = jk / kblocksize;

      int j = jj + jblocksize*bj;
      int k = kk + kblocksize*bk;
      
      FloatType v = 0.0;
      for(int bi=0;bi<iblocks;bi++){
	int istart = bi * iblocksize;
	int ilessthan = istart + iblocksize;
	if(ilessthan > szi) ilessthan = szi;
	int iblocksize_actual = ilessthan - istart;

	if(j < szj){
	  int ii = kk;
	  int i = ii + istart;	    
	  abuf[ii + iblocksize*jj] = i < szi ? a_v(j,i) : 0.;

	  ii += kblocksize;
	  i = ii + istart;
	  abuf[ii + iblocksize*jj] = i < szi ? a_v(j,i) : 0.;
	}
	if(k < szk){
	  int ii = jj;
	  int i = ii + istart;	    
	  bbuf[ii + iblocksize*kk] = i < szi ? b_v(k,i) : 0.;

	  ii += jblocksize;
	  i = ii + istart;
	  bbuf[ii + iblocksize*kk] = i < szi ? b_v(k,i) : 0.;
	}
    
	acceleratorSynchronizeBlock();
	  
	for(int ii=0;ii<iblocksize_actual;ii++){	
	  v += abuf[ii + iblocksize*jj] * bbuf[ii + iblocksize*kk];
	}
	acceleratorSynchronizeBlock();
      }
	
      if(j < szj && k < szk) out_v(j,k) = v;
      
    });
  
  return out;  
}


template<typename FloatType>
Matrix<FloatType> thinMulMatMatTranspose_v5(const Matrix<FloatType> &a, const Matrix<FloatType> &b){
  int szj = a.size(0);
  int szi = a.size(1);
  int szk = b.size(0);
  assert(b.size(1) == szi);
  
  Matrix<FloatType> out(szj,szk);
  autoView(out_v,out,DeviceWrite);
  autoView(a_v,a,DeviceRead);
  autoView(b_v,b,DeviceRead);

  int jblocksize = 8;
  int jblocks = (szj + jblocksize-1)/jblocksize;

  int kblocksize = 8;
  int kblocks = (szk + kblocksize-1)/kblocksize;

  int iblocksize = 16;
  int iblocks = (szi + iblocksize-1)/iblocksize;

  assert(jblocksize == kblocksize);
  assert(iblocksize == 2*jblocksize);
  
  accelerator_for3d_shm(jk,jblocksize*kblocksize, bk, kblocks,bj,jblocks,   1, (jblocksize + kblocksize) * iblocksize*sizeof(FloatType),  {
      extern __shared__ FloatType shared[];
      FloatType *abuf = shared;
      FloatType *bbuf = shared + jblocksize * iblocksize;
      
      //jk = kk+kblocksize*jj
      int kk = jk % kblocksize;
      int jj = jk / kblocksize;

      int j = jj + jblocksize*bj;
      int k = kk + kblocksize*bk;      

      FloatType v = 0.0;
      for(int bi=0;bi<iblocks;bi++){
	int istart = bi * iblocksize;
	int ilessthan = istart + iblocksize;
	if(ilessthan > szi) ilessthan = szi;
	int iblocksize_actual = ilessthan - istart;
  
	if(j<szj){
	  int ii = kk;
	  int i = ii + istart;	    
	  abuf[ii + iblocksize*jj] = i < szi ? a_v(j,i) : 0.;

	  ii += kblocksize;
	  i = ii + istart;
	  abuf[ii + iblocksize*jj] = i < szi ? a_v(j,i) : 0.;
	}
	{
	  //efficient to swap mapping of jk -> jj, kj here such that jj is the fast moving index
	  int jj_tmp = jk % jblocksize;
	  int kk_tmp = jk / jblocksize;
	  int k_tmp = kk_tmp + kblocksize*bk;

	  if(k_tmp < szk){
	    int ii = jj_tmp;
	    int i = ii + istart;	    
	    bbuf[ii + iblocksize*kk_tmp] = i < szi ? b_v(k_tmp,i) : 0.;
	      
	    ii += jblocksize;
	    i = ii + istart;
	    bbuf[ii + iblocksize*kk_tmp] = i < szi ? b_v(k_tmp,i) : 0.;
	  }
	}
    
	acceleratorSynchronizeBlock();
	  
	for(int ii=0;ii<iblocksize_actual;ii++){	
	  v += abuf[ii + iblocksize*jj] * bbuf[ii + iblocksize*kk];
	}
	acceleratorSynchronizeBlock();
      }
      if(j < szj && k < szk) out_v(j,k) = v;
	 
    });
  
  return out;  
}

int main(int argc, char** argv){
  initialize(argc,argv);
  std::mt19937 rng(1234);
  
  std::vector<int> data_sizes = { 1, 30, 60, 120, 240, 480, 960, 1920 };
  std::vector<int> batch_sizes = {1, 8, 16, 32, 64};
  
  for(auto data_size : data_sizes){
    for(auto batch_size : batch_sizes){ 

      Matrix<float> a(data_size, batch_size);
      Matrix<float> b(data_size, batch_size);
      random(a,rng);
      random(b,rng);

      double mu, sigma;

      Matrix<float> c(data_size, batch_size);
      benchmark(mu, sigma, 100, 1, [&]{
	c = thinMulMatMatTranspose_v5(a,b);
      }, []{});

      Matrix<float> ctest = thinMulMatMatTransposeBase(a,b);
 
      std::cout << "neurons:" << data_size << "\tbatch: " << batch_size << "\tresult: " << mu/1e-6 << "us " << sigma/1e-6 << "us" << std::endl;
      assert(near(c,ctest,1e-4f,true));
    }
  }
      
  return 0;
}
