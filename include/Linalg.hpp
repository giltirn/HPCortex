#include <Tensors.hpp>
#include <Accelerator.hpp>

// C_jk = \sum_i  A_ji B_ki     for ni small-ish (batch size)
template<typename FloatType>
void thinMulMatMatTranspose_p(FloatType* out_p, const Matrix<FloatType> &a, const Matrix<FloatType> &b){
  int szj = a.size(0);
  int szi = a.size(1);
  int szk = b.size(0);
  assert(b.size(1) == szi);
  
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
      if(j < szj && k < szk) out_p[k+szk*j] = v;
	 
    });
}



template<typename FloatType>
Matrix<FloatType> thinMulMatMatTranspose(const Matrix<FloatType> &a, const Matrix<FloatType> &b){
  Matrix<FloatType> out(a.size(0),b.size(0));
  autoView(out_v,out,DeviceWrite);
  thinMulMatMatTranspose_p(out_v.data(),a,b);
  return out;  
}


