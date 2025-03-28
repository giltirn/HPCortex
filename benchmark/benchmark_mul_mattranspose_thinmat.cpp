#include<HPCortex.hpp>
#include<Testing.hpp>
#include<cuda_pipeline.h>

template<typename FloatType>
Matrix<FloatType> mulMatTransposeThinMatBase(const Matrix<FloatType> &a, const Matrix<FloatType> &b){
  int sizei = a.size(1);
  int sizej = a.size(0);
  int sizek = b.size(1);
  assert(b.size(0) == sizej);
  
  Matrix<FloatType> c(sizei,sizek);
  autoView(c_v,c,DeviceWrite);
  autoView(a_v,a,DeviceRead);
  autoView(b_v,b,DeviceRead);
  
  accelerator_for2d(k, sizek,i,sizei,1,{
      FloatType v = a_v(0,i) * b_v(0,k);
      for(int j=1;j<sizej;j++)
	v += a_v(j,i) * b_v(j,k);
      c_v(i,k) = v;
    });
  return c;
}

template<typename FloatType>
Matrix<FloatType> mulMatTransposeThinMat_v2(const Matrix<FloatType> &a, const Matrix<FloatType> &b){
  int sizei = a.size(1);
  int sizej = a.size(0);
  int sizek = b.size(1);
  assert(b.size(0) == sizej);
  
  Matrix<FloatType> c(sizei,sizek);
  autoView(c_v,c,DeviceWrite);
  autoView(a_v,a,DeviceRead);
  autoView(b_v,b,DeviceRead);

  int iblocksize = 8;
  int iblocks = (sizei + iblocksize - 1)/iblocksize;
  
  accelerator_for3d(k, sizek, ii, iblocksize, bi, iblocks,    iblocksize,{
      int i = ii + iblocksize * bi;
      if(i < sizei){      
	FloatType v = a_v(0,i) * b_v(0,k);
	for(int j=1;j<sizej;j++)
	  v += a_v(j,i) * b_v(j,k);
	c_v(i,k) = v;
      }
    });
  return c;
}

template<typename FloatType>
Matrix<FloatType> mulMatTransposeThinMat_v3(const Matrix<FloatType> &a, const Matrix<FloatType> &b){
  int sizei = a.size(1);
  int sizej = a.size(0);
  int sizek = b.size(1);
  assert(b.size(0) == sizej);
  
  Matrix<FloatType> c(sizei,sizek);
  autoView(c_v,c,DeviceWrite);
  autoView(a_v,a,DeviceRead);
  autoView(b_v,b,DeviceRead);

  int jblocksize = std::min(32,sizek);
  int jblocks = (sizej + jblocksize-1)/jblocksize;
  
  accelerator_for2d_shm(k, sizek,i,sizei,   1,  jblocksize*sizeof(FloatType),  {
      extern __shared__ FloatType shared[];
      FloatType v = 0.;
      for(int bj=0;bj<jblocks;bj++){	  
	int jstart = bj * jblocksize;
	int jlessthan = jstart + jblocksize;
	if(jlessthan > sizej) jlessthan = sizej;
	int jblocksize_actual = jlessthan - jstart;

	if(k < jblocksize_actual) //range of k is always >= jblocksize
	  shared[k] = a_v(jstart + k,i);
	acceleratorSynchronizeBlock();
	
	for(int j=jstart;j<jlessthan;j++){	
	  v += shared[j-jstart] * b_v(j,k);
	}
	acceleratorSynchronizeBlock();
      }
      c_v(i,k) = v;	
    });

  return c;
}


template<typename FloatType>
Matrix<FloatType> mulMatTransposeThinMat_v4(const Matrix<FloatType> &a, const Matrix<FloatType> &b){
  int sizei = a.size(1);
  int sizej = a.size(0);
  int sizek = b.size(1);
  assert(b.size(0) == sizej);
  
  Matrix<FloatType> c(sizei,sizek);
  autoView(c_v,c,DeviceWrite);
  autoView(a_v,a,DeviceRead);
  autoView(b_v,b,DeviceRead);

  int jblocksize = 32;
  
  if(sizek < jblocksize){  
    int iblocksize = 8;
    int iblocks = (sizei + iblocksize - 1)/iblocksize;
  
    accelerator_for3d(k, sizek, ii, iblocksize, bi, iblocks,    iblocksize,{
	int i = ii + iblocksize * bi;
	if(i < sizei){      
	  FloatType v = a_v(0,i) * b_v(0,k);
	  for(int j=1;j<sizej;j++)
	    v += a_v(j,i) * b_v(j,k);
	  c_v(i,k) = v;
	}
      });
  }else{
    int jblocks = (sizej + jblocksize-1)/jblocksize;
  
    accelerator_for2d_shm(k, sizek,i,sizei,   1,  jblocksize*sizeof(FloatType),  {
	extern __shared__ FloatType shared[];
	FloatType v = 0.;
	for(int bj=0;bj<jblocks;bj++){	  
	  int jstart = bj * jblocksize;
	  int jlessthan = jstart + jblocksize;
	  if(jlessthan > sizej) jlessthan = sizej;
	  int jblocksize_actual = jlessthan - jstart;

	  if(k < jblocksize_actual) //range of k is always >= jblocksize
	    shared[k] = a_v(jstart + k,i);
	  acceleratorSynchronizeBlock();
	
	  for(int j=jstart;j<jlessthan;j++){	
	    v += shared[j-jstart] * b_v(j,k);
	  }
	  acceleratorSynchronizeBlock();
	}
	c_v(i,k) = v;	
      });
  }

  return c;
}


template<typename FloatType>
Matrix<FloatType> mulMatTransposeThinMat_v5(const Matrix<FloatType> &a, const Matrix<FloatType> &b){
  int sizei = a.size(1);
  int sizej = a.size(0);
  int sizek = b.size(1);
  assert(b.size(0) == sizej);
  
  Matrix<FloatType> c(sizei,sizek);
  autoView(c_v,c,DeviceWrite);
  autoView(a_v,a,DeviceRead);
  autoView(b_v,b,DeviceRead);
  
  accelerator_for2d(k, sizek,i,sizei, 16,{
      FloatType v = a_v(0,i) * b_v(0,k);
      for(int j=1;j<sizej;j++)
	v += a_v(j,i) * b_v(j,k);
      c_v(i,k) = v;
    });
  return c;
}

template<typename FloatType>
Matrix<FloatType> mulMatTransposeThinMat_v6(const Matrix<FloatType> &a, const Matrix<FloatType> &b){
  int sizei = a.size(1);
  int sizej = a.size(0);
  int sizek = b.size(1);
  assert(b.size(0) == sizej);
  
  Matrix<FloatType> c(sizei,sizek);
  autoView(c_v,c,DeviceWrite);
  autoView(a_v,a,DeviceRead);
  autoView(b_v,b,DeviceRead);
  
  accelerator_for2d_shm(k, sizek,i,sizei, 1,   sizej*sizeof(FloatType), {
      extern __shared__ FloatType shared[];
      for(int j=k;j<sizej;j+=sizek){
	shared[j] = a_v(j,i);
      }
      acceleratorSynchronizeBlock();
      
      FloatType v = shared[0] * b_v(0,k);
      for(int j=1;j<sizej;j++)
	v += shared[j] * b_v(j,k);
      c_v(i,k) = v;

      acceleratorSynchronizeBlock();
    });
  return c;
}


template<typename FloatType>
Matrix<FloatType> mulMatTransposeThinMat_v7(const Matrix<FloatType> &a, const Matrix<FloatType> &b){
  int sizei = a.size(1);
  int sizej = a.size(0);
  int sizek = b.size(1);
  assert(b.size(0) == sizej);
  
  Matrix<FloatType> c(sizei,sizek);
  autoView(c_v,c,DeviceWrite);
  autoView(a_v,a,DeviceRead);
  autoView(b_v,b,DeviceRead);

  int jblocksize = 64;
  
  accelerator_for2d_shm(k, sizek,i,sizei, 1,   jblocksize*sizeof(FloatType), {
      extern __shared__ FloatType shared[];
      FloatType v = 0.;
      for(int jbase=0; jbase < sizej; jbase += jblocksize){
	int rem = sizej - jbase;
	int jblocksize_actual = rem < jblocksize ? rem : jblocksize;
	
	for(int jj=k;jj<jblocksize_actual;jj+=sizek){
	  shared[jj] = a_v(jj + jbase,i);
	}
	acceleratorSynchronizeBlock();
	
	for(int jj=0;jj<jblocksize_actual;jj++)
	  v += shared[jj] * b_v(jj+jbase,k);

	acceleratorSynchronizeBlock();
      }
      c_v(i,k) = v;
    });
  return c;
}



template<typename FloatType>
Matrix<FloatType> mulMatTransposeThinMat_v8(const Matrix<FloatType> &a, const Matrix<FloatType> &b){
  int sizei = a.size(1);
  int sizej = a.size(0);
  int sizek = b.size(1);
  assert(b.size(0) == sizej);
  
  Matrix<FloatType> c(sizei,sizek);
  autoView(c_v,c,DeviceWrite);
  autoView(a_v,a,DeviceRead);
  autoView(b_v,b,DeviceRead);

  int jblocksize = 64;
  int blocki = 16;
  
  accelerator_for2d_shm(k, sizek,i,sizei, blocki,   blocki*jblocksize*sizeof(FloatType), {
      extern __shared__ FloatType shared[];
      FloatType *shared_p = shared + threadIdx.y*jblocksize;
      
      FloatType v = 0.;
      for(int jbase=0; jbase < sizej; jbase += jblocksize){
	int rem = sizej - jbase;
	int jblocksize_actual = rem < jblocksize ? rem : jblocksize;
	
	for(int jj=k;jj<jblocksize_actual;jj+=sizek){
	  shared_p[jj] = a_v(jj + jbase,i);
	}
	acceleratorSynchronizeBlock();
	
	for(int jj=0;jj<jblocksize_actual;jj++)
	  v += shared_p[jj] * b_v(jj+jbase,k);

	acceleratorSynchronizeBlock();
      }
      c_v(i,k) = v;
    });
  return c;
}


template<typename FloatType>
Matrix<FloatType> mulMatTransposeThinMat_v9(const Matrix<FloatType> &a, const Matrix<FloatType> &b){
  int sizei = a.size(1);
  int sizej = a.size(0);
  int sizek = b.size(1);
  assert(b.size(0) == sizej);
  
  Matrix<FloatType> c(sizei,sizek);
  autoView(c_v,c,DeviceWrite);
  autoView(a_v,a,DeviceRead);
  autoView(b_v,b,DeviceRead);

  int jblocksize = 64;
  
  accelerator_for2d_shm(k, sizek,i,sizei, 1,   2*jblocksize*sizeof(FloatType), {
      extern __shared__ FloatType shared[];
      FloatType *shared_current = shared;
      FloatType *shared_prefetch = shared + jblocksize;

      { //prefetch 1st iter
	int jbase = 0;
	int rem = sizej - jbase;
	int jblocksize_actual = rem < jblocksize ? rem : jblocksize;
	for(int jj=k;jj<jblocksize_actual;jj+=sizek){
	  __pipeline_memcpy_async(shared_prefetch + jj,  &a_v(jj + jbase,i), sizeof(FloatType));
	}
	__pipeline_commit();
      }
      
      FloatType v = 0.;
      for(int jbase=0; jbase < sizej; jbase += jblocksize){
	int rem = sizej - jbase;
	int jblocksize_actual = rem < jblocksize ? rem : jblocksize;

	__pipeline_wait_prior(1);
	
	FloatType* tmp = shared_current;
	shared_current = shared_prefetch;
	shared_prefetch = tmp;
	
	{ //prefetch next iter
	  int jbasep = jbase + jblocksize;
	  int rem = sizej - jbasep;
	  int jblocksize_actualp = rem < jblocksize ? rem : jblocksize;
	  for(int jj=k;jj<jblocksize_actualp;jj+=sizek){
	    __pipeline_memcpy_async(shared_prefetch + jj,  &a_v(jj + jbasep,i), sizeof(FloatType));
	  }
	  __pipeline_commit();
	}
	
	acceleratorSynchronizeBlock();
	
	for(int jj=0;jj<jblocksize_actual;jj++)
	  v += shared_current[jj] * b_v(jj+jbase,k);

	acceleratorSynchronizeBlock();
      }
      c_v(i,k) = v;
    });
  return c;
}

//390us
template<typename FloatType>
Matrix<FloatType> mulMatTransposeThinMat_v10(const Matrix<FloatType> &a, const Matrix<FloatType> &b){
  int sizei = a.size(1);
  int sizej = a.size(0);
  int sizek = b.size(1);
  assert(b.size(0) == sizej);
  
  Matrix<FloatType> c(sizei,sizek);
  autoView(c_v,c,DeviceWrite);
  autoView(a_v,a,DeviceRead);
  autoView(b_v,b,DeviceRead);

  int jblocksize = 64;

  constexpr int prefetch_depth = 4;
  constexpr int nbuf = prefetch_depth+1;
  
  accelerator_for2d_shm(k, sizek,i,sizei, 1,   nbuf*jblocksize*sizeof(FloatType), {
      extern __shared__ FloatType shared[];
      
      FloatType* shared_p[nbuf];
      for(int p=0;p<nbuf;p++)
	shared_p[p] = shared + jblocksize * p;

      for(int p=0;p<prefetch_depth;p++){
	int jbase = p * jblocksize;
	int rem = sizej - jbase;
	int jblocksize_actual = rem < jblocksize ? rem : jblocksize;
	for(int jj=k;jj<jblocksize_actual;jj+=sizek){
	  __pipeline_memcpy_async(shared_p[p] + jj,  &a_v(jj + jbase,i), sizeof(FloatType));
	}
	__pipeline_commit();
      }
      int bufidx_use = 0;
      int bufidx_pref = prefetch_depth;
      
      FloatType v = 0.;
      for(int jbase=0; jbase < sizej; jbase += jblocksize){
	int rem = sizej - jbase;
	int jblocksize_actual = rem < jblocksize ? rem : jblocksize;

	__pipeline_wait_prior(prefetch_depth);	
	
	if(rem > jblocksize){ //prefetch next iter
	  FloatType* shared_into = shared_p[bufidx_pref];
	  int jbasep = jbase + prefetch_depth * jblocksize;
	  int rem = sizej - jbasep;
	  int jblocksize_actualp = rem < jblocksize ? rem : jblocksize;
	  for(int jj=k;jj<jblocksize_actualp;jj+=sizek){
	    __pipeline_memcpy_async(shared_into + jj,  &a_v(jj + jbasep,i), sizeof(FloatType));
	  }
	  __pipeline_commit();
	}       

	acceleratorSynchronizeBlock();
	
	FloatType* shared_use = shared_p[bufidx_use];
	for(int jj=0;jj<jblocksize_actual;jj++)
	  v += shared_use[jj] * b_v(jj+jbase,k);

	bufidx_use = (bufidx_use + 1) % nbuf;
	bufidx_pref = (bufidx_pref + 1) % nbuf;

	acceleratorSynchronizeBlock();
      }
      c_v(i,k) = v;
    });
  return c;
}


//C_ik = \sum_j A_ji B_jk for nk small-ish (batch size)
//240us
template<typename FloatType>
Matrix<FloatType> mulMatTransposeThinMat_v11(const Matrix<FloatType> &a, const Matrix<FloatType> &b){
  int sizei = a.size(1);
  int sizej = a.size(0);
  int sizek = b.size(1);
  assert(b.size(0) == sizej);
  
  Matrix<FloatType> c(sizei,sizek,0.);
  autoView(c_v,c,DeviceReadWrite);
  autoView(a_v,a,DeviceRead);
  autoView(b_v,b,DeviceRead);

  constexpr int iblocksize = 8;
  constexpr int jblocksize = 16;

  int niblocks = (sizei + iblocksize  - 1) / iblocksize;
  int njblocks = (sizej + jblocksize  - 1) / jblocksize;  

  int kthr = std::max(sizek,jblocksize);

 
  accelerator_for3d_shm(kk, kthr, bj, njblocks, bi,niblocks, 1,   iblocksize*jblocksize*sizeof(FloatType), {
      extern __shared__ FloatType shared[];
      int ibase = iblocksize * bi;
      int irem = sizei - ibase;
      int icount = irem < iblocksize ? irem : iblocksize;
            
      int jbase = jblocksize * bj;
      int jrem = sizej - jbase;
      int jcount = jrem < jblocksize ? jrem : jblocksize;

      if(kk < jcount){
	int jj = kk;
	int j = jbase + jj;

	for(int ii=0;ii<icount;ii++)
	  shared[jj + jblocksize*ii] = a_v(j,ii + ibase);
      }
	
      // if(kk==0){
      // 	for(int ii=0;ii<icount;ii++)
      // 	  for(int jj=0;jj<jcount;jj++)
      // 	    shared[jj + jblocksize*ii] = a_v(jj+jbase,ii + ibase);
      // }	
      
      acceleratorSynchronizeBlock();

      if(kk < sizek){
	for(int ii=0;ii<icount;ii++){
	  FloatType v = 0.;
	  for(int jj=0;jj<jcount;jj++){
	    //v += a_v(jj+jbase, ii+ibase) * b_v(jj+jbase,kk);
	    v += shared[jj + jblocksize*ii] * b_v(jj+jbase,kk);
	  }  
	  atomicAdd(&c_v(ii+ibase,kk), v);
	}
      }
    });
  return c;
}


//220us
template<typename FloatType>
Matrix<FloatType> mulMatTransposeThinMat_v12(const Matrix<FloatType> &a, const Matrix<FloatType> &b){
  int sizei = a.size(1);
  int sizej = a.size(0);
  int sizek = b.size(1);
  assert(b.size(0) == sizej);
  
  Matrix<FloatType> c(sizei,sizek,0.);
  autoView(c_v,c,DeviceReadWrite);
  autoView(a_v,a,DeviceRead);
  autoView(b_v,b,DeviceRead);

  constexpr int iblocksize = 16;
  constexpr int jblocksize = 16;

  int niblocks = (sizei + iblocksize  - 1) / iblocksize;
  int njblocks = (sizej + jblocksize  - 1) / jblocksize;  

  int kthr = std::max(sizek,iblocksize);

 
  accelerator_for3d_shm(kk, kthr, bj, njblocks, bi,niblocks, 1,   iblocksize*jblocksize*sizeof(FloatType), {
      extern __shared__ FloatType shared[];
      int ibase = iblocksize * bi;
      int irem = sizei - ibase;
      int icount = irem < iblocksize ? irem : iblocksize;
            
      int jbase = jblocksize * bj;
      int jrem = sizej - jbase;
      int jcount = jrem < jblocksize ? jrem : jblocksize;

      if(kk < icount){
	int ii = kk;
	int i = ibase + ii;

	for(int jj=0;jj<jcount;jj++)
	  shared[jj + jblocksize*ii] = a_v(jj+jbase, i);	
      }
	     
      acceleratorSynchronizeBlock();

      if(kk < sizek){
	for(int ii=0;ii<icount;ii++){
	  FloatType v = 0.;
	  for(int jj=0;jj<jcount;jj++){
	    v += shared[jj + jblocksize*ii] * b_v(jj+jbase,kk);
	  }  
	  atomicAdd(&c_v(ii+ibase,kk), v);
	}
      }
    });
  return c;
}


//336us
template<typename FloatType>
Matrix<FloatType> mulMatTransposeThinMat_v13(const Matrix<FloatType> &a, const Matrix<FloatType> &b){
  int sizei = a.size(1);
  int sizej = a.size(0);
  int sizek = b.size(1);
  assert(b.size(0) == sizej);
  
  Matrix<FloatType> c(sizei,sizek,0.);
  autoView(c_v,c,DeviceReadWrite);
  autoView(a_v,a,DeviceRead);
  autoView(b_v,b,DeviceRead);

  constexpr int iblocksize = 16;
  constexpr int jblocksize = 16;

  int niblocks = (sizei + iblocksize  - 1) / iblocksize;
  int njblocks = (sizej + jblocksize  - 1) / jblocksize;  

  int kthr = std::max(sizek,iblocksize);

 
  accelerator_for3d_shm(kk, kthr, bj, njblocks, bi,niblocks, 1,   iblocksize*(jblocksize+1)*sizeof(FloatType), {
      extern __shared__ FloatType shared[];
      int ibase = iblocksize * bi;
      int irem = sizei - ibase;
      int icount = irem < iblocksize ? irem : iblocksize;
            
      int jbase = jblocksize * bj;
      int jrem = sizej - jbase;
      int jcount = jrem < jblocksize ? jrem : jblocksize;

      int istride = jblocksize+1; //trying to reduce bank conflict but made performance worse somehow!
      
      if(kk < icount){
	int ii = kk;
	int i = ibase + ii;

	for(int jj=0;jj<jcount;jj++)
	  shared[jj + istride*ii] = a_v(jj+jbase, i);
	
	// for(int jj=0;jj<jcount;jj++)
	//   __pipeline_memcpy_async(shared + jj + jblocksize*ii,  &a_v(jj + jbase,i), sizeof(FloatType));

	// __pipeline_commit();
	// __pipeline_wait_prior(0);
      }
	     
      acceleratorSynchronizeBlock();

      if(kk < sizek){
	for(int ii=0;ii<icount;ii++){
	  FloatType v = 0.;
	  for(int jj=0;jj<jcount;jj++){
	    v += shared[jj + istride*ii] * b_v(jj+jbase,kk);
	  }  
	  atomicAdd(&c_v(ii+ibase,kk), v);
	}
      }
    });
  return c;
}


//180us
template<typename FloatType>
Matrix<FloatType> mulMatTransposeThinMat_v14(const Matrix<FloatType> &a, const Matrix<FloatType> &b){
  int sizei = a.size(1);
  int sizej = a.size(0);
  int sizek = b.size(1);
  assert(b.size(0) == sizej);
  
  Matrix<FloatType> c(sizei,sizek,0.);
  autoView(c_v,c,DeviceReadWrite);
  autoView(a_v,a,DeviceRead);
  autoView(b_v,b,DeviceRead);

  constexpr int iblocksize = 32;
  constexpr int jblocksize = 16;
  constexpr int kblocksize = 64;
 
  int niblocks = (sizei + iblocksize  - 1) / iblocksize;
  int njblocks = (sizej + jblocksize  - 1) / jblocksize;  
  int nkblocks = (sizek + kblocksize  - 1) / kblocksize;
  
  int kthr = std::max(kblocksize,iblocksize);

 
  accelerator_for3d_shm(kk, kthr, bjk, nkblocks*njblocks, bi,niblocks, 1,   (iblocksize*jblocksize + jblocksize*kblocksize)*sizeof(FloatType), {
      int bk = bjk % nkblocks;
      int bj = bjk / nkblocks;
      
      extern __shared__ FloatType shared[];
      FloatType* shared_a = shared;
      FloatType* shared_b = shared + iblocksize*jblocksize;
      
      int ibase = iblocksize * bi;
      int irem = sizei - ibase;
      int icount = irem < iblocksize ? irem : iblocksize;
            
      int jbase = jblocksize * bj;
      int jrem = sizej - jbase;
      int jcount = jrem < jblocksize ? jrem : jblocksize;

      int kbase = kblocksize * bk;
      int krem = sizek - kbase;
      int kcount = krem < kblocksize ? krem : kblocksize;
      
      
      if(kk < icount){
	int ii = kk;
	int i = ibase + ii;

	for(int jj=0;jj<jcount;jj++)
	  shared_a[jj + jblocksize*ii] = a_v(jj+jbase, i);	
      }
      if(kk < kcount){	
	for(int jj=0;jj<jcount;jj++)
	  shared_b[kk + kblocksize * jj] = b_v(jj+jbase,kk + kbase);
      }
      
      acceleratorSynchronizeBlock();

      if(kk < kcount){	
	for(int ii=0;ii<icount;ii++){
	  FloatType v = 0.;
	  for(int jj=0;jj<jcount;jj++){
	    v += shared_a[jj + jblocksize*ii] * shared_b[kk + kblocksize * jj];
	  }  
	  atomicAdd(&c_v(ii+ibase,kk + kbase), v);
	}		
      }
    });
  return c;
}



//combo of v14 and base for optimal sizes
template<typename FloatType>
Matrix<FloatType> mulMatTransposeThinMat_v15(const Matrix<FloatType> &a, const Matrix<FloatType> &b){
  int sizei = a.size(1);
  int sizej = a.size(0);
  int sizek = b.size(1);
  assert(b.size(0) == sizej);

  if(sizej < 300){
    Matrix<FloatType> c(sizei,sizek);
    autoView(c_v,c,DeviceWrite);
    autoView(a_v,a,DeviceRead);
    autoView(b_v,b,DeviceRead);
    
    accelerator_for2d(k, sizek,i,sizei,1,{
	FloatType v = a_v(0,i) * b_v(0,k);
	for(int j=1;j<sizej;j++)
	  v += a_v(j,i) * b_v(j,k);
	c_v(i,k) = v;
      });

    return c;
  }else{

    Matrix<FloatType> c(sizei,sizek,0.);
    autoView(c_v,c,DeviceReadWrite);
    autoView(a_v,a,DeviceRead);
    autoView(b_v,b,DeviceRead);
    
    constexpr int iblocksize = 32;
    constexpr int jblocksize = 16;
    constexpr int kblocksize = 64;
 
    int niblocks = (sizei + iblocksize  - 1) / iblocksize;
    int njblocks = (sizej + jblocksize  - 1) / jblocksize;  
    int nkblocks = (sizek + kblocksize  - 1) / kblocksize;
  
    int kthr = std::max(kblocksize,iblocksize);

 
    accelerator_for3d_shm(kk, kthr, bjk, nkblocks*njblocks, bi,niblocks, 1,   (iblocksize*jblocksize + jblocksize*kblocksize)*sizeof(FloatType), {
	int bk = bjk % nkblocks;
	int bj = bjk / nkblocks;
      
	extern __shared__ FloatType shared[];
	FloatType* shared_a = shared;
	FloatType* shared_b = shared + iblocksize*jblocksize;
      
	int ibase = iblocksize * bi;
	int irem = sizei - ibase;
	int icount = irem < iblocksize ? irem : iblocksize;
            
	int jbase = jblocksize * bj;
	int jrem = sizej - jbase;
	int jcount = jrem < jblocksize ? jrem : jblocksize;

	int kbase = kblocksize * bk;
	int krem = sizek - kbase;
	int kcount = krem < kblocksize ? krem : kblocksize;
      
      
	if(kk < icount){
	  int ii = kk;
	  int i = ibase + ii;

	  for(int jj=0;jj<jcount;jj++)
	    shared_a[jj + jblocksize*ii] = a_v(jj+jbase, i);	
	}
	if(kk < kcount){	
	  for(int jj=0;jj<jcount;jj++)
	    shared_b[kk + kblocksize * jj] = b_v(jj+jbase,kk + kbase);
	}
      
	acceleratorSynchronizeBlock();

	if(kk < kcount){	
	  for(int ii=0;ii<icount;ii++){
	    FloatType v = shared_a[jblocksize*ii] * shared_b[kk];
	    for(int jj=1;jj<jcount;jj++){
	      v += shared_a[jj + jblocksize*ii] * shared_b[kk + kblocksize * jj];
	    }  
	    atomicAdd(&c_v(ii+ibase,kk + kbase), v);
	  }		
	}
      });
    return c;
  }
  
  
}





int main(int argc, char** argv){
  initialize(argc,argv);
  std::mt19937 rng(1234);
  
  std::vector<int> data_sizes = { 1, 30, 60, 120, 240, 480, 960, 1920 };
  std::vector<int> batch_sizes = {1, 8, 16, 32, 64};

  //std::vector<int> data_sizes = {1920};
  // std::vector<int> batch_sizes = {64};
  
  for(auto size0 : data_sizes){
    for(auto size1 : data_sizes){    
      for(auto batch_size : batch_sizes){ 

	//C_ik = \sum_j A_ji B_jk for nk small-ish (batch size)
	Matrix<float> a(size0, size1);
	Matrix<float> b(size0, batch_size);
	random(a,rng);
	random(b,rng);

	Matrix<float> c = mulMatTransposeThinMat_v15(a,b);
	Matrix<float> ctest = mulMatTransposeThinMatBase(a,b);
	assert(abs_near(c,ctest,1e-3f,true));
      
	double mu, sigma;

	profileStart();
	benchmark(mu, sigma, 100, 1, [&]{
	  c = mulMatTransposeThinMat_v15(a,b);
	  //c = mulMatTransposeThinMatBase(a,b);
	}, []{});
	profileStop();
   
	std::cout << "arows:" << size0 << " acols:" << size1 << "\tbatch: " << batch_size << "\tresult: " << mu/1e-6 << "us " << sigma/1e-6 << "us" << std::endl;
      }      
    }
  }

      
  return 0;
}
