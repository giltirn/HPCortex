#pragma once
#include <Tensors.hpp>
#include <Accelerator.hpp>

// C_jk = \sum_i  A_ji B_ki     for ni small-ish (batch size)  pointer version
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


//C_jk = \sum_i  A_ji B_ki     for ni small-ish (batch size)
template<typename FloatType>
Matrix<FloatType> thinMulMatMatTranspose(const Matrix<FloatType> &a, const Matrix<FloatType> &b){
  Matrix<FloatType> out(a.size(0),b.size(0));
  autoView(out_v,out,DeviceWrite);
  thinMulMatMatTranspose_p(out_v.data(),a,b);
  return out;  
}


//C_ik = \sum_j A_ji B_jk for nk small-ish (batch size)
template<typename FloatType>
Matrix<FloatType> mulMatTransposeThinMat(const Matrix<FloatType> &a, const Matrix<FloatType> &b){
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


//out(i, b) = above_deriv(i,b) * activation_deriv(i,b)
template<typename FloatType>
Matrix<FloatType> computeThinMatOuterProd(const Matrix<FloatType> &above_deriv, const Matrix<FloatType> &activation_deriv){
  int size0 = above_deriv.size(0);
  int batch_size =  above_deriv.size(1);
  assert(activation_deriv.size(0) == size0 && activation_deriv.size(1) == batch_size);
  
  Matrix<FloatType> activated_above_deriv(size0,batch_size);
  autoView(above_deriv_v,above_deriv,DeviceRead);
  autoView(activation_deriv_v,activation_deriv,DeviceRead);
  autoView(activated_above_deriv_v,activated_above_deriv,DeviceWrite);

  int bblocksize = std::min(128,batch_size);
  int nbblocks = (batch_size + bblocksize - 1) / bblocksize; 
  
  accelerator_for3d(bb,bblocksize,bblock,nbblocks,i,size0,1,{      
      int b = bb + bblocksize*bblock;
      if(b < batch_size){      
	activated_above_deriv_v(i,b) = above_deriv_v(i,b) * activation_deriv_v(i,b);
      }
    });
  return activated_above_deriv;
}


//matrix a * b + c with b having a modest number of columns
template<typename FloatType>
Matrix<FloatType> axpyMatThinMat(const Matrix<FloatType> &a, const Matrix<FloatType> &b, const Vector<FloatType> &c){
  int size0 = a.size(0);
  assert(c.size(0) == size0);
  int size1 = a.size(1);
  assert(b.size(0) == size1);
  int size2 = b.size(1);
    
  Matrix<FloatType> out(size0,size2);
  {
    autoView(c_v,c,DeviceRead);
    autoView(out_v,out,DeviceWrite);
    autoView(b_v,b,DeviceRead);
    autoView(a_v,a,DeviceRead);

    //Basic version where columns are summed over within a thread and rows/batches distributed over threads
    accelerator_for2d(k,size2,i,size0,1,{
	FloatType v = c_v(i);
	for(int j=0;j<size1;j++)
	  v += a_v(i,j)* b_v(j,k);
	out_v(i,k) = v;	  
      });      
  }
  return out;
}


//Contract batched 3-tensors (those for whom the last dimension is the batch index) over some dimension \in [0,1]
//eg  C_{ijb} = \sum_k A_{kib} B_{jkb} * nrm

template<typename FloatType>
Tensor<FloatType,3> batch3tensorContract(const Tensor<FloatType,3> &A, const Tensor<FloatType,3> &B, int contract_dimA, int contract_dimB, FloatType nrm = 1.0){
  assert(A.size(2) == B.size(2));
  size_t batch_size = A.size(2);

  assert(contract_dimA == 0 || contract_dimA == 1);
  assert(contract_dimB == 0 || contract_dimB == 1);
  assert(A.size(contract_dimA) == B.size(contract_dimB));
  
  int sizes_out[3];
  sizes_out[0] = contract_dimA == 0 ? A.size(1) : A.size(0);
  sizes_out[1] = contract_dimB == 0 ? B.size(1) : B.size(0);
  sizes_out[2] = batch_size;
  Tensor<FloatType,3> out(sizes_out);

  //Let a be the other index for A, and b for B
  int sizek = A.size(contract_dimA);

  size_t Astride[2] = { A.size(1)*batch_size, batch_size };  //(i,j,b) -> b + batch_size*(j + sizej * i)
  size_t Bstride[2] = { B.size(1)*batch_size, batch_size };
  
  size_t kstrideA = Astride[contract_dimA];
  size_t astride = Astride[1-contract_dimA];
  
  size_t kstrideB = Bstride[contract_dimB];
  size_t bstride = Bstride[1-contract_dimB];

  {
    autoView(out_v,out,DeviceWrite);
    autoView(A_v,A,DeviceRead);
    autoView(B_v,B,DeviceRead);
    
    accelerator_for3d(batch, batch_size, a, sizes_out[0], b, sizes_out[1],   1, {
	FloatType* A_p = A_v.data() + batch + a*astride;
	FloatType* B_p = B_v.data() + batch + b*bstride;
	FloatType res = (*A_p) * (*B_p);
	A_p += kstrideA;
	B_p += kstrideB;
	for(int k=1;k<sizek;k++){
	  res += (*A_p) * (*B_p);
	  A_p += kstrideA;
	  B_p += kstrideB;
	}
	out_v(a,b,batch) = res * nrm;
      });
  }
  return out;
}

  
  
  
