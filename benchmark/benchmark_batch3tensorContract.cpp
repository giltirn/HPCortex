#include<HPCortex.hpp>
#include<Testing.hpp>

#ifdef USE_GPU

//eg  C_{ijb} = \sum_k A_{kib} B_{jkb} * nrm

template<typename FloatType>
Tensor<FloatType,3> batch3tensorContractBase(const Tensor<FloatType,3> &A, const Tensor<FloatType,3> &B, int contract_dimA, int contract_dimB, FloatType nrm){
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

//about as fast as base
template<typename FloatType>
Tensor<FloatType,3> batch3tensorContract_v2(const Tensor<FloatType,3> &A, const Tensor<FloatType,3> &B, int contract_dimA, int contract_dimB, FloatType nrm){
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

  constexpr int ablocksz = 2;
  int asize = sizes_out[0];
  int ablocks = (asize + ablocksz - 1)/ablocksz;

  constexpr int bblocksz = 2;
  int bsize = sizes_out[1];
  int bblocks = (bsize + bblocksz - 1)/bblocksz;

  constexpr int kblocksz = 2;
  int kblocks = (sizek + kblocksz - 1)/kblocksz;

  constexpr int batchblocksz = 32;
  int batchblocks = (batch_size + batchblocksz-1)/batchblocksz;
  
  {
    autoView(out_v,out,DeviceWrite);
    autoView(A_v,A,DeviceRead);
    autoView(B_v,B,DeviceRead);
    constexpr size_t shm_size = ablocksz*kblocksz*batchblocksz*sizeof(FloatType)
      + bblocksz*kblocksz*batchblocksz*sizeof(FloatType)
      + ablocksz*bblocksz*batchblocksz*sizeof(FloatType)
      + 3*batchblocksz*sizeof(FloatType);
    constexpr size_t bufoffb = ablocksz*kblocksz*batchblocksz*sizeof(FloatType) + batchblocksz*sizeof(FloatType);
    constexpr size_t bufoffab = bufoffb + bblocksz*kblocksz*batchblocksz*sizeof(FloatType) + batchblocksz*sizeof(FloatType);
    
    //akblocks:
    //0..ak-1|ak..2ak-1|2ak..3ak-1|
    //0..ak-1|*ak..2ak-1|**2ak..3ak-1|
    
    accelerator_for_1_3_shm(bbatch, batchblocksz, batchblock, batchblocks, blocka, ablocks, blockb, bblocks, 1, shm_size,  {
	FloatType* abuf = (FloatType*)(shared  +            bbatch * kblocksz*ablocksz*sizeof(FloatType) + bbatch*sizeof(FloatType) );  
	FloatType* bbuf = (FloatType*)(shared  + bufoffb  + bbatch * kblocksz*bblocksz*sizeof(FloatType) + bbatch*sizeof(FloatType) );
	FloatType* abbuf = (FloatType*)(shared + bufoffab + bbatch*ablocksz*bblocksz*sizeof(FloatType) + bbatch*sizeof(FloatType) );
	
	int batch = bbatch + batchblocksz*batchblock;
	if(batch < batch_size){
	  FloatType* abbufp = abbuf;
	  for(int aa=0;aa<ablocksz;aa++)
	    for(int bb=0;bb<bblocksz;bb++)
	      *(abbufp++) = 0.;

	  for(int blockk=0;blockk<kblocks;blockk++){    
	    FloatType* abufp = abuf;
	    int a = blocka*ablocksz;
	    for(int aa=0;aa<ablocksz;aa++){
	      int k = blockk*kblocksz;
	      for(int kk=0;kk<kblocksz;kk++){
		*(abufp++) = a < asize && k < sizek ? *(A_v.data() + batch + a*astride + k*kstrideA) : 0.;
		++k;
	      }
	      ++a;
	    }
	    FloatType* bbufp = bbuf;
	    int b = blockb*bblocksz;
	    for(int bb=0;bb<bblocksz;bb++){
	      int k = blockk*kblocksz;
	      for(int kk=0;kk<kblocksz;kk++){
		*(bbufp++) = b < bsize && k < sizek ? *(B_v.data() + batch + b*bstride + k*kstrideB) : 0.;
		++k;
	      }
	      ++b;
	    }

	    for(int aa=0;aa<ablocksz;aa++){
	      for(int bb=0;bb<bblocksz;bb++){
		abufp = abuf + aa*kblocksz;
		bbufp = bbuf + bb*kblocksz;
		  
		FloatType res = (*abufp++) * (*bbufp++);
		for(int kk=1;kk<kblocksz;kk++)
		  res += (*abufp++) * (*bbufp++);
		  
		abbuf[bb + bblocksz*aa] += res*nrm;
	      }
	    }	    
	  }

	  for(int aa=0;aa<ablocksz;aa++){
	    int a = blocka*ablocksz + aa;
	    if(a < asize){
	      for(int bb=0;bb<bblocksz;bb++){
		int b = blockb*bblocksz + bb;
		if(b < bsize)
		  out_v(a,b,batch) = abbuf[bb + bblocksz*aa];		
	      }
	    }
	  }

	  
	}
	  
      });
  }
  return out;
}

//slower
template<typename FloatType>
Tensor<FloatType,3> batch3tensorContract_v3(const Tensor<FloatType,3> &A, const Tensor<FloatType,3> &B, int contract_dimA, int contract_dimB, FloatType nrm){
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

  constexpr int ablocksz = 2;
  int asize = sizes_out[0];
  int ablocks = (asize + ablocksz - 1)/ablocksz;

  constexpr int bblocksz = 2;
  int bsize = sizes_out[1];
  int bblocks = (bsize + bblocksz - 1)/bblocksz;

  constexpr int kblocksz = 2;
  int kblocks = (sizek + kblocksz - 1)/kblocksz;

  constexpr int batchblocksz = 32;
  int batchblocks = (batch_size + batchblocksz-1)/batchblocksz;
  
  {
    autoView(out_v,out,DeviceWrite);
    autoView(A_v,A,DeviceRead);
    autoView(B_v,B,DeviceRead);
    constexpr size_t shm_size = ablocksz*kblocksz*batchblocksz*sizeof(FloatType)
      + bblocksz*kblocksz*batchblocksz*sizeof(FloatType)
      + 2*batchblocksz*sizeof(FloatType);
    constexpr size_t bufoffb = ablocksz*kblocksz*batchblocksz*sizeof(FloatType) + batchblocksz*sizeof(FloatType);
    
    //akblocks:
    //0..ak-1|ak..2ak-1|2ak..3ak-1|
    //0..ak-1|*ak..2ak-1|**2ak..3ak-1|
    
    accelerator_for_1_3_shm(bbatch, batchblocksz, batchblock, batchblocks, blocka, ablocks, blockb, bblocks, 1, shm_size,  {
	FloatType* abuf = (FloatType*)(shared  +            bbatch * kblocksz*ablocksz*sizeof(FloatType) + bbatch*sizeof(FloatType) );  
	FloatType* bbuf = (FloatType*)(shared  + bufoffb  + bbatch * kblocksz*bblocksz*sizeof(FloatType) + bbatch*sizeof(FloatType) );
	
	int batch = bbatch + batchblocksz*batchblock;
	if(batch < batch_size){
	  FloatType abbuf[ablocksz * bblocksz] = {0.};
	  
	  for(int blockk=0;blockk<kblocks;blockk++){    
	    FloatType* abufp = abuf;
	    int a = blocka*ablocksz;
	    for(int aa=0;aa<ablocksz;aa++){
	      int k = blockk*kblocksz;
	      for(int kk=0;kk<kblocksz;kk++){
		*(abufp++) = a < asize && k < sizek ? *(A_v.data() + batch + a*astride + k*kstrideA) : 0.;
		++k;
	      }
	      ++a;
	    }
	    FloatType* bbufp = bbuf;
	    int b = blockb*bblocksz;
	    for(int bb=0;bb<bblocksz;bb++){
	      int k = blockk*kblocksz;
	      for(int kk=0;kk<kblocksz;kk++){
		*(bbufp++) = b < bsize && k < sizek ? *(B_v.data() + batch + b*bstride + k*kstrideB) : 0.;
		++k;
	      }
	      ++b;
	    }

	    for(int aa=0;aa<ablocksz;aa++){
	      for(int bb=0;bb<bblocksz;bb++){
		abufp = abuf + aa*kblocksz;
		bbufp = bbuf + bb*kblocksz;
		  
		FloatType res = (*abufp++) * (*bbufp++);
		for(int kk=1;kk<kblocksz;kk++)
		  res += (*abufp++) * (*bbufp++);
		  
		abbuf[bb + bblocksz*aa] += res*nrm;
	      }
	    }	    
	  }

	  for(int aa=0;aa<ablocksz;aa++){
	    int a = blocka*ablocksz + aa;
	    if(a < asize){
	      for(int bb=0;bb<bblocksz;bb++){
		int b = blockb*bblocksz + bb;
		if(b < bsize)
		  out_v(a,b,batch) = abbuf[bb + bblocksz*aa];		
	      }
	    }
	  }

	  
	}
	  
      });
  }
  return out;
}

//slower
template<typename FloatType>
Tensor<FloatType,3> batch3tensorContract_v4(const Tensor<FloatType,3> &A, const Tensor<FloatType,3> &B, int contract_dimA, int contract_dimB, FloatType nrm){
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

  constexpr int ablocksz = 2;
  int asize = sizes_out[0];
  int ablocks = (asize + ablocksz - 1)/ablocksz;

  constexpr int bblocksz = 2;
  int bsize = sizes_out[1];
  int bblocks = (bsize + bblocksz - 1)/bblocksz;

  constexpr int kblocksz = 2;
  int kblocks = (sizek + kblocksz - 1)/kblocksz;

  constexpr int batchblocksz = 16;
  int batchblocks = (batch_size + batchblocksz-1)/batchblocksz;
  
  {
    autoView(out_v,out,DeviceWrite);
    autoView(A_v,A,DeviceRead);
    autoView(B_v,B,DeviceRead);
    constexpr size_t shm_size = 2*ablocksz*kblocksz*batchblocksz*sizeof(FloatType)
      + 2*bblocksz*kblocksz*batchblocksz*sizeof(FloatType)
      + 2*ablocksz*bblocksz*batchblocksz*sizeof(FloatType)
      + 6*batchblocksz*sizeof(FloatType);
    constexpr size_t bufoffb = 2*ablocksz*kblocksz*batchblocksz*sizeof(FloatType) + 2*batchblocksz*sizeof(FloatType);
    constexpr size_t bufoffab = bufoffb + 2*bblocksz*kblocksz*batchblocksz*sizeof(FloatType) + 2*batchblocksz*sizeof(FloatType);
    
    //akblocks:
    //0..ak-1|ak..2ak-1|2ak..3ak-1|
    //0..ak-1|*ak..2ak-1|**2ak..3ak-1|
    
    accelerator_for_2_3_shm(bbatch, batchblocksz, kb, 2, batchblock, batchblocks, blocka, ablocks, blockb, bblocks, shm_size,  {
	int bkb = bbatch + batchblocksz * kb;
	FloatType* abuf = (FloatType*)(shared  +            bkb * kblocksz*ablocksz*sizeof(FloatType) + bkb*sizeof(FloatType) );  
	FloatType* bbuf = (FloatType*)(shared  + bufoffb  + bkb * kblocksz*bblocksz*sizeof(FloatType) + bkb*sizeof(FloatType) );
	FloatType* abbuf = (FloatType*)(shared + bufoffab + bkb*ablocksz*bblocksz*sizeof(FloatType) + bkb*sizeof(FloatType) );
	
	int batch = bbatch + batchblocksz*batchblock;

	FloatType* abbufp = abbuf;
	for(int aa=0;aa<ablocksz;aa++)
	  for(int bb=0;bb<bblocksz;bb++)
	    *(abbufp++) = 0.;

	for(int blockk=kb;blockk<kblocks;blockk+=2){    
	  FloatType* abufp = abuf;
	  int a = blocka*ablocksz;
	  for(int aa=0;aa<ablocksz;aa++){
	    int k = blockk*kblocksz;
	    for(int kk=0;kk<kblocksz;kk++){
	      *(abufp++) = batch < batch_size && a < asize && k < sizek ? *(A_v.data() + batch + a*astride + k*kstrideA) : 0.;
	      ++k;
	    }
	    ++a;
	  }
	  FloatType* bbufp = bbuf;
	  int b = blockb*bblocksz;
	  for(int bb=0;bb<bblocksz;bb++){
	    int k = blockk*kblocksz;
	    for(int kk=0;kk<kblocksz;kk++){
	      *(bbufp++) = batch < batch_size && b < bsize && k < sizek ? *(B_v.data() + batch + b*bstride + k*kstrideB) : 0.;
	      ++k;
	    }
	    ++b;
	  }

	  for(int aa=0;aa<ablocksz;aa++){
	    for(int bb=0;bb<bblocksz;bb++){
	      abufp = abuf + aa*kblocksz;
	      bbufp = bbuf + bb*kblocksz;
		  
	      FloatType res = (*abufp++) * (*bbufp++);
	      for(int kk=1;kk<kblocksz;kk++)
		res += (*abufp++) * (*bbufp++);
		  
	      abbuf[bb + bblocksz*aa] += res*nrm;
	    }
	  }	    
	}
	acceleratorSynchronizeBlock();
	if(batch < batch_size && kb==0){
	  FloatType* abbuf_nextkb = (FloatType*)(shared + bufoffab + (bkb + batchblocksz)*ablocksz*bblocksz*sizeof(FloatType) + (bkb + batchblocksz)*sizeof(FloatType) );
	  
	  for(int aa=0;aa<ablocksz;aa++){
	    int a = blocka*ablocksz + aa;
	    if(a < asize){
	      for(int bb=0;bb<bblocksz;bb++){
		int b = blockb*bblocksz + bb;
		if(b < bsize)
		  out_v(a,b,batch) = abbuf[bb + bblocksz*aa] + abbuf_nextkb[bb + bblocksz*aa];		
	      }
	    }
	  }
	}
	  
      });
  }
  return out;
}

//slower
template<typename FloatType>
Tensor<FloatType,3> batch3tensorContract_v5(const Tensor<FloatType,3> &A, const Tensor<FloatType,3> &B, int contract_dimA, int contract_dimB, FloatType nrm){
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

  constexpr int ablocksz = 2;
  int asize = sizes_out[0];
  int ablocks = (asize + ablocksz - 1)/ablocksz;

  constexpr int bblocksz = 2;
  int bsize = sizes_out[1];
  int bblocks = (bsize + bblocksz - 1)/bblocksz;

  constexpr int kblocksz = 2;
  int kblocks = (sizek + kblocksz - 1)/kblocksz;

  constexpr int batchblocksz = 32;
  int batchblocks = (batch_size + batchblocksz-1)/batchblocksz;
  
  {
    autoView(out_v,out,DeviceWrite);
    autoView(A_v,A,DeviceRead);
    autoView(B_v,B,DeviceRead);
    constexpr size_t nthr_block =  ablocksz*batchblocksz;
    constexpr size_t shm_size = kblocksz*nthr_block*sizeof(FloatType) //abuf[kk]
      + bblocksz*kblocksz*nthr_block*sizeof(FloatType) //bbuf[kk + kblocksz*bb]
      + bblocksz*nthr_block*sizeof(FloatType) //abbuf[bb]
      + 3*nthr_block*sizeof(FloatType); //offset for avoiding shared mem bank conflicts
    constexpr size_t bufoffb = kblocksz*nthr_block*sizeof(FloatType) + nthr_block*sizeof(FloatType);
    constexpr size_t bufoffab = bufoffb + bblocksz*kblocksz*nthr_block*sizeof(FloatType) + nthr_block*sizeof(FloatType);
    
    //akblocks:
    //0..ak-1|ak..2ak-1|2ak..3ak-1|
    //0..ak-1|*ak..2ak-1|**2ak..3ak-1|
    
    accelerator_for_2_3_shm(bbatch, batchblocksz, aa, ablocksz, batchblock, batchblocks, blocka, ablocks, blockb, bblocks, shm_size,  {
	int aabbatch = bbatch+batchblocksz*aa;
	FloatType* abuf = (FloatType*)(shared  +            aabbatch * kblocksz*sizeof(FloatType) + aabbatch*sizeof(FloatType) );  
	FloatType* bbuf = (FloatType*)(shared  + bufoffb  + aabbatch * kblocksz*bblocksz*sizeof(FloatType) + aabbatch*sizeof(FloatType) );
	FloatType* abbuf = (FloatType*)(shared + bufoffab + aabbatch * bblocksz*sizeof(FloatType) + aabbatch*sizeof(FloatType) );
	
	int batch = bbatch + batchblocksz*batchblock;
	int a = aa + blocka*ablocksz;
	
	for(int bb=0;bb<bblocksz;bb++)
	  abbuf[bb] = 0.;

	for(int blockk=0;blockk<kblocks;blockk++){    
	  FloatType* abufp = abuf;
	  int k = blockk*kblocksz;
	  for(int kk=0;kk<kblocksz;kk++){
	    *(abufp++) = batch < batch_size && a < asize && k < sizek ? *(A_v.data() + batch + a*astride + k*kstrideA) : 0.;
	    ++k;
	  }
	  FloatType* bbufp = bbuf;
	  int b = blockb*bblocksz;
	  for(int bb=0;bb<bblocksz;bb++){
	    int k = blockk*kblocksz;
	    for(int kk=0;kk<kblocksz;kk++){
	      *(bbufp++) = batch < batch_size && b < bsize && k < sizek ? *(B_v.data() + batch + b*bstride + k*kstrideB) : 0.;
	      ++k;
	    }
	    ++b;
	  }

	  for(int bb=0;bb<bblocksz;bb++){
	    abufp = abuf;
	    bbufp = bbuf + bb*kblocksz;
	    
	    FloatType res = (*abufp++) * (*bbufp++);
	    for(int kk=1;kk<kblocksz;kk++)
	      res += (*abufp++) * (*bbufp++);
	    
	    abbuf[bb] += res*nrm;
	  }

	}

	if(batch < batch_size && a < asize){
	  for(int bb=0;bb<bblocksz;bb++){
	    int b = blockb*bblocksz + bb;
	    if(b < bsize)
	      out_v(a,b,batch) = abbuf[bb];
	  }
	}
      });
  }
  return out;
}


//marginally faster than base in many cases, worse in others. Does better for batch size=64
template<typename FloatType>
Tensor<FloatType,3> batch3tensorContract_v6(const Tensor<FloatType,3> &A, const Tensor<FloatType,3> &B, int contract_dimA, int contract_dimB, FloatType nrm){
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

  constexpr int bblocksz = 4;
  int bsize = sizes_out[1];
  int bblocks = (bsize + bblocksz - 1)/bblocksz;

  constexpr int kblocksz = 8;
  int kblocks = (sizek + kblocksz - 1)/kblocksz;

  constexpr int batchblocksz = 32;
  int batchblocks = (batch_size + batchblocksz-1)/batchblocksz;
  
  {
    autoView(out_v,out,DeviceWrite);
    autoView(A_v,A,DeviceRead);
    autoView(B_v,B,DeviceRead);
    constexpr size_t shm_size = kblocksz*batchblocksz*sizeof(FloatType)
      + bblocksz*batchblocksz*sizeof(FloatType)
      + 2*batchblocksz*sizeof(FloatType);
    constexpr size_t bufoffab = kblocksz*batchblocksz*sizeof(FloatType) + batchblocksz*sizeof(FloatType);
    
    //akblocks:
    //0..ak-1|ak..2ak-1|2ak..3ak-1|
    //0..ak-1|*ak..2ak-1|**2ak..3ak-1|
    
    accelerator_for_1_3_shm(bbatch, batchblocksz, batchblock, batchblocks, a, sizes_out[0], blockb, bblocks, 1, shm_size,  {
	FloatType* abuf = (FloatType*)(shared  +            bbatch * kblocksz*sizeof(FloatType) + bbatch*sizeof(FloatType) );  
	FloatType* abbuf = (FloatType*)(shared + bufoffab + bbatch*bblocksz*sizeof(FloatType) + bbatch*sizeof(FloatType) );
	
	int batch = bbatch + batchblocksz*batchblock;
	for(int bb=0;bb<bblocksz;bb++)
	  abbuf[bb] = 0.;

	for(int blockk=0;blockk<kblocks;blockk++){    
	  int k = blockk*kblocksz;
	  for(int kk=0;kk<kblocksz;kk++){
	    abuf[kk] = batch < batch_size && k < sizek ? *(A_v.data() + batch + a*astride + k*kstrideA) : 0.;
	    ++k;
	  }
	  int b = blockb*bblocksz;
	  for(int bb=0;bb<bblocksz;bb++){
	    int k = blockk*kblocksz;
	    FloatType* abufp = abuf;
	    FloatType res = 0.0;
	    for(int kk=0;kk<kblocksz;kk++){
	      FloatType Bbk = batch < batch_size && b < bsize && k < sizek ? *(B_v.data() + batch + b*bstride + k*kstrideB) : 0.;
	      res += (*abufp++) * Bbk;
	      ++k;
	    }
	    abbuf[bb] += res*nrm;
	    ++b;
	  }
	}

	if(batch < batch_size){
	  for(int bb=0;bb<bblocksz;bb++){
	    int b = blockb*bblocksz + bb;
	    if(b < bsize)
	      out_v(a,b,batch) = abbuf[bb];		
	  }
	}
	  
      });
  }
  return out;
}


template<typename FloatType>
Tensor<FloatType,3> batch3tensorContract_v7(const Tensor<FloatType,3> &A, const Tensor<FloatType,3> &B, int contract_dimA, int contract_dimB, FloatType nrm){
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
    
    constexpr int batchblocksz = 32;
    int batchblocks = (batch_size + batchblocksz-1)/batchblocksz;
  
    accelerator_for_1_3(bbatch, batchblocksz, batchblock, batchblocks,  a, sizes_out[0], b, sizes_out[1],   1, {
	int batch = bbatch + batchblock*batchblocksz;
	if(batch < batch_size){
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
	}
      });
  }
  return out;
}


template<typename FloatType>
Tensor<FloatType,3> batch3tensorContract_v8(const Tensor<FloatType,3> &A, const Tensor<FloatType,3> &B, int contract_dimA, int contract_dimB, FloatType nrm){
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

  autoView(out_v,out,DeviceWrite);
  autoView(A_v,A,DeviceRead);
  autoView(B_v,B,DeviceRead);
  
  if(sizes_out[0] % 2 == 0){
    int sizea = sizes_out[0];
    
    accelerator_for3d(batch, batch_size, ablock, sizea/2, b, sizes_out[1],   1, {
	FloatType* A_p = A_v.data() + batch + 2*ablock*astride;
	FloatType* B_p = B_v.data() + batch + b*bstride;
	FloatType res0 = 0., res1 = 0.;
	
	for(int k=0;k<sizek;k++){
	  FloatType Bk = *B_p;
	  FloatType Ak0 = *A_p;
	  FloatType Ak1 = *(A_p + astride);
	  res0 += Ak0 * Bk;
	  res1 += Ak1 * Bk;
	  A_p += kstrideA;
	  B_p += kstrideB;
	}
	out_v(2*ablock,b,batch) = res0 * nrm;
	out_v(2*ablock+1,b,batch) = res1 * nrm;
      });

  }else{

    
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

template<typename FloatType>
Tensor<FloatType,3> batch3tensorContract_v9(const Tensor<FloatType,3> &A, const Tensor<FloatType,3> &B, int contract_dimA, int contract_dimB, FloatType nrm){
#ifdef USE_BLAS
  Vector<FloatType> Abatch = transformBatchMatrix(contract_dimA, !contract_dimA, A); //A[!contract_dim_A, contract_dim_A] in column major
  Vector<FloatType> Bbatch = transformBatchMatrix(!contract_dimB, contract_dimB, B); //B[contract_dim_B, !contract_dim_B]
  int nbatch = A.size(2);
  
  int m = A.size(!contract_dimA);
  int n = B.size(!contract_dimB);
  int k = A.size(contract_dimA);
  FloatType beta = 0.;
  int lda = A.size(!contract_dimA); //column major,  leading dimension is number of rows
  int ldb = B.size(contract_dimB);

  int omat_sz = A.size(!contract_dimA) * B.size(!contract_dimB);
  int ldc = A.size(!contract_dimA); //C[ A.size(!contract_dim_A), B.size(!contract_dim_B) ] in column major
  Vector<FloatType> Cbatch(omat_sz * nbatch);

  int Astride = A.size(!contract_dimA)*A.size(contract_dimA);
  int Bstride = B.size(!contract_dimB)*B.size(contract_dimB);
  int Cstride = omat_sz;
  {
    autoView(Cbatch_v,Cbatch,DeviceWrite);
    autoView(Abatch_v,Abatch,DeviceRead);
    autoView(Bbatch_v,Bbatch,DeviceRead);
    
    batchedGEMM(NoTranspose, NoTranspose, 
		m, n, k,
		&nrm,
		Abatch_v.data(), lda, Astride,
		Bbatch_v.data(), ldb, Bstride,
		&beta,
		Cbatch_v.data(), ldc, Cstride,
		nbatch);
  }
  //C_{ijb} = \sum_k A_{kib} B_{jkb} * nrm
  Tensor<FloatType,3> C(A.size(!contract_dimA),B.size(!contract_dimB),A.size(2));
  untransformBatchMatrix(1,0,C, Cbatch);
  return C;
#else
  return batch3tensorContract(A,B,contract_dimA,contract_dimB,nrm);
#endif
}
  

int main(int argc, char** argv){
  initialize(argc,argv);
  std::mt19937 rng(1234);

  std::vector<int> contract_dim_sizes = { 2, 5,  8, 16, 32, 33, 64, 256, 512 };
  std::vector<int> batch_sizes = {1, 5, 8 , 16, 32, 33, 64};
  std::vector<int> other_dim_sizes = { 2, 5, 8, 16, 32, 33, 64, 256, 512 };

  // std::vector<int> contract_dim_sizes = {32,64, 128};
  // std::vector<int> batch_sizes = {16, 32, 64};
  // std::vector<int> other_dim_sizes = { 16, 32, 64, 128, 256 };
  
  // std::vector<int> contract_dim_sizes = { 32 };
  // std::vector<int> batch_sizes = { 64};
  // std::vector<int> other_dim_sizes = { 256 };
  
  
  for(int contract_dim_A=0; contract_dim_A<1; contract_dim_A++){
    for(int contract_dim_B=0; contract_dim_B<1; contract_dim_B++){
      for(auto other_dim_size : other_dim_sizes){
	for(auto contract_dim_size : contract_dim_sizes){
	  for(auto batch_size : batch_sizes){
	    int asz[3] = { other_dim_size, other_dim_size, batch_size };
	    asz[contract_dim_A] = contract_dim_size;
	  
	    int bsz[3] = { other_dim_size, other_dim_size, batch_size };
	    bsz[contract_dim_B] = contract_dim_size;

	    size_t FLOPS = size_t(other_dim_size) * size_t(other_dim_size) *size_t(batch_size) * size_t(contract_dim_size)*2  +   size_t(other_dim_size) * size_t(other_dim_size) *size_t(batch_size);
	    
#if 1
	    {
	      Tensor<double,3> a(asz), b(bsz);
	      uniformRandom(a,rng);
	      uniformRandom(b,rng);
	      
	      Tensor<double,3> cgot = batch3tensorContract_v9(a,b,contract_dim_A,contract_dim_B,3.141);
	      Tensor<double,3> ctest = batch3tensorContractBase(a,b,contract_dim_A,contract_dim_B,3.141);
	      assert(abs_near(cgot,ctest,1e-5,true));
	    }
#endif
	    
	    Tensor<float,3> a(asz), b(bsz);
	    uniformRandom(a,rng);
	    uniformRandom(b,rng);
	    
	    double mu, sigma;
	  
	    Tensor<float,3> c;
	    benchmark(mu, sigma, 100, 1, [&]{
	      profileStart();
	      c = batch3tensorContract_v9(a,b,contract_dim_A,contract_dim_B,3.141f);
	      profileStop();
	    }, []{});

#if 1
	    double mu_base, sigma_base;
	    benchmark(mu_base, sigma_base, 100, 1, [&]{
	      profileStart();
	      c = batch3tensorContract(a,b,contract_dim_A,contract_dim_B,3.141f);
	      profileStop();
	    }, []{});

	    std::cout << "contract_dim_A:" << contract_dim_A << " contract_dim_B:" << contract_dim_B << "\tother_dim_size:" << other_dim_size << "\tcontract_dim_size:" << contract_dim_size << "\tbatch_size:" << batch_size << "\tresult: " << mu/1e-6 << "+-" << sigma/1e-6 << "us (" << FLOPS/mu/1e9  << " Gflops) base: " << mu_base/1e-6 << "+-" << sigma_base/1e-6 << "us (" << FLOPS/mu_base/1e9 << " Gflops)" << std::endl;
#endif
	  }
	}
      }
    }
  }
 
  return 0;
}


#else
int main(void){
  std::cout << "Benchmarks currently GPU-only" << std::endl;
  return 0;
}
#endif
