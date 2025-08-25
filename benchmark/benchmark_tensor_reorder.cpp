#include <HPCortex.hpp>
#include <Testing.hpp>

#ifdef USE_GPU

template<typename FloatType, int Dim>
void untransformBatchMatrix_base(int rowdim, int coldim, Tensor<FloatType,Dim> &tens, Vector<FloatType> &from){
  assert(rowdim != coldim);
  autoView(tens_v,tens,DeviceWrite);
  autoView(from_v,from,DeviceRead);
  
  accelerator_for_gen(1,0,splitBlock<32>(), i, tens.data_len(),{
      int coord[Dim];
      tensorOffsetUnmap<Dim>(coord, tens_v.sizeArray(), i);

      int o=0;
      for(int d=0;d<Dim;d++) if(d != rowdim && d != coldim) o = o * tens_v.size(d) + coord[d];
      //for(int d=Dim-1;d>=0;d--) if(d != rowdim && d != coldim) o = o * tens_v.size(d) + coord[d];
    
      int off = coord[coldim] + tens_v.size(coldim)*( coord[rowdim] + tens_v.size(rowdim) * o );
      tens_v.data()[i] = from_v(off);
    });
}


void benchmarkUntransformBatchMatrix(){
  std::cout << "Benchmarking untransformBatchMatrix\n";
  std::mt19937 rng(1234);
  typedef float FloatType;

  // std::vector<int> osizes = { 1, 5, 8, 16, 32, 33, 64, 128, 256, 512 };
  // std::vector<int> bsizes = { 1, 5, 8, 16, 32, 33, 64 };

  std::vector<int> osizes = { 256 };
  std::vector<int> bsizes = { 64 };
 
  for(auto osize : osizes){
    for(auto bsize : bsizes){
      
      int tsz[3] = {osize,osize,bsize};
      Tensor<float,3> tens(tsz);
  
      Vector<FloatType> arr(tens.data_len());
      uniformRandom(arr, rng);

      for(int rowdim=0;rowdim<2;rowdim++){
	Tensor<float,3> tens_expect(tsz);
	untransformBatchMatrix_base(rowdim, !rowdim, tens_expect, arr);
	untransformBatchMatrix(rowdim, !rowdim, tens, arr);
	assert(equal(tens,tens_expect,true));

	double mu, sigma;
  
	benchmark(mu, sigma, 100, 1, [&]{
	  profileStart();
	  untransformBatchMatrix(rowdim, !rowdim, tens, arr);
	  profileStop();
	}, []{});

	double mu_base, sigma_base;
	benchmark(mu_base, sigma_base, 100, 1, [&]{
	  profileStart();
	  untransformBatchMatrix_base(rowdim, !rowdim, tens_expect, arr);
	  profileStop();
	}, []{});
  
	std::cout << "rowdim:" << rowdim << " sizei:" << tsz[0] << " sizej:" << tsz[1] << " batch_size:" << tsz[2] << "\tresult: " << mu/1e-6 << "+-" << sigma/1e-6 << "us, base: " << mu_base/1e-6 << "+-" << sigma_base/1e-6 << "us" << std::endl;
      }
    }
  }
}

template<typename FloatType, int Dim>
Vector<FloatType> transformBatchMatrix_base(int rowdim, int coldim, const Tensor<FloatType,Dim> &tens){
  assert(rowdim != coldim);
  Vector<FloatType> into(tens.data_len());
  autoView(tens_v,tens,DeviceRead);
  autoView(into_v,into,DeviceWrite);
  
  accelerator_for_gen(1,0,splitBlock<32>(), i, tens.data_len(),{
      int coord[Dim];
      tensorOffsetUnmap<Dim>(coord, tens_v.sizeArray(), i);

      int o=0;
      //for(int d=Dim-1;d>=0;d--) if(d != rowdim && d != coldim) o = o * tens_v.size(d) + coord[d];
      for(int d=0;d<Dim;d++) if(d != rowdim && d != coldim) o = o * tens_v.size(d) + coord[d];
    
      int off = coord[coldim] + tens_v.size(coldim)*( coord[rowdim] + tens_v.size(rowdim) * o );
      into_v(off) = tens_v.data()[i];
    });

  return into;
}

void benchmarkTransformBatchMatrix(){
  std::cout << "Benchmarking transformBatchMatrix\n";
  std::mt19937 rng(1234);
  typedef float FloatType;
  
  std::vector<int> osizes = { 1, 5, 8, 16, 32, 33, 64, 128, 256, 512 };
  std::vector<int> bsizes = { 1, 5, 8, 16, 32, 33, 64 };

  //std::vector<int> osizes = { 256 };
  //std::vector<int> bsizes = { 64 };
 
  for(auto osize : osizes){
    for(auto bsize : bsizes){
      int tsz[3] = {osize,osize,bsize};
      Tensor<float,3> tens(tsz);
      uniformRandom(tens, rng);
      
      for(int rowdim=0;rowdim<2;rowdim++){
	Tensor<float,3> tens_expect(tsz);
	Vector<FloatType> expect = transformBatchMatrix_base(rowdim, !rowdim, tens);
	Vector<FloatType> got = transformBatchMatrix(rowdim, !rowdim, tens);
	assert(equal(got,expect,true));

	double mu, sigma;
  
	benchmark(mu, sigma, 100, 1, [&]{
	  profileStart();
	  got = transformBatchMatrix(rowdim, !rowdim, tens);
	  profileStop();
	}, []{});

	double mu_base, sigma_base;
	benchmark(mu_base, sigma_base, 100, 1, [&]{
	  profileStart();
	  got = transformBatchMatrix_base(rowdim, !rowdim, tens);
	  profileStop();
	}, []{});
  
	std::cout << "rowdim:" << rowdim << " sizei:" << tsz[0] << " sizej:" << tsz[1] << " batch_size:" << tsz[2] << "\tresult: " << mu/1e-6 << "+-" << sigma/1e-6 << "us, base: " << mu_base/1e-6 << "+-" << sigma_base/1e-6 << "us" << std::endl;
      }
    }
  }


}


template<typename FloatType, int Dim>
Vector<FloatType> transformBatchVector_base(int vecdim, const Tensor<FloatType,Dim> &tens){
  Vector<FloatType> into(tens.data_len());
  autoView(tens_v,tens,DeviceRead);
  autoView(into_v,into,DeviceWrite);
  
  accelerator_for_gen(1,0,splitBlock<32>(), i, tens.data_len(),{
      int coord[Dim];
      tensorOffsetUnmap<Dim>(coord, tens_v.sizeArray(), i);

      int o=0;
      for(int d=0;d<Dim;d++) if(d != vecdim) o = o * tens_v.size(d) + coord[d];
    
      int off = coord[vecdim] + tens_v.size(vecdim)*o;
      into_v(off) = tens_v.data()[i];
    });

  return into;
}

template<typename FloatType, int Dim>
Vector<FloatType> transformBatchVector_v2(int vecdim, const Tensor<FloatType,Dim> &tens){
  assert(vecdim != Dim-1); //batch dim
  Vector<FloatType> into(tens.data_len());
  autoView(tens_v,tens,DeviceRead);
  autoView(into_v,into,DeviceWrite);

  int stride = tensorDimensionStride<Dim>(vecdim, tens.sizeArray());
  int other_size_lin=1;
  for(int i=0;i<Dim-1;i++)
    if(i != vecdim) other_size_lin *= tens.size(i);
  int batch_size = tens.size(Dim-1);

  constexpr int iblocksize = 32;
  int iblocks = (tens.size(vecdim) + iblocksize -1)/iblocksize;

  accelerator_for_3d_gen(1,2,shm( (batch_size+1)*iblocksize*sizeof(FloatType)), t, batch_size, bi, iblocks, o, other_size_lin,{
      FloatType* bstore = (FloatType*)shared;
      int ioff = bi*iblocksize;
      int irem = tens_v.size(vecdim) - ioff;
      //int iblocksize_actual = irem < iblocksize ? irem : iblocksize;
      int iblocksize_actual = min(irem,iblocksize);
      FloatType* tens_p = tens_v.data() + batchTensorDimensionBaseLin<Dim>(vecdim, t, o, tens_v.sizeArray());
      
      //parallel load batch_size data into shared
      for(int ii=0;ii<iblocksize_actual;ii++){      
	int i = ii + ioff;
	bstore[t + (batch_size+1)*ii] = *(tens_p + i*stride);
      }
      acceleratorSynchronizeBlock();

      //parallel write iblocksize into output
      for(int b=0;b<batch_size;b++){
	int ii=0;
	while( (ii + t) < iblocksize_actual){
	  into_v( ii + t + ioff + tens_v.size(vecdim)*(b + batch_size*o) ) = bstore[b + (batch_size+1)*(ii+t) ];	  
	  ii += batch_size;
	}
      }

    });
  
  return into;
}

template<typename FloatType, int Dim>
Vector<FloatType> transformBatchVector_v3(int vecdim, const Tensor<FloatType,Dim> &tens){
  assert(vecdim != Dim-1); //batch dim
  Vector<FloatType> into(tens.data_len());
  autoView(tens_v,tens,DeviceRead);
  autoView(into_v,into,DeviceWrite);

  int stride = tensorDimensionStride<Dim>(vecdim, tens.sizeArray());
  int other_size_lin=1;
  for(int i=0;i<Dim-1;i++)
    if(i != vecdim) other_size_lin *= tens.size(i);
  int batch_size = tens.size(Dim-1);

  constexpr int bblocksize = 32;
  int bblocks = (batch_size + bblocksize - 1)/bblocksize;
  
  constexpr int iblocksize = 8;
  int iblocks = (tens.size(vecdim) + iblocksize -1)/iblocksize;

  accelerator_for_5d_gen(2,3,shm( (bblocksize+1)*iblocksize*sizeof(FloatType)), t, bblocksize, u, iblocksize,  bblock, bblocks, bi, iblocks, o, other_size_lin,{
      FloatType* bstore = (FloatType*)shared;
      int ioff = bi*iblocksize;
      int iblocksize_actual = min(int(tens_v.size(vecdim) - ioff),iblocksize);

      int boff = bblock*bblocksize;
      int bblocksize_actual = min(batch_size - boff,bblocksize);     
      
      FloatType* tens_p = tens_v.data() + batchTensorDimensionBaseLin<Dim>(vecdim, 0, o, tens_v.sizeArray());
      
      //parallel load batch_size * iblocksize_actual data into shared
      int bb=t;
      int ii=u;      
      
      if(ii<iblocksize_actual && bb < bblocksize_actual){
	int i = ii + ioff;
	bstore[bb + (bblocksize+1)*ii] = tens_p[bb + boff + i*stride];
      }
      acceleratorSynchronizeBlock();

      //swap the role of t and u to get coalesced writes
      bb=u;
      ii=t;
      while(bb < bblocksize_actual){
	if(ii < iblocksize_actual)
	  into_v( ii + ioff + tens_v.size(vecdim)*(bb + boff + batch_size*o) ) = bstore[bb + (bblocksize+1)*ii ];
	bb += iblocksize;
      }

    });
  
  return into;
}


template<typename FloatType, int Dim>
Vector<FloatType> transformBatchVector_v4(int vecdim, const Tensor<FloatType,Dim> &tens){
  assert(vecdim != Dim-1); //batch dim
  Vector<FloatType> into(tens.data_len());
  autoView(tens_v,tens,DeviceRead);
  autoView(into_v,into,DeviceWrite);

  int stride = tensorDimensionStride<Dim>(vecdim, tens.sizeArray());
  int other_size_lin=1;
  for(int i=0;i<Dim-1;i++)
    if(i != vecdim) other_size_lin *= tens.size(i);
  int batch_size = tens.size(Dim-1);

  constexpr int bblocksize = 32;
  int bblocks = (batch_size + bblocksize -1)/bblocksize;

  int isize = tens.size(vecdim);
  constexpr int iblocksize = 32;
  int iblocks = (isize + iblocksize -1)/iblocksize;
  
  accelerator_for_4d_gen(1,3,shm( (bblocksize+1)*iblocksize*sizeof(FloatType)), t, bblocksize, bblock, bblocks, bi, iblocks, o, other_size_lin,{
      FloatType* bstore = (FloatType*)shared;
      int ioff = bi*iblocksize;
      int iblocksize_actual = min(isize - ioff,iblocksize);

      int boff = bblock*bblocksize;
      int bblocksize_actual = min(batch_size - boff,bblocksize);
      
      FloatType* tens_p = tens_v.data() + batchTensorDimensionBaseLin<Dim>(vecdim, t + boff, o, tens_v.sizeArray());
      
      //parallel load batch_size data into shared
      for(int ii=0;ii<iblocksize_actual;ii++){      
	int i = ii + ioff;
	if(t < bblocksize_actual) bstore[t + (bblocksize+1)*ii] = *(tens_p + i*stride);
      }
      acceleratorSynchronizeBlock();

      //parallel write iblocksize into output
      for(int bb=0;bb<bblocksize_actual;bb++){
	int ii=t;
	while( ii < iblocksize_actual){
	  into_v( ii + ioff + isize*(bb + boff + batch_size*o) ) = bstore[bb + (bblocksize+1)*ii ];	  
	  ii += bblocksize;
	}
      }

    });
  
  return into;
}


void benchmarkTransformBatchVector(){
  std::cout << "Benchmarking transformBatchVector\n";
  std::mt19937 rng(1234);
  typedef float FloatType;
  
  std::vector<int> osizes = { 1, 5, 8, 16, 32, 33, 64, 128, 256, 512 };
  std::vector<int> bsizes = { 1, 5, 8, 16, 32, 33, 64 };

  // std::vector<int> osizes = { 512 };
  //  std::vector<int> bsizes = { 32 };
  
  
  for(auto osize : osizes){
    for(auto bsize : bsizes){
      int tsz[3] = {osize,osize,bsize};
      Tensor<float,3> tens(tsz);
      uniformRandom(tens, rng);
      
      for(int vecdim=0;vecdim<2;vecdim++){
	Tensor<float,3> tens_expect(tsz);
	Vector<FloatType> expect = transformBatchVector_base(vecdim,tens);
	Vector<FloatType> got = transformBatchVector_v4(vecdim, tens);
	assert(equal(got,expect,true));

	double mu, sigma;
  
	benchmark(mu, sigma, 100, 1, [&]{
	  profileStart();
	  got = transformBatchVector_v4(vecdim, tens);
	  profileStop();
	}, []{});

	double mu_base, sigma_base;
	benchmark(mu_base, sigma_base, 100, 1, [&]{
	  profileStart();
	  got = transformBatchVector(vecdim, tens);
	  profileStop();
	}, []{});
  
	std::cout << "vecdim:" << vecdim << " sizei:" << tsz[0] << " sizej:" << tsz[1] << "batch_size:" << tsz[2] << "\tresult: " << mu/1e-6 << "+-" << sigma/1e-6 << "us, base: " << mu_base/1e-6 << "+-" << sigma_base/1e-6 << "us" << std::endl;
      }
    }
  }


}

template<typename FloatType, int Dim>
void untransformBatchVector_base(int vecdim, Tensor<FloatType,Dim> &tens, const Vector<FloatType> &from){
  autoView(tens_v,tens,DeviceWrite);
  autoView(from_v,from,DeviceRead);
  
  accelerator_for_gen(1,0,splitBlock<32>(), i, tens.data_len(),{
      int coord[Dim];
      tensorOffsetUnmap<Dim>(coord, tens_v.sizeArray(), i);

      int o=0;
      for(int d=0;d<Dim;d++) if(d != vecdim) o = o * tens_v.size(d) + coord[d];
    
      int off = coord[vecdim] + tens_v.size(vecdim)*o;
      tens_v.data()[i] = from_v(off);
    });
}

template<typename FloatType, int Dim>
void untransformBatchVector_v2(int vecdim, Tensor<FloatType,Dim> &tens, const Vector<FloatType> &from){
  assert(vecdim != Dim-1); //batch dim
  autoView(tens_v,tens,DeviceWrite);
  autoView(from_v,from,DeviceRead);

  int stride = tensorDimensionStride<Dim>(vecdim, tens.sizeArray());
  int other_size_lin=1;
  for(int i=0;i<Dim-1;i++)
    if(i != vecdim) other_size_lin *= tens.size(i);
  int batch_size = tens.size(Dim-1);

  int iblocksz = std::min(32,tens.size(vecdim));
  int iblocks = (tens.size(vecdim) + iblocksz -1)/iblocksz;

  constexpr int bblocksz = 32;
  int bblocks = (batch_size + bblocksz - 1)/bblocksz;
  
  accelerator_for_4d_gen(1,3,shm( (iblocksz+1)*bblocksz*sizeof(FloatType)), t, iblocksz, bi,iblocks,  bblock, bblocks, o, other_size_lin,{
      FloatType* bstore = (FloatType*)shared;
      int ioff = bi*iblocksz;
      int irem = tens_v.size(vecdim) - ioff;
      int iblocksz_actual = irem < iblocksz ? irem : iblocksz;

      int boff = bblock*bblocksz;
      int brem = batch_size - boff;
      int bblocksz_actual = brem < bblocksz ? brem : bblocksz;

      //parallel load iblocksz data into shared
      for(int bb=0;bb<bblocksz_actual;bb++)
	bstore[t + (iblocksz+1)*bb] = from_v( t + ioff + tens_v.size(vecdim)*(bb + boff + batch_size*o ) );

      acceleratorSynchronizeBlock();
      
      FloatType* tens_p = tens_v.data() + batchTensorDimensionBaseLin<Dim>(vecdim, boff, o, tens_v.sizeArray());

      //parallel write bblocksz into output
      for(int ii=0;ii<iblocksz_actual;ii++){
	int bb =0;
	while( (bb+t) < bblocksz_actual){
	  *(tens_p + bb+t + (ii + ioff)*stride) = bstore[ii + (iblocksz+1)*(bb+t)];
	  bb+=iblocksz;
	}
      }
	  
    });
}



template<typename FloatType, int Dim>
void untransformBatchVector_v3(int vecdim, Tensor<FloatType,Dim> &tens, const Vector<FloatType> &from){
  assert(vecdim != Dim-1); //batch dim
  autoView(tens_v,tens,DeviceWrite);
  autoView(from_v,from,DeviceRead);

  int stride = tensorDimensionStride<Dim>(vecdim, tens.sizeArray());
  int other_size_lin=1;
  for(int i=0;i<Dim-1;i++)
    if(i != vecdim) other_size_lin *= tens.size(i);
  int batch_size = tens.size(Dim-1);
  int isize = tens_v.size(vecdim);
  
  constexpr int iblocksz = 32;
  int iblocks = (isize + iblocksz -1)/iblocksz;

  constexpr int bblocksz = 32;
  int bblocks = (batch_size + bblocksz - 1)/bblocksz;
  
  accelerator_for_4d_gen(1,3,shm( (iblocksz+1)*bblocksz*sizeof(FloatType)), t, iblocksz, bi,iblocks,  bblock, bblocks, o, other_size_lin,{
      FloatType* bstore = (FloatType*)shared;
      int ioff = bi*iblocksz;
      int iblocksz_actual = min(isize - ioff, iblocksz);

      int boff = bblock*bblocksz;
      int bblocksz_actual = min(batch_size - boff, bblocksz);

      //parallel load iblocksz data into shared
      for(int bb=0;bb<bblocksz_actual;bb++)
	if(t < iblocksz_actual) bstore[t + (iblocksz+1)*bb] = from_v( t + ioff + isize*(bb + boff + batch_size*o ) );

      acceleratorSynchronizeBlock();
      
      FloatType* tens_p = tens_v.data() + batchTensorDimensionBaseLin<Dim>(vecdim, boff, o, tens_v.sizeArray());

      //parallel write bblocksz into output
      for(int ii=0;ii<iblocksz_actual;ii++){
	int bb =t;
	while( bb < bblocksz_actual){
	  *(tens_p + bb + (ii + ioff)*stride) = bstore[ii + (iblocksz+1)*bb];
	  bb+=iblocksz;
	}
      }
	  
    });
}


void benchmarkUntransformBatchVector(){
  std::cout << "Benchmarking untransformBatchVector\n";
  std::mt19937 rng(1234);
  typedef float FloatType;
  
  std::vector<int> osizes = { 1, 5, 8, 16, 32, 33, 64, 128, 256, 512 };
  std::vector<int> bsizes = { 1, 5, 8, 16, 32, 33, 64 };

  //std::vector<int> osizes = { 512 };
  // std::vector<int> bsizes = { 32 };
  
  for(auto osize : osizes){
    for(auto bsize : bsizes){
      int tsz[3] = {osize,osize,bsize};
      Tensor<float,3> tens(tsz), tens_expect(tsz);
      Vector<FloatType> vec(tens.data_len());
      uniformRandom(vec, rng);
      
      for(int vecdim=0;vecdim<2;vecdim++){
	untransformBatchVector_base(vecdim,tens_expect,vec);
	untransformBatchVector_v3(vecdim, tens, vec);
	assert(equal(tens,tens_expect,true));

	double mu, sigma;
  
	benchmark(mu, sigma, 100, 1, [&]{
	  profileStart();
	  untransformBatchVector_v3(vecdim, tens, vec);
	  profileStop();
	}, []{});

	double mu_base, sigma_base;
	benchmark(mu_base, sigma_base, 100, 1, [&]{
	  profileStart();
	  untransformBatchVector(vecdim, tens, vec);
	  profileStop();
	}, []{});
  
	std::cout << "vecdim:" << vecdim << " sizei:" << tsz[0] << " sizej:" << tsz[1] << "batch_size:" << tsz[2] << "\tresult: " << mu/1e-6 << "+-" << sigma/1e-6 << "us, base: " << mu_base/1e-6 << "+-" << sigma_base/1e-6 << "us" << std::endl;
      }
    }
  }


}



int main(int argc, char** argv){  
  initialize(argc,argv);
  //benchmarkUntransformBatchMatrix();
  //benchmarkTransformBatchMatrix();
  //benchmarkTransformBatchVector();
  benchmarkUntransformBatchVector();
  return 0;
}

#else
int main(void){
  return 0;
}
#endif
