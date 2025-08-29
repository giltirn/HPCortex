#pragma once
#include <HPCortexConfig.h>
#include<strings.h>
#include<cstdlib>
#include<memory.h>
#include<stdio.h>
#include<cassert>
#include<cmath>
#include<iostream>
//Functionality for writing generic GPU kernels with CPU fallback
//Adapted from Peter Boyle's Grid library https://github.com/paboyle/Grid

//Host-side looping, always available
#ifdef _OPENMP
#define USE_OMP
#include <omp.h>
#endif

#ifdef USE_OMP //host-side with threading
#define DO_PRAGMA_(x) _Pragma (#x)
#define DO_PRAGMA(x) DO_PRAGMA_(x)
#define thread_num(a) omp_get_thread_num()
#define thread_max(a) omp_get_max_threads()
#define set_threads(a) omp_set_num_threads(a)
#define in_thread_parallel_region(a) omp_in_parallel()

#define thread_for( i, num, ... )  \
  DO_PRAGMA(omp parallel for schedule(static)) for ( uint64_t i=0;i<num;i++) { __VA_ARGS__ } ;

#define thread_for3d( i1, n1, i2, n2, i3, n3, ... )	\
  DO_PRAGMA(omp parallel for collapse(3))  \
  for ( uint64_t i3=0;i3<n3;i3++) {	   \
  for ( uint64_t i2=0;i2<n2;i2++) {	   \
  for ( uint64_t i1=0;i1<n1;i1++) {	   \
  { __VA_ARGS__ } ;			   \
  }}}

#define thread_for2d( i1, n1,i2,n2, ... )  \
  DO_PRAGMA(omp parallel for collapse(2))  \
  for ( uint64_t i2=0;i2<n2;i2++) {	   \
  for ( uint64_t i1=0;i1<n1;i1++) {	   \
  { __VA_ARGS__ } ;			   \
  }}

#else //no threading
#define DO_PRAGMA_(x) 
#define DO_PRAGMA(x) 
#define thread_num(a) (0)
#define thread_max(a) (1)
#define set_threads(a)
#define in_thread_parallel_region(a) (false)

#define thread_for( i, num, ... )  \
  for ( uint64_t i=0;i<num;i++) { __VA_ARGS__ } ;

#define thread_for3d( i1, n1, i2, n2, i3, n3, ... )	\
  for ( uint64_t i3=0;i3<n3;i3++) {	   \
  for ( uint64_t i2=0;i2<n2;i2++) {	   \
  for ( uint64_t i1=0;i1<n1;i1++) {	   \
  { __VA_ARGS__ } ;			   \
  }}}

#define thread_for2d( i1, n1,i2,n2, ... )  \
  for ( uint64_t i2=0;i2<n2;i2++) {	   \
  for ( uint64_t i1=0;i1<n1;i1++) {	   \
  { __VA_ARGS__ } ;			   \
  }}

#endif


void     acceleratorInit(void);
void acceleratorReport();

template<typename decompCoordPolicy, int thrDims, int blockDims, int splitBlockSize>
struct decomp;

#define strong_inline     __attribute__((always_inline)) inline

#ifdef USE_CUDA
#  include "implementation/Accelerator_CUDA.tcc"
#  define USE_GPU
#endif
#ifdef USE_HIP
#  include "implementation/Accelerator_HIP.tcc"
#  define USE_GPU
#endif
#ifdef USE_SYCL
#  include "implementation/Accelerator_SYCL.tcc"
#  define USE_GPU
#endif

//Host-side fallback if no GPU acceleration
#ifndef USE_GPU
#  ifndef USE_OMP
#    error "No accelerator API available"
#  else
#    include "implementation/Accelerator_OMP.tcc"
#  endif
#endif

////////////////////////////////// GENERAL ///////////////////////////////////////
template<typename decompCoordPolicy, int thrDims, int blockDims, int splitBlockSize>
struct decomp{
  typedef typename decompCoordPolicy::itemPosContainerType itemPosContainerType;
  enum { totalDims = thrDims + blockDims };
  static_assert(thrDims <= 3 && blockDims < 3);
  int decomp_sizes[6];
  int dim_sizes[totalDims]; 
  size_t total_size;
  
  template<int w, bool isThread = (w < thrDims), bool isLastThread = (w == thrDims - 1)>
  struct helper;

  template<int w>
  struct helper<w, true,  false> {
    static accelerator_inline int value(itemPosContainerType pos) { return decompCoordPolicy::template itemPos<w>(pos); }
  };
  template<int w>
  struct helper<w, true,  true> {
    static accelerator_inline int value(itemPosContainerType pos) { return decompCoordPolicy::template itemPos<w>(pos) + splitBlockSize * decompCoordPolicy::template itemPos<3>(pos); }
  }; 
  template<int w, bool na>
  struct helper<w, false, na> {
    static accelerator_inline int value(itemPosContainerType pos) { return decompCoordPolicy::template itemPos<w - thrDims + 4>(pos); }
  };

  template<int w>
  static accelerator_inline int coord(itemPosContainerType pos) {
    return helper<w>::value(pos);
  }
 
  decomp(int _dim_sizes[totalDims]): decomp_sizes{1,1,1,1,1,1}, total_size(1){
    int const* lst = _dim_sizes;
    memcpy(dim_sizes, lst, totalDims*sizeof(int));

    //split the last thread dimension over that and the first dimension of the block
    memcpy(decomp_sizes, lst, thrDims * sizeof(int));
    int nb = (decomp_sizes[thrDims-1] + splitBlockSize - 1) / splitBlockSize;
    decomp_sizes[thrDims-1] = splitBlockSize;
    decomp_sizes[3] = nb;

    memcpy(decomp_sizes+4, lst + thrDims, blockDims * sizeof(int));
    for(int i=0;i<totalDims;i++)
      total_size *= lst[i];
  }
  void print(){
    std::cout << "Blocked decomposition: ";
    for(int i=0;i<6;i++)
      std::cout << decomp_sizes[i] << (i == 2 ? " || " : " ");
    std::cout << std::endl;
  }
  
};

template<typename decompCoordPolicy, int thrDims, int blockDims>
struct decomp<decompCoordPolicy, thrDims,blockDims,1>{
  typedef typename decompCoordPolicy::itemPosContainerType itemPosContainerType;
  enum { totalDims = thrDims + blockDims };
  static_assert(thrDims <= 3 && blockDims <= 3);
  int decomp_sizes[6];
  int dim_sizes[totalDims]; 
  size_t total_size;
  
  template<int w, bool isThread = (w < thrDims)>
  struct helper;

  template<int w>
  struct helper<w, true> {
    static accelerator_inline int value(itemPosContainerType pos){ return decompCoordPolicy::template itemPos<w>(pos);  }
  };
  template<int w>
  struct helper<w, false> {
    static accelerator_inline int value(itemPosContainerType pos){ return decompCoordPolicy::template itemPos<w - thrDims + 3>(pos); }
  };
  template<int w>
  static accelerator_inline int coord(itemPosContainerType pos) {  return helper<w>::value(pos);  }

  decomp(int _dim_sizes[totalDims]): decomp_sizes{1,1,1,1,1,1}, total_size(1){
    int const* lst = _dim_sizes;
    memcpy(dim_sizes, lst, totalDims*sizeof(int));    
    memcpy(decomp_sizes, lst, thrDims * sizeof(int));
    memcpy(decomp_sizes+3, lst + thrDims, blockDims * sizeof(int));
    for(int i=0;i<totalDims;i++)
      total_size *= lst[i];
  }
  void print(){
    std::cout << "Regular decomposition: ";
    for(int i=0;i<6;i++)
      std::cout << decomp_sizes[i] << (i == 2 ? " || " : " ");
    std::cout << std::endl;
  }
};

template<int _splitBlockSize = 1>
struct loopOptions{
  enum { splitBlockSize = _splitBlockSize };
  size_t shm_size;
  bool do_barrier;

  loopOptions(): shm_size(0), do_barrier(true){};

  template<int B>
  inline loopOptions<B> splitBlock(){
    loopOptions<B> out; out.shm_size = shm_size; out.do_barrier = do_barrier; return out;
  }
  inline loopOptions<_splitBlockSize> shm(size_t shm){
    shm_size = shm; return *this;
  }
  inline loopOptions<_splitBlockSize> barrier(bool doit){
    do_barrier = doit; return *this;
  }
  inline loopOptions<_splitBlockSize> normal(){
    return *this;
  } 
};


#define gen_lambda1(iter1, _num1, ...)	\
  int num1 = _num1; \
  int dims[1] = {num1};				\
  auto lambda = [=] accelerator_only (typename dimAccessorType::itemPosContainerType pos, char* shared) LAMBDA_MOD { \
    int iter1 = dimAccessorType::coord<0>(pos); \
    if(iter1 < num1){		     \
      __VA_ARGS__; \
    }\
  } 

#define accelerator_for_gen(thrDim, blockDim, options, \
			    iter1, num1, ... )	\
  { \
    static_assert(thrDim+blockDim==1);	 \
    auto opt = loopOptions<>().options ; \
    typedef decomp<DECOMP_POLICY, thrDim, blockDim, decltype(opt)::splitBlockSize> dimAccessorType; \
    gen_lambda1(iter1, num1, { __VA_ARGS__ });		\
    accelerator_for_body<thrDim,blockDim>(dims, opt, lambda); \
  };


#define gen_lambda2(iter1, _num1, iter2, _num2, ...)	\
  int num1 = _num1, num2 = _num2;			\
  int dims[2] = {num1,num2};				\
  auto lambda = [=] accelerator_only (typename dimAccessorType::itemPosContainerType pos, char* shared) LAMBDA_MOD { \
    int iter1 = dimAccessorType::coord<0>(pos); \
    int iter2 = dimAccessorType::coord<1>(pos); \
    if(iter1 < num1 && iter2 < num2){		     \
      __VA_ARGS__; \
    }\
  } 

#define accelerator_for_2d_gen(thrDim, blockDim, options,	\
			      iter1, num1, iter2, num2, ... )	\
  { \
    static_assert(thrDim+blockDim==2);	 \
    auto opt = loopOptions<>().options ; \
    typedef decomp<DECOMP_POLICY, thrDim, blockDim, decltype(opt)::splitBlockSize> dimAccessorType; \
    gen_lambda2(iter1, num1, iter2, num2, { __VA_ARGS__ });		\
    accelerator_for_body<thrDim,blockDim>(dims, opt, lambda); \
  };

#define gen_lambda3(iter1, _num1, iter2, _num2, iter3, _num3, ...) \
  int num1 = _num1, num2 = _num2, num3 = _num3;			\
  int dims[3] = {num1,num2,num3};					\
  auto lambda = [=] accelerator_only (typename dimAccessorType::itemPosContainerType pos, char* shared) LAMBDA_MOD { \
    int iter1 = dimAccessorType::coord<0>(pos); \
    int iter2 = dimAccessorType::coord<1>(pos); \
    int iter3 = dimAccessorType::coord<2>(pos);		\
    if(iter1 < num1 && iter2 < num2 && iter3 < num3){			\
      __VA_ARGS__; \
    }\
  } 

#define accelerator_for_3d_gen(thrDim, blockDim, options,	\
				iter1, num1, iter2, num2, iter3, num3, ... ) \
  { \
    static_assert(thrDim+blockDim==3);	 \
    auto opt = loopOptions<>().options ; \
    typedef decomp<DECOMP_POLICY, thrDim, blockDim, decltype(opt)::splitBlockSize> dimAccessorType; \
    gen_lambda3(iter1, num1, iter2, num2, iter3, num3, { __VA_ARGS__ }); \
    accelerator_for_body<thrDim,blockDim>(dims, opt, lambda);		\
  };

#define gen_lambda4(iter1, _num1, iter2, _num2, iter3, _num3, iter4, _num4, ...) \
  int num1 = _num1, num2 = _num2, num3 = _num3, num4 = _num4;			\
  int dims[4] = {num1,num2,num3,num4};					\
  auto lambda = [=] accelerator_only (typename dimAccessorType::itemPosContainerType pos, char* shared) LAMBDA_MOD { \
    int iter1 = dimAccessorType::coord<0>(pos); \
    int iter2 = dimAccessorType::coord<1>(pos); \
    int iter3 = dimAccessorType::coord<2>(pos);		\
    int iter4 = dimAccessorType::coord<3>(pos);		\
    if(iter1 < num1 && iter2 < num2 && iter3 < num3 && iter4 < num4){			\
      __VA_ARGS__; \
    }\
  } 

#define accelerator_for_4d_gen(thrDim, blockDim, options,	\
				iter1, num1, iter2, num2, iter3, num3, iter4, num4, ... ) \
  { \
    static_assert(thrDim+blockDim==4);	 \
    auto opt = loopOptions<>().options ; \
    typedef decomp<DECOMP_POLICY, thrDim, blockDim, decltype(opt)::splitBlockSize> dimAccessorType; \
    gen_lambda4(iter1, num1, iter2, num2, iter3, num3, iter4, num4, { __VA_ARGS__ }); \
    accelerator_for_body<thrDim,blockDim>(dims, opt, lambda);		\
  };

#define gen_lambda5(iter1, _num1, iter2, _num2, iter3, _num3, iter4, _num4, iter5, _num5, ...) \
  int num1 = _num1, num2 = _num2, num3 = _num3, num4 = _num4, num5 = _num5;			\
  int dims[5] = {num1,num2,num3,num4,num5};					\
  auto lambda = [=] accelerator_only (typename dimAccessorType::itemPosContainerType pos, char* shared) LAMBDA_MOD { \
    int iter1 = dimAccessorType::coord<0>(pos); \
    int iter2 = dimAccessorType::coord<1>(pos); \
    int iter3 = dimAccessorType::coord<2>(pos);		\
    int iter4 = dimAccessorType::coord<3>(pos);		\
    int iter5 = dimAccessorType::coord<4>(pos);				\
    if(iter1 < num1 && iter2 < num2 && iter3 < num3 && iter4 < num4 && iter5 < num5){			\
      __VA_ARGS__; \
    }\
  } 

#define accelerator_for_5d_gen(thrDim, blockDim, options,	\
			       iter1, num1, iter2, num2, iter3, num3, iter4, num4, iter5, num5, ... ) \
  { \
    static_assert(thrDim+blockDim==5);	 \
    auto opt = loopOptions<>().options ; \
    typedef decomp<DECOMP_POLICY, thrDim, blockDim, decltype(opt)::splitBlockSize> dimAccessorType; \
    gen_lambda5(iter1, num1, iter2, num2, iter3, num3, iter4, num4, iter5, num5, { __VA_ARGS__ }); \
    accelerator_for_body<thrDim,blockDim>(dims, opt, lambda);		\
  };


///////////////////////////// OLD MACROS, UNDERGOING DEPRECATION /////////////////////////////////


#define accelerator_for3dNB( iter1, num1, iter2, num2, iter3, num3, block2, ... ) \
  accelerator_for_3d_gen(1,2,splitBlock<block2>().barrier(false),iter1,num1,iter2,num2,iter3,num3, { __VA_ARGS__ })

#define accelerator_for3dNB_shm( iter1, num1, iter2, num2, iter3, num3, block2, shm_size, ... ) \
  accelerator_for_3d_gen(1,2,splitBlock<block2>().shm(shm_size).barrier(false),iter1,num1,iter2,num2,iter3,num3, { __VA_ARGS__ })

#define accelerator_for_1_3_NB_shm( iter1, num1, iter2, num2, iter3, num3, iter4, num4, block2, shm_size, ... ) \
  accelerator_for_4d_gen(1,3,splitBlock<block2>().shm(shm_size).barrier(false),iter1,num1,iter2,num2,iter3,num3,iter4,num4, { __VA_ARGS__ })

#define accelerator_for_1_3_shm( iter1, num1, iter2, num2, iter3, num3, iter4, num4, block2, shm_size, ... ) \
  accelerator_for_4d_gen(1,3,splitBlock<block2>().shm(shm_size),iter1,num1,iter2,num2,iter3,num3,iter4,num4, { __VA_ARGS__ })

#define accelerator_for_1_3_NB( iter1, num1, iter2, num2, iter3, num3, iter4, num4, block2, ... ) \
  accelerator_for_4d_gen(1,3,splitBlock<block2>().barrier(false),iter1,num1,iter2,num2,iter3,num3,iter4,num4, { __VA_ARGS__ })

#define accelerator_for_1_3( iter1, num1, iter2, num2, iter3, num3, iter4, num4, block2, ... ) \
  accelerator_for_4d_gen(1,3,splitBlock<block2>(),iter1,num1,iter2,num2,iter3,num3,iter4,num4, { __VA_ARGS__ })

#define accelerator_for_2_3_NB_shm( iter1, num1, iter2, num2, iter3, num3, iter4, num4, iter5, num5, shm_size, ... ) \
  accelerator_for_5d_gen(2,3,shm(shm_size).barrier(false),iter1,num1,iter2,num2,iter3,num3,iter4,num4,iter5,num5, { __VA_ARGS__ })

#define accelerator_for_2_3_shm( iter1, num1, iter2, num2, iter3, num3, iter4, num4, iter5, num5, shm_size, ... ) \
  accelerator_for_5d_gen(2,3,shm(shm_size),iter1,num1,iter2,num2,iter3,num3,iter4,num4,iter5,num5, { __VA_ARGS__ })

#define accelerator_for3d(iter1, num1, iter2, num2, iter3, num3, block2, ... ) \
  accelerator_for3dNB(iter1, num1, iter2, num2, iter3, num3, block2, { __VA_ARGS__ } ); \
  accelerator_barrier(dummy);

#define accelerator_for2dNB(iter1, num1, iter2, num2, block2, ... ) \
accelerator_for3dNB(iter1, num1, iter2, num2, dummy, 1, block2, { __VA_ARGS__ } );

#define accelerator_for2d(iter1, num1, iter2, num2, block2, ... ) \
  accelerator_for2dNB(iter1, num1, iter2, num2, block2, { __VA_ARGS__ } ); \
  accelerator_barrier(dummy);

#define accelerator_forNB( iter1, num1, ... ) accelerator_for3dNB( iter1, num1, dummy1, 1, dummy2,1,  32, {__VA_ARGS__} );  //note iter is over blocks

#define accelerator_for( iter, num, ... )		\
  accelerator_forNB(iter, num, { __VA_ARGS__ } );	\
  accelerator_barrier(dummy);



#define accelerator_for3d_shm(iter1, num1, iter2, num2, iter3, num3, block2, shm_size,... ) \
  accelerator_for3dNB_shm(iter1, num1, iter2, num2, iter3, num3, block2, shm_size,{ __VA_ARGS__ } ); \
  accelerator_barrier(dummy);

#define accelerator_for2dNB_shm(iter1, num1, iter2, num2, block2, shm_size, ... ) \
accelerator_for3dNB_shm(iter1, num1, iter2, num2, dummy, 1, block2, shm_size,{ __VA_ARGS__ } );

#define accelerator_for2d_shm(iter1, num1, iter2, num2, block2, shm_size, ... ) \
  accelerator_for2dNB_shm(iter1, num1, iter2, num2, block2, shm_size,{ __VA_ARGS__ } ); \
  accelerator_barrier(dummy);

#define accelerator_forNB_shm( iter1, num1, shm_size, ... ) accelerator_for3dNB_shm( iter1, num1, dummy1,1,dummy2,1,  1, shm_size,{__VA_ARGS__} );  

#define accelerator_for_shm( iter, num, shm_size, ... )		\
  accelerator_forNB_shm(iter, num, shm_size,{ __VA_ARGS__ } );	\
  accelerator_barrier(dummy);



//Because View classes cannot have non-trivial destructors, if the view requires a free it needs to be managed externally
//This class calls free on the view. It should be constructed after the view (and be destroyed before, which should happen automatically at the end of the scope)
template<typename ViewType>
struct viewDeallocator{
  ViewType &v;
  viewDeallocator(ViewType &v): v(v){}

  ~viewDeallocator(){
    v.free();
  }

  static void free(ViewType &v){ v.free(); }
};

//Create a view of a managed object and a deallocator that automatically frees it when out of scope
#define autoView(ViewName, ObjName, mode)		\
  auto ViewName = ObjName .view(mode); \
  viewDeallocator<typename std::decay<decltype(ViewName)>::type> ViewName##_d(ViewName);

//Open a HostReadWrite view 'a_v' on managed object 'a' and applies the action
//It is intended to simplify test code
//*NOT INTENDED FOR PERFORMANCE CODE!*
#define doHost(a, ... )\
  {\
    autoView(a##_v,a,HostReadWrite); \
    { __VA_ARGS__ } \
  }
//Same as above for 2 managed objects
#define doHost2(a,b, ... )			\
  {\
    autoView(a##_v,a,HostReadWrite); \
    autoView(b##_v,b,HostReadWrite); \
    { __VA_ARGS__ } \
  }
//3 managed objects...
#define doHost3(a,b,c, ... )			\
  {\
    autoView(a##_v,a,HostReadWrite); \
    autoView(b##_v,b,HostReadWrite); \
    autoView(c##_v,c,HostReadWrite); \
    { __VA_ARGS__ } \
  }
//4 managed objects...
#define doHost4(a,b,c,d, ... )			\
  {\
    autoView(a##_v,a,HostReadWrite); \
    autoView(b##_v,b,HostReadWrite); \
    autoView(c##_v,c,HostReadWrite); \
    autoView(d##_v,d,HostReadWrite); \
    { __VA_ARGS__ } \
  }


#include "implementation/Accelerator.tcc"
