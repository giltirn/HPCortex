#pragma once

#include <Tensors.hpp>
#include <random>

template<typename FloatType>
bool near(FloatType a, FloatType b, FloatType rel_tol, FloatType *reldiff_p = nullptr){
  FloatType diff = a - b;
  FloatType avg = (a + b)/2.;
  FloatType reldiff;
  if(avg == 0.0){
    if(diff != 0.0) reldiff=1.0;
    else reldiff = 0.0;
  }else{
    reldiff = diff / avg;
  }
  if(reldiff_p)  *reldiff_p = reldiff;
  
  if(fabs(reldiff) > rel_tol) return false;
  else return true;
}


template<typename FloatType>
bool near(const Vector<FloatType> &a,const Vector<FloatType> &b, FloatType rel_tol, bool verbose=false){
  if(a.size(0) != b.size(0)) return false;
  autoView(a_v,a,HostRead);
  autoView(b_v,b,HostRead);
  for(size_t i=0;i<a.size(0);i++){
    FloatType reldiff;
    bool nr = near(a_v(i),b_v(i),rel_tol,&reldiff);
    if(!nr){
      if(verbose) std::cout << i << " a:" << a_v(i) << " b:" << b_v(i) << " rel.diff:" << reldiff << std::endl;
      return false;
    }
  }
  return true;
}


template<typename FloatType>
bool near(const Matrix<FloatType> &a,const Matrix<FloatType> &b, FloatType rel_tol, bool verbose=false){
  if(a.size(0) != b.size(0) || a.size(1) != b.size(1)) return false;
  autoView(a_v,a,HostRead);
  autoView(b_v,b,HostRead);
  for(size_t i=0;i<a.size(0);i++){
    for(size_t j=0;j<a.size(1);j++){
    
      FloatType reldiff;
      bool nr = near(a_v(i,j),b_v(i,j),rel_tol,&reldiff);
      if(!nr){
	if(verbose) std::cout << i << " " << j << " a:" << a_v(i,j) << " b:" << b_v(i,j) << " rel.diff:" << reldiff << std::endl;
	return false;
      }
    }
  }
  return true;
}


template<typename FloatType>
bool abs_near(FloatType a, FloatType b, FloatType abs_tol, FloatType *absdiff_p = nullptr){
  FloatType absdiff = fabs(a - b);
  if(absdiff_p) *absdiff_p = absdiff;
  if(absdiff > abs_tol) return false;
  else return true;
}


template<typename FloatType>
bool abs_near(const Matrix<FloatType> &a,const Matrix<FloatType> &b, FloatType abs_tol, bool verbose=false){
  if(a.size(0) != b.size(0) || a.size(1) != b.size(1)) return false;
  autoView(a_v,a,HostRead);
  autoView(b_v,b,HostRead);
  for(size_t i=0;i<a.size(0);i++){
    for(size_t j=0;j<a.size(1);j++){
    
      FloatType absdiff;
      bool nr = abs_near(a_v(i,j),b_v(i,j),abs_tol,&absdiff);
      if(!nr){
	if(verbose) std::cout << i << " " << j << " a:" << a_v(i,j) << " b:" << b_v(i,j) << " abs.diff:" << absdiff << std::endl;
	return false;
      }
    }
  }
  return true;
}

template<typename FloatType, int Dim>
bool abs_near(const Tensor<FloatType,Dim> &a,const Tensor<FloatType,Dim> &b, FloatType abs_tol, bool verbose=false){
  int const* a_sz = a.sizeArray();
  int const* b_sz = b.sizeArray();
  for(int d=0;d<Dim;d++) if(a_sz[d] != b_sz[d]) return false;
  
  autoView(a_v,a,HostRead);
  autoView(b_v,b,HostRead);
  for(size_t v = 0; v < a_v.data_len(); v++){
    FloatType absdiff;
    FloatType aa = a_v.data()[v];
    FloatType bb = b_v.data()[v];
    
    bool nr = abs_near(aa,bb,abs_tol,&absdiff);
    if(!nr){
      if(verbose){     
	int coord[Dim];    
	tensorOffsetUnmap<Dim>(coord,a_sz,v);
	for(int d=0;d<Dim;d++)
	  std::cout << coord[d] << " ";
	std::cout << "a:" << aa << " b:" << bb << " abs.diff:" << absdiff << std::endl;
      }
      return false;
    }
  }
  return true;
}



template<typename FloatType, typename RNG>
void random(Matrix<FloatType> &m, RNG &rng){
  std::uniform_real_distribution<FloatType> dist(-1.0, 1.0);
  autoView(m_v,m,HostWrite);
  for(int i=0;i<m.size(0);i++)
    for(int j=0;j<m.size(1);j++)
      m_v(i,j) = dist(rng);
}
template<typename FloatType, typename RNG>
void random(Vector<FloatType> &m, RNG &rng){
  std::uniform_real_distribution<FloatType> dist(-1.0, 1.0);
  autoView(m_v,m,HostWrite);
  for(int i=0;i<m.size(0);i++)
    m_v(i) = dist(rng);
}    
template<typename FloatType, int Dim, typename RNG>
void random(Tensor<FloatType,Dim> &m, RNG &rng){
  std::uniform_real_distribution<FloatType> dist(-1.0, 1.0);
  autoView(m_v,m,HostWrite);
  int const* dims = m.sizeArray();
  size_t sz = tensorSize<Dim>(dims);
  for(size_t i=0; i<sz; i++){
    int coord[Dim];
    tensorOffsetUnmap<Dim>(coord, dims, i);
    m_v(coord) = dist(rng);
  }
}


template<typename Op, typename PreOp>
void benchmark(double &mean, double &std, int nrpt, int nwarmup, const Op &op, const PreOp &preop){
  auto t = now();
  for(int i=0;i<nwarmup;i++){
    preop();
    op();
  }
  double twarm = since(t);
  //std::cout << "Warmup " << twarm << std::endl;
  
  mean = std = 0.;
  for(int i=0;i<nrpt;i++){
    preop();
    t = now();
    op();
    double dt = since(t);
    mean += dt;
    std += dt*dt;

    //std::cout << i << " " << dt  << std::endl;
  }
  mean /= nrpt;
  std = sqrt( std/nrpt - mean*mean ); 
}
