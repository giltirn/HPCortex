#pragma once

#include <Tensors.hpp>

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
