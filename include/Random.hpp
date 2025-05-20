#pragma once
#include<random>
#include "Tensors.hpp"

typedef std::mt19937 GlobalRNGtype;
constexpr size_t default_seed = 1234;

GlobalRNGtype & globalRNG(){
  static GlobalRNGtype rng(1234);
  return rng;
}
inline void reseedGlobalRNG(size_t seed){
  globalRNG() = GlobalRNGtype(seed);
}

template<typename FloatType, int Dim, typename Dist, typename RNG>
void random(Tensor<FloatType,Dim> &m, Dist &dist, RNG &rng){
  if(m.data_len() == 0) return;
  autoView(m_v,m,HostWrite);
  for(size_t i=0;i<m.data_len();i++)
    m_v.data()[i] = dist(rng);
}
  
template<typename FloatType, int Dim, typename RNG>
void uniformRandom(Tensor<FloatType,Dim> &m, RNG &rng, FloatType min = FloatType(-1.0), FloatType max = FloatType(1.0) ){
  std::uniform_real_distribution<FloatType> dist(min, max);
  random(m,dist,rng);
}
template<typename FloatType, int Dim>
inline void uniformRandom(Tensor<FloatType,Dim> &m, FloatType min = FloatType(-1.0), FloatType max = FloatType(1.0) ){ return uniformRandom(m, globalRNG(), min,max); }


//https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
//https://fluxml.ai/Flux.jl/stable/reference/utilities/#Flux.glorot_uniform
template<typename FloatType, typename RNG>
void glorotUniformRandom(Matrix<FloatType> &m, RNG &rng, FloatType gain = FloatType(1.0)){
  FloatType lim = gain * sqrt( FloatType(6.0)/(m.size(0) + m.size(1) ) );
  uniformRandom(m, rng, -lim,lim);
}
template<typename FloatType>
inline void glorotUniformRandom(Matrix<FloatType> &m, FloatType gain = FloatType(1.0)){
  return glorotUniformRandom(m,globalRNG(),gain);
}
