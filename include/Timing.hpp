#pragma once
#include<chrono>

//Get the epoch time in us
inline std::chrono::system_clock::time_point now(){ return std::chrono::system_clock::now(); }

inline size_t usSinceEpoch(){
  auto _now = now();
  auto duration = _now.time_since_epoch();
  return std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
}

//Microseconds resolution time since "when"
inline size_t usCountSince(const std::chrono::system_clock::time_point &when){
  auto dt = now() - when;
  return std::chrono::duration_cast<std::chrono::microseconds>(dt).count();
}

//Decimal time duration in seconds, accurate to us
inline double since(const std::chrono::system_clock::time_point &when){
  return double(usCountSince(when)) / 1e6;
}



