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

/**
 * @brief A simple timer class
 */
class Timer{
private:
  double ttot;
  std::chrono::system_clock::time_point tp;
public:
  /**
   * @brief Reset the accumulated time to zero and (optionally) begin timing from now
   */
  inline void restart(bool start = false){
    ttot = 0;
    if(start) tp = now();
  }
  /**
   * @brief Resume a timer when paused
   */
  inline void resume(){
    tp = now();
  }
  /**
   * @brief Pause a timer, adding the time since start/resume to the accumulated time
   */
  inline void pause(){
    ttot += since(tp);
  }
  
  /**
   * @brief Get the accumulated time 
   */
  inline double time() const{
    return ttot;
  }

  /**
   * @brief Construct and (optionally) start the timer
   */  
  Timer(bool start_on_create = false){ restart(start_on_create); }
};

#define TIME(into, ...) into.resume(); __VA_ARGS__; into.pause(); 
