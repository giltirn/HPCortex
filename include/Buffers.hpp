#pragma once
#include<vector>

//Assume push, pop cycle
//push appends a new entry, pop outputs the oldest
//e.g buf size 3

//Iter   Buf    Push item    Push idx     Pop item     Pop idx
// 0     0--     0              0            -            -
// 1     01-     1              1            -            -
// 2     012     2              2            0            0
// 3     312     3              0            1            1
// 4     342     4              1            2            3
// 5     345     5              2            3            0

//so we push to idx off=0 1 2 0 1 2 = iter % 3  and pop from (off+1) % 3
template<typename T>
class RingBuffer{
  std::vector<T> ring;
  size_t off;
  bool filled;
public:
  RingBuffer(size_t size): ring(size), off(0), filled(false){}
  RingBuffer(): RingBuffer(1){} 

  void resize(size_t size){
    ring.resize(size);
    off=0;
    filled=false;
  }
  
  inline void push(T&& v){
    ring[off] = std::move(v);
    off = (off + 1) % ring.size();
    if(!off) filled = true;
  }
  /**
   * @brief Push a copy
   */
  inline void push(const T &v){
    push(T(v));
  }
  
  inline T pop(){
    if(!filled) throw std::runtime_error("Cannot pop from an unfilled RingBuffer as the returned value will be uninitialized");
    return std::move(ring[off]);
  }
  //Return whether the RingBuffer has been populated such that pop() can be performed
  bool isFilled() const{ return filled; }
  
  size_t size() const{ return ring.size(); }

  //This function is used to provide a valid object even if the buffer is not completely filled
  const T &latest() const{    
    return ring[  (off - 1 + ring.size()) % ring.size() ];
  }
    
};
 
template<typename T>
class BufferSingle{
  T val;
  bool valid;
public:
  BufferSingle(size_t size): valid(false){ resize(size); }
  BufferSingle(): BufferSingle(1){}
  void resize(size_t size){
    if(size != 1) throw std::runtime_error("BufferSingle cannot accommodate more than 1 entry");
  }
  inline void push(T&& v){
    val = std::move(v);
    valid=true;
  }
  /**
   * @brief Push a copy
   */
  inline void push(const T &v){
    push(T(v));
  }
  
  inline T pop(){
    T ret(std::move(val));
    valid = false;
    return ret;
  }
  inline bool isFilled() const{ return valid; }
  
  inline size_t size() const{ return 1; }

  inline const T &latest() const{
    if(!valid) throw std::runtime_error("Cannot call latest when the buffer is depopulated");
    return val;
  }
};

//This buffer has n slots. It is expected that it will be used in a cycle where all n slots will first be filled then all n drained
template<typename T>
class FillEmptyRingBuffer{
  std::vector<T> ring;
  size_t push_off;
  size_t pop_off;
  int cycle; //0 fill, 1 empty
    
public:
  FillEmptyRingBuffer(size_t size): ring(size), push_off(0), pop_off(0), cycle(0){}
  FillEmptyRingBuffer(): FillEmptyRingBuffer(1){} 

  void resize(size_t size){
    ring.resize(size);
    push_off = pop_off=0;
    cycle=0;
  }
  
  inline void push(T&& v){
    if(cycle == 1) throw std::runtime_error("Cannot push to FillEmptyRingBuffer when in a drain cycle");    
    ring[push_off] = std::move(v);
    push_off = (push_off + 1) % ring.size();

    //when it cycles back to 0 we switch to the drain cycle
    if(!push_off) cycle = 1;
  }
  /**
   * @brief Push a copy
   */
  inline void push(const T &v){
    push(T(v));
  }
 
  inline T pop(){
    if(cycle == 0) throw std::runtime_error("Cannot pop FillEmptyRingBuffer when in a fill cycle");    
    T ret(std::move(ring[pop_off]));
    pop_off =  (pop_off + 1) % ring.size();

    //when it cycles back to 1 we switch to the fill cycle
    if(!pop_off) cycle = 0;    
    return ret;
  }
  //Return whether the we are on the drain cycle
  bool isFilled() const{ return cycle == 1; }
  
  size_t size() const{ return ring.size(); }

  //This function is used to provide a valid object even if the buffer is not completely filled
  const T &latest() const{    
    return ring[  (push_off - 1 + ring.size()) % ring.size() ];
  }
    
};


enum EnableDeriv {DerivNo=0, DerivYes=1};
