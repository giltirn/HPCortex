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
  
  void push(const T&v){
    ring[off] = v;
    off = (off + 1) % ring.size();
    if(!off) filled = true;
  }
  T pop() const{
    if(!filled) throw std::runtime_error("Cannot pop from an unfilled RingBuffer as the returned value will be uninitialized");
    return ring[off];
  }
  //Return whether the RingBuffer has been populated such that pop() can be performed
  bool isFilled() const{ return filled; }
  
  size_t size() const{ return ring.size(); } 
};
    
