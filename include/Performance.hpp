#pragma once

struct FLOPScounter{
private:
  bool _locked;
  size_t _value;
public:
  
  FLOPScounter(): _locked(false), _value(0){}

  inline size_t add(size_t v){
    assert(!_locked);
    _value += v;
    return _value;
  }
  
  inline void lock(){ _locked = true; }
  inline bool locked() const { return _locked; }
  
  inline size_t value() const{
    assert(_locked);
    return _value;
  }
};
