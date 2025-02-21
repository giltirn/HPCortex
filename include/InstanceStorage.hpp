#pragma once

struct StorageTag{};

//A structure to store instances based on input rvalue references
template<typename T>
struct LeafStore{
  T v;
  typedef StorageTag tag;
  typedef T type;
  
  LeafStore(T && v): v(std::move(v)){
  }
  LeafStore(const LeafStore &r) = delete;
  LeafStore(LeafStore &&r): v(std::move(r.v)){}
  
};
//A structure to store instance references based on input lvalue references
template<typename T>
struct LeafRef{
  T &v;
  typedef StorageTag tag;
  typedef T type;
  
  LeafRef(T &v): v(v){
  }
  LeafRef(const LeafRef &r) = delete;
  LeafRef(LeafRef &&r): v(r.v){}

};

//Deduce the appropriate storage type given a reference type
template<typename T>
struct deduceStorage{};

template<typename T>
struct deduceStorage<T&>{
  typedef LeafRef<T> type;
};

template<typename T>
struct deduceStorage<T&&>{
  typedef LeafStore<T> type;
};

#define DDST(a) typename deduceStorage<decltype(a)>::type
#define ISSTORAGE(a) std::is_same<typename std::decay<a>::type::tag,StorageTag>::value
