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

  T release(){
    return std::move(v);
  }
  
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

  T & release(){
    return v;
  }  
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

template<typename RefType, typename OfType>
using enable_if_fwd_ref = typename std::enable_if< std::is_same<typename std::decay<RefType>::type, OfType>::value, int>::type;

template<typename RefType1, typename RefType2, typename OfType>
using enable_if_fwd_ref2 = typename std::enable_if< std::is_same<typename std::decay<RefType1>::type, OfType>::value && std::is_same<typename std::decay<RefType2>::type, OfType>::value, int>::type;

template<typename RefType1, typename RefType2, typename RefType3, typename OfType>
using enable_if_fwd_ref3 = typename std::enable_if< std::is_same<typename std::decay<RefType1>::type, OfType>::value && std::is_same<typename std::decay<RefType2>::type, OfType>::value && std::is_same<typename std::decay<RefType3>::type, OfType>::value, int>::type;


/**
 * @brief For an input forwarding reference BASETYPE&& ${BASENM}_ref, create a reference container ${BASENM}_con and a const reference to the object ${BASENM}
 */
#define INPUT_CON(BASENM, BASETYPE)					\
  DDST(BASENM##_ref) BASENM##_con(std::forward<BASETYPE>(BASENM##_ref)); \
  const BASETYPE &BASENM = BASENM##_con.v;
