#pragma once
#include <Tensors.hpp>
#include <layers/LayerCommon.hpp>
#include <LossFunctions.hpp>
#include <cstdint>

//return 0 if little-endian, 1 for big-endian
enum class Endianness { Big, Little, System };
std::string toString(const Endianness e);

//get the endianness of the system
Endianness endianness();

//internal lookup table for bit reverse
uint8_t BitReverseTable256(size_t i);

//flip the bits in the (POD) data input
template<typename T>
inline T bitReverse(T in);

//A class to write tensors and models in a binary format
class BinaryWriter{
  std::ofstream of;
  bool do_flip;
  template<typename T>
  void writeValue(T v){
    T tmp = do_flip ? bitReverse(v) : v;
    of.write((char const*)&tmp,sizeof(T));    
  }    
public:
  BinaryWriter(const std::string &filename, const Endianness end = Endianness::System);

  template<typename T, int Dim>
  void write(const Tensor<T,Dim> &t);

  //write a model that is not wrapped by a loss function wrapper
  template<typename Model, typename std::enable_if<ISLEAF(Model), int>::type = 0>
  void write(const Model &model);
  
  //write a model wrapped by a loss function wrapper
  template<typename Store,typename CostFunc>
  void write(const CostFuncWrapper<Store,CostFunc> &model);
    
  inline void close(){
    of.close();
  }
};

class BinaryReader{
  std::ifstream of;
  bool do_flip;
  template<typename T>
  inline T readValue(){
    T tmp; of.read((char*)&tmp, sizeof(T)); assert(of.good());
    return do_flip ? bitReverse(tmp) : tmp;
  }    
public:
  BinaryReader(const std::string &filename);

  //Requires tensor to have appropriate size
  template<typename T, int Dim>
  void read(Tensor<T,Dim> &t);
  
  //read a model that is not wrapped by a loss function wrapper
  template<typename Model, typename std::enable_if<ISLEAF(Model), int>::type = 0>
  void read(Model &model);
  
  //read a model wrapped by a loss function wrapper
  //write a model wrapped by a loss function wrapper
  template<typename Store,typename CostFunc>
  void read(CostFuncWrapper<Store,CostFunc> &model);

  inline void close(){
    of.close();
  }

};

#include "implementation/Serialization.tcc"
