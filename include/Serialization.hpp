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
public:
  BinaryWriter(const std::string &filename, const Endianness end = Endianness::System);

  template<typename T, typename std::enable_if<!ISLEAF(T), int>::type = 0>
  void write(const T &v);

  template<typename T, typename U>
  void write(const std::pair<T,U> &v);
  
  template<typename T>
  void write(const std::vector<T> &v);
  
  template<typename T, int Dim>
  void write(const Tensor<T,Dim> &t);

  //write a model that is not wrapped by a loss function wrapper
  template<typename Model, typename std::enable_if<ISLEAF(Model), int>::type = 0>
  void write(const Model &model);
  
  //write a model wrapped by a loss function wrapper
  template<typename Store,typename CostFunc>
  void write(const CostFuncWrapper<Store,CostFunc> &model);

  template<typename T>
  
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

  template<typename T, typename std::enable_if<!ISLEAF(T), int>::type = 0>
  void read(T&v);

  template<typename T, typename U>
  void read(std::pair<T,U> &v);
  
  template<typename T>
  void read(std::vector<T> &v);
  
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
