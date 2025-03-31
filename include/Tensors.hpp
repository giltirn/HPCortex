#pragma once

#include<vector>
#include <array>
#include <memory>
#include <cassert>
#include <iostream>
#include <ManagedArray.hpp>

template<size_t Dim>
accelerator_inline size_t tensorSize(int const* dims){
  size_t out=1;
#pragma unroll
  for(int d=0;d<Dim;d++) out *= dims[d];
  return out;
}
template<size_t Dim>
accelerator_inline size_t tensorOffset(int const* coord, int const* dims){
  size_t out = *coord++; ++dims;
#pragma unroll
  for(int i=1;i<Dim;i++) out = out * (*dims++) + (*coord++);
  return out;
}

template<size_t Dim>
accelerator_inline void tensorOffsetUnmap(int * coord, int const* dims, size_t offset){
  size_t rem = offset;
#pragma unroll
  for(int i=Dim-1;i>=0;i--){
    coord[i] = rem % dims[i];
    rem /= dims[i];
  }
}
     
 

template<typename _FloatType, int Dim>
struct Tensor{
public:
  typedef _FloatType FloatType;
private:
  ManagedArray<FloatType> vals;
  int _size[Dim];    
public:
  typedef int * Dims;
  typedef int * Coord;
  
  constexpr int dimension(){ return Dim; }
  Tensor(): _size{0}{}
  Tensor(const Dims dims, MemoryManager::Pool alloc_pool = MemoryManager::Pool::DevicePool): vals(tensorSize<Dim>(dims),alloc_pool){ memcpy(_size,dims,Dim*sizeof(int));  }
  Tensor(const Dims dims, FloatType init, MemoryManager::Pool alloc_pool = MemoryManager::Pool::DevicePool): vals(tensorSize<Dim>(dims),init,alloc_pool){ memcpy(_size,dims,Dim*sizeof(int));  }
  
  Tensor(const Dims dims, const std::vector<FloatType> &init_vals): vals(init_vals){
    memcpy(_size,dims,Dim*sizeof(int));
    assert(tensorSize<Dim>(dims) == init_vals.size());
  }  

  inline int size(int i) const{ return _size[i]; }

  class View: private ManagedArray<FloatType>::View{
    typedef typename ManagedArray<FloatType>::View Base;
    int* _size;
    bool is_device_ptr;
  public:
    inline View(ViewMode mode, const Tensor<FloatType,Dim> &parent): Base(mode, parent.vals){
      if(mode == DeviceRead || mode == DeviceWrite || mode == DeviceReadWrite){
	_size = (int*)acceleratorAllocDevice(Dim*sizeof(int));
	acceleratorCopyToDevice(_size,parent._size,Dim*sizeof(int));
	is_device_ptr = true;
      }else{
	_size = (int*)malloc(Dim*sizeof(int));
	memcpy(_size,parent._size,Dim*sizeof(int));
	is_device_ptr = false;
      }	       
    }

    inline void free(){
      if(is_device_ptr) acceleratorFreeDevice(_size);
      else ::free(_size);      
      return this->Base::free();
    }
    
    accelerator_inline FloatType & operator()(const Coord coord){
      return this->Base::operator[](tensorOffset<Dim>(coord, _size));
    }
    accelerator_inline FloatType operator()(const Coord coord) const{
      return this->Base::operator[](tensorOffset<Dim>(coord, _size));
    }

    //1D tensor only
    template<int D=Dim, typename std::enable_if<D==1,int>::type = 0>
    accelerator_inline FloatType & operator()(int i){
      return this->Base::operator[](i);
    }
    template<int D=Dim, typename std::enable_if<D==1,int>::type = 0>
    accelerator_inline FloatType operator()(int i) const{
      return this->Base::operator[](i);
    }

    //2D tensor only
    template<int D=Dim, typename std::enable_if<D==2,int>::type = 0>
    accelerator_inline FloatType & operator()(int i,int j){
      return this->Base::operator[](j+_size[1]*i);
    }
    template<int D=Dim, typename std::enable_if<D==2,int>::type = 0>
    accelerator_inline FloatType operator()(int i,int j) const{
      return this->Base::operator[](j+_size[1]*i);
    }

    //3D tensor only
    template<int D=Dim, typename std::enable_if<D==3,int>::type = 0>
    accelerator_inline FloatType & operator()(int i,int j,int k){
      return this->Base::operator[](k + _size[2]*(j+_size[1]*i));
    }
    template<int D=Dim, typename std::enable_if<D==3,int>::type = 0>
    accelerator_inline FloatType operator()(int i,int j,int k) const{
      return this->Base::operator[](k + _size[2]*(j+_size[1]*i));
    }
    
    //4D tensor only
    template<int D=Dim, typename std::enable_if<D==4,int>::type = 0>
    accelerator_inline FloatType & operator()(int i,int j,int k,int l){
      return this->Base::operator[](l+_size[3]*(k + _size[2]*(j+_size[1]*i)));
    }
    template<int D=Dim, typename std::enable_if<D==4,int>::type = 0>
    accelerator_inline FloatType operator()(int i,int j,int k,int l) const{
      return this->Base::operator[](l+_size[3]*(k + _size[2]*(j+_size[1]*i)));
    }
    
    accelerator_inline FloatType const* data() const{ return this->Base::data(); }
    accelerator_inline FloatType* data(){ return this->Base::data(); }
    accelerator_inline size_t data_len() const{ return this->Base::size(); }

    accelerator_inline size_t size(int i) const{ return _size[i]; }
  };

  View view(ViewMode mode) const{
    return View(mode, *this);
  }
  //Lock the associated memory, preventing eviction
  inline void lock() const{ vals.lock(); }
  inline void unlock() const{ vals.unlock(); }  
};

template<typename _FloatType>
struct Vector{
public:
  typedef _FloatType FloatType;
private:
  ManagedArray<FloatType> vals;
public:
  Vector(){}
  Vector(int size1, MemoryManager::Pool alloc_pool = MemoryManager::Pool::DevicePool): vals(size1,alloc_pool){}
  Vector(int size1, FloatType init, MemoryManager::Pool alloc_pool = MemoryManager::Pool::DevicePool): vals(size1, init, alloc_pool){}
  Vector(const std::vector<FloatType> &init_vals): vals(init_vals){}
  
  constexpr int dimension(){ return 1; }
  inline size_t size(int i) const{ return vals.size(); }

  class View: private ManagedArray<FloatType>::View{
    typedef typename ManagedArray<FloatType>::View Base;
  public:
    inline View(ViewMode mode, const Vector<FloatType> &parent): Base(mode, parent.vals){}

    inline void free(){ return this->Base::free(); }
    
    accelerator_inline FloatType & operator()(const int i){ return this->Base::operator[](i); }
    accelerator_inline FloatType operator()(const int i) const{ return this->Base::operator[](i); }

    accelerator_inline FloatType const* data() const{ return this->Base::data(); }
    accelerator_inline FloatType* data(){ return this->Base::data(); }
    accelerator_inline size_t data_len() const{ return this->Base::size(); }

    accelerator_inline size_t size(int i) const{ return data_len(); }
  };

  View view(ViewMode mode) const{
    return View(mode, *this);
  }

  //Lock the associated memory, preventing eviction
  inline void lock() const{ vals.lock(); }
  inline void unlock() const{ vals.unlock(); }  
};

template<typename FloatType>
std::ostream & operator<<(std::ostream &os, const Vector<FloatType> &v);

template<typename _FloatType>
struct Matrix{
public:
  typedef _FloatType FloatType;
private:
private:
  ManagedArray<FloatType> vals;
  int size0;
  int size1;
public:
  Matrix(): size0(0),size1(0){}
  Matrix(int size0, int size1, MemoryManager::Pool alloc_pool = MemoryManager::Pool::DevicePool): size0(size0), size1(size1), vals(size0*size1,alloc_pool){}  
  Matrix(int size0, int size1, FloatType init, MemoryManager::Pool alloc_pool = MemoryManager::Pool::DevicePool):
    size0(size0), size1(size1), vals(size0*size1,init,alloc_pool){}
  Matrix(int size0, int size1, const std::vector<FloatType> &init_vals): size0(size0), size1(size1), vals(init_vals){}    

  constexpr int dimension(){ return 2; }
  inline int size(int i) const{ return i==0 ? size0 : size1; }
  
  class View: private ManagedArray<FloatType>::View{
    typedef typename ManagedArray<FloatType>::View Base;
    int size0;
    int size1;
  public:
    inline View(ViewMode mode, const Matrix<FloatType> &parent): Base(mode, parent.vals), size0(parent.size0),size1(parent.size1){}

    inline void free(){ return this->Base::free(); }
    
    accelerator_inline FloatType & operator()(const int i, const int j){ return this->Base::operator[](j+size1*i); }
    accelerator_inline FloatType operator()(const int i, const int j) const{ this->Base::operator[](j+size1*i); }
    
    accelerator_inline FloatType const* data() const{ return this->Base::data(); }
    accelerator_inline FloatType* data(){ return this->Base::data(); }
    accelerator_inline size_t data_len() const{ return this->Base::size(); }

    accelerator_inline size_t size(int i) const{ return i==0 ? size0 : size1; }
  };

  View view(ViewMode mode) const{
    return View(mode, *this);
  }

  //Insert 'data' as column 'col' of this matrix
  void pokeColumn(int col, const Vector<FloatType> &data);
  //Retrieve column 'col' of this matrix
  Vector<FloatType> peekColumn(int col) const;

  //Retrieve multiple columns as a new matrix
  Matrix peekColumns(int col_start, int col_end) const;
  //Insert multiple columns, collected as a matrix 'cols', into this matrix
  void pokeColumns(int col_start, int col_end, const Matrix &cols);
  
  //Lock the associated memory, preventing eviction
  inline void lock() const{ vals.lock(); }
  inline void unlock() const{ vals.unlock(); }  
};

template<typename FloatType>
std::ostream & operator<<(std::ostream &os, const Matrix<FloatType> &v);

template<typename FloatType>
Matrix<FloatType> & operator+=(Matrix<FloatType> &a, const Matrix<FloatType> &b);

template<typename FloatType>
Matrix<FloatType> operator+(const Matrix<FloatType> &a, const Matrix<FloatType> &b);

template<typename FloatType>
Vector<FloatType> operator*(const Matrix<FloatType> &A, const Vector<FloatType> &x);

template<typename FloatType>
Vector<FloatType> operator+(const Vector<FloatType> &a, const Vector<FloatType> &b);

template<typename FloatType>
Vector<FloatType> & operator+=(Vector<FloatType> &a, const Vector<FloatType> &b);

template<typename FloatType>
Vector<FloatType> operator-(const Vector<FloatType> &a, const Vector<FloatType> &b);

template<typename FloatType>
Vector<FloatType> operator*(FloatType eps, const Vector<FloatType> &b);

template<typename FloatType>
Vector<FloatType> & operator*=(Vector<FloatType> &a, FloatType eps);

#include "implementation/Tensors.tcc"

// #ifndef TENSORS_EXTERN_TEMPLATE_INST
// #define SS extern
// #else
// #define SS
// #endif
// SS template class Matrix<float>;
// SS template class Matrix<double>;
// SS template class Vector<float>;
// SS template class Vector<double>;
// #undef SS
