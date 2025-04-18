#pragma once

#include<vector>
#include <array>
#include <memory>
#include <cassert>
#include <iostream>
#include <sstream>
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
  typedef const int * Dims;
  typedef const int * Coord;
  enum { Dimension = Dim };

#define _1D_TENSOR_ONLY template<int D=Dim, typename std::enable_if<D==1,int>::type = 0>
#define _2D_TENSOR_ONLY template<int D=Dim, typename std::enable_if<D==2,int>::type = 0>
#define _3D_TENSOR_ONLY template<int D=Dim, typename std::enable_if<D==3,int>::type = 0>
#define _4D_TENSOR_ONLY template<int D=Dim, typename std::enable_if<D==4,int>::type = 0>
  
  static constexpr int dimension(){ return Dim; }
  Tensor(): _size{0}{}
  
  Tensor(Dims dims, MemoryManager::Pool alloc_pool = MemoryManager::Pool::DevicePool): vals(tensorSize<Dim>(dims),alloc_pool){ memcpy(_size,dims,Dim*sizeof(int));  }
  Tensor(Dims dims, FloatType init, MemoryManager::Pool alloc_pool = MemoryManager::Pool::DevicePool): vals(tensorSize<Dim>(dims),init,alloc_pool){ memcpy(_size,dims,Dim*sizeof(int));  }
  
  Tensor(Dims dims, const std::vector<FloatType> &init_vals): vals(init_vals){
    memcpy(_size,dims,Dim*sizeof(int));
    assert(tensorSize<Dim>(dims) == init_vals.size());
  }  

  Tensor(Dims dims, FloatType const* init_vals): vals(tensorSize<Dim>(dims)){
    memcpy(_size,dims,Dim*sizeof(int));    
    autoView(vals_v,vals,HostWrite);
    memcpy(vals_v.data(), init_vals, vals.size()*sizeof(FloatType));
  }  
  
  _1D_TENSOR_ONLY
  Tensor(int len, MemoryManager::Pool alloc_pool = MemoryManager::Pool::DevicePool): vals(len,alloc_pool){ _size[0]=len; }
  _1D_TENSOR_ONLY
  Tensor(int len, FloatType init, MemoryManager::Pool alloc_pool = MemoryManager::Pool::DevicePool): vals(len,init,alloc_pool){ _size[0]=len; }
  _1D_TENSOR_ONLY
  Tensor(const std::vector<FloatType> &init_vals): vals(init_vals){ _size[0]=init_vals.size(); }  

  _2D_TENSOR_ONLY
  Tensor(int size0, int size1, MemoryManager::Pool alloc_pool = MemoryManager::Pool::DevicePool): vals(size0*size1,alloc_pool){ _size[0]=size0; _size[1]=size1; }
  _2D_TENSOR_ONLY
  Tensor(int size0, int size1, FloatType init, MemoryManager::Pool alloc_pool = MemoryManager::Pool::DevicePool): vals(size0*size1,init,alloc_pool){ _size[0]=size0; _size[1]=size1; }
  _2D_TENSOR_ONLY
  Tensor(int size0, int size1, const std::vector<FloatType> &init_vals): vals(init_vals){ _size[0]=size0; _size[1]=size1; assert(init_vals.size() == size0*size1); }  

  _3D_TENSOR_ONLY
  Tensor(int size0, int size1, int size2, MemoryManager::Pool alloc_pool = MemoryManager::Pool::DevicePool): vals(size0*size1*size2,alloc_pool){ _size[0]=size0; _size[1]=size1; _size[2]=size2; }
  _3D_TENSOR_ONLY
  Tensor(int size0, int size1, int size2, FloatType init, MemoryManager::Pool alloc_pool = MemoryManager::Pool::DevicePool): vals(size0*size1*size2,init,alloc_pool){ _size[0]=size0; _size[1]=size1; _size[2]=size2; }
  _3D_TENSOR_ONLY
  Tensor(int size0, int size1, int size2, const std::vector<FloatType> &init_vals): vals(init_vals){ _size[0]=size0; _size[1]=size1; _size[2]=size2; assert(init_vals.size() == size0*size1*size2); }  

  _4D_TENSOR_ONLY
  Tensor(int size0, int size1, int size2, int size3, MemoryManager::Pool alloc_pool = MemoryManager::Pool::DevicePool): vals(size0*size1*size2*size3,alloc_pool){ _size[0]=size0; _size[1]=size1; _size[2]=size2; _size[3]=size3; }
  _4D_TENSOR_ONLY
  Tensor(int size0, int size1, int size2, int size3, FloatType init, MemoryManager::Pool alloc_pool = MemoryManager::Pool::DevicePool): vals(size0*size1*size2*size3,init,alloc_pool){
    _size[0]=size0; _size[1]=size1; _size[2]=size2; _size[3]=size3; }
  _4D_TENSOR_ONLY
  Tensor(int size0, int size1, int size2, int size3, const std::vector<FloatType> &init_vals): vals(init_vals){
    _size[0]=size0; _size[1]=size1; _size[2]=size2; _size[3]=size3; assert(init_vals.size() == size0*size1*size2*size3); }  

  
  inline int const* sizeArray() const{ return _size; }
  
  inline int size(int i) const{ return _size[i]; }

  std::string sizeArrayString() const{
    std::ostringstream os; os << "(";
    for(int i=0;i<Dim-1;i++) os << this->size(i) << " ";
    os << this->size(Dim-1) << ")";
    return os.str();
  }
  
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

    _1D_TENSOR_ONLY
    accelerator_inline FloatType & operator()(int i){
      return this->Base::operator[](i);
    }
    _1D_TENSOR_ONLY
    accelerator_inline FloatType operator()(int i) const{
      return this->Base::operator[](i);
    }

    _2D_TENSOR_ONLY
    accelerator_inline FloatType & operator()(int i,int j){
      return this->Base::operator[](j+_size[1]*i);
    }
    _2D_TENSOR_ONLY
    accelerator_inline FloatType operator()(int i,int j) const{
      return this->Base::operator[](j+_size[1]*i);
    }

    _3D_TENSOR_ONLY
    accelerator_inline FloatType & operator()(int i,int j,int k){
      return this->Base::operator[](k + _size[2]*(j+_size[1]*i));
    }
    _3D_TENSOR_ONLY
    accelerator_inline FloatType operator()(int i,int j,int k) const{
      return this->Base::operator[](k + _size[2]*(j+_size[1]*i));
    }

    _4D_TENSOR_ONLY
    accelerator_inline FloatType & operator()(int i,int j,int k,int l){
      return this->Base::operator[](l+_size[3]*(k + _size[2]*(j+_size[1]*i)));
    }
    _4D_TENSOR_ONLY
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

  //Return a tensor where the last dimension contains the slice between idx_start and idx_end
  Tensor sliceLastDimension(int idx_start, int idx_end) const;

  //Insert a tensor of Dim-1 such that (*this)(i,j,k,..., idx) = ins(i,j,k,...)
  void pokeLastDimension(const Tensor<FloatType,Dim-1> &ins, const int idx);

  //Return tensor of Dim-1 such that out(i,j,k,...) = (*this)(i,j,k,..., idx)
  Tensor<FloatType,Dim-1> peekLastDimension(const int idx) const;

};

#undef _1D_TENSOR_ONLY
#undef _2D_TENSOR_ONLY
#undef _3D_TENSOR_ONLY
#undef _4D_TENSOR_ONLY

template<typename FloatType>
using Vector = Tensor<FloatType,1>;

template<typename FloatType>
using Matrix = Tensor<FloatType,2>;

//Insert 'data' as column 'col' of this matrix
template<typename FloatType>
void pokeColumn(Matrix<FloatType> &into, int col, const Vector<FloatType> &data);

//Retrieve column 'col' of this matrix
template<typename FloatType>
Vector<FloatType> peekColumn(const Matrix<FloatType> &m, int col);

//Retrieve multiple columns as a new matrix
template<typename FloatType>
Matrix<FloatType> peekColumns(const Matrix<FloatType> &m, int col_start, int col_end);

//Insert multiple columns, collected as a matrix 'cols', into this matrix
template<typename FloatType>
void pokeColumns(Matrix<FloatType> &into, int col_start, int col_end, const Matrix<FloatType> &cols);


template<typename FloatType>
std::ostream & operator<<(std::ostream &os, const Vector<FloatType> &v);

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
