#pragma once

#include<vector>
#include <array>
#include <memory>
#include <cassert>
#include <iostream>

template<size_t dim>
inline size_t tensorSize(const std::array<int,dim> &dims){
  size_t out=1;
  for(int d=0;d<dim;d++) out *= dims[d];
  return out;
}
template<size_t Dim>
inline size_t compute_off(int const* coord, int const* dims){
  size_t out = *coord++; ++dims;
  for(int i=1;i<Dim;i++) out = out * (*dims++) + (*coord++);
  return out;
}
template<size_t Dim>
inline size_t compute_off(const std::array<int,Dim> &coord, const std::array<int,Dim> &dims){
  return compute_off<Dim>(coord.data(),dims.data());
}

template<int Dim>
struct Tensor{
  std::vector<double> vals;
  int _size[Dim];
    
public:
  typedef std::array<int,Dim> Dims;
  typedef std::array<int,Dim> Coord;
  
  constexpr int dimension(){ return Dim; }
  Tensor(): _size{0}{}
  Tensor(const Dims &dims, double init): vals(tensorSize(dims),init){ memcpy(_size,dims.data(),Dim*sizeof(int));  }
  Tensor(const Dims &dims, const std::vector<double> &init_vals): vals(init_vals){
    memcpy(_size,dims.data(),Dim*sizeof(int));
    assert(tensorSize(dims) == init_vals.size());
  }  
  inline double & operator()(const Coord &coord){
    size_t off = compute_off<Dim>(coord.data(), _size);
    return vals[off];
  }
  inline double operator()(const Coord &coord) const{
    size_t off = compute_off<Dim>(coord.data(), _size);
    return vals[off];
  }

  inline int size(int i) const{ return _size[i]; }

};

struct Vector{
  std::vector<double> vals;
public:
  Vector(){}
  Vector(int size1): vals(size1){}
  Vector(int size1, double init): vals(size1, init){}
  Vector(const std::vector<double> &init_vals): vals(init_vals){}    
  
  inline double & operator()(const int i){ return vals[i]; }
  inline double operator()(const int i) const{ return vals[i]; }

  inline int size(int i) const{ return vals.size(); }

  inline double const* data() const{ return vals.data(); }
  inline double* data(){ return vals.data(); }
  inline size_t data_len() const{ return vals.size(); }
};

std::ostream & operator<<(std::ostream &os, const Vector &v);

struct Matrix{
  std::vector<double> vals;
  int size0;
  int size1;
public:
  Matrix(): size0(0),size1(0){}
  Matrix(int size0, int size1): size0(size0), size1(size1), vals(size0*size1){}  
  Matrix(int size0, int size1, double init): size0(size0), size1(size1), vals(size0*size1,init){}
  Matrix(int size0, int size1, const std::vector<double> &init_vals): size0(size0), size1(size1), vals(init_vals){}    
  
  inline double & operator()(const int i, const int j){ return vals[j+size1*i]; }
  inline double operator()(const int i, const int j) const{ return vals[j+size1*i]; }

  inline int size(int i) const{ return i==0 ? size0 : size1; }

  //Insert 'data' as column 'col' of this matrix
  void pokeColumn(int col, const Vector &data);
  //Retrieve column 'col' of this matrix
  Vector peekColumn(int col) const;

  //Retrieve multiple columns as a new matrix
  Matrix peekColumns(int col_start, int col_end) const;
  //Insert multiple columns, collected as a matrix 'cols', into this matrix
  void pokeColumns(int col_start, int col_end, const Matrix &cols);
  
  inline double const* data() const{ return vals.data(); }
  inline double* data(){ return vals.data(); }
  inline size_t data_len() const{ return vals.size(); }   
};

std::ostream & operator<<(std::ostream &os, const Matrix &v);

Vector operator*(const Matrix &A, const Vector &x);

Vector operator+(const Vector &a, const Vector &b);

Vector & operator+=(Vector &a, const Vector &b);

Vector operator-(const Vector &a, const Vector &b);

Vector operator*(double eps, const Vector &b);

Vector & operator*=(Vector &a, double eps);
