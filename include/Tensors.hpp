#pragma once

#include<vector>
#include <array>
#include <memory>
#include <cassert>

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
  inline double & operator()(const Coord &coord){ return vals[compute_off<Dim>(coord.data(), size)]; }
  inline double operator()(const Coord &coord) const{ return vals[compute_off<Dim>(coord.data(), size)]; }

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

  double const* data() const{ return vals.data(); }
  double* data(){ return vals.data(); }
  size_t data_len() const{ return vals.size(); }
};

std::ostream & operator<<(std::ostream &os, const Vector &v){
  if(v.size(0)==0){ os << "()"; return os; }    
  os << "(" << v(0);
  for(int i=1;i<v.size(0);i++) os << ", " << v(i);
  os << ")";
  return os;  
}

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

  void pokeColumn(int col, const Vector &data){
    assert(data.size(0) == size0);
    for(int i=0;i<size0;i++)
      this->operator()(i,col) = data(i);
  }
  Vector peekColumn(int col) const{
    Vector out(size0);
    for(int i=0;i<size0;i++) out(i)=this->operator()(i,col);
    return out;
  }
  double const* data() const{ return vals.data(); }
  double* data(){ return vals.data(); }
  size_t data_len() const{ return vals.size(); }
   
};

std::ostream & operator<<(std::ostream &os, const Matrix &v){
  if(v.size(0)==0 || v.size(1) == 0){ os << "||"; return os; }
  for(int r=0;r<v.size(0);r++){
    os << "|" << v(r,0);
    for(int i=1;i<v.size(1);i++) os << ", " << v(r,i);
    os << "|";
    if(r != v.size(0)-1) os << std::endl;
  }
  return os;  
}


Vector operator*(const Matrix &A, const Vector &x){
  Vector out(A.size(0), 0.);
  for(int i=0;i<A.size(0);i++)
    for(int j=0;j<A.size(1);j++)
      out(i) += A(i,j) * x(j);
  return out;
}
Vector operator+(const Vector &a, const Vector &b){
  Vector out(a.size(0));
  for(int i=0;i<a.size(0);i++)
    out(i) = a(i) + b(i);
  return out;
}
Vector operator-(const Vector &a, const Vector &b){
  Vector out(a.size(0));
  for(int i=0;i<a.size(0);i++)
    out(i) = a(i) - b(i);
  return out;
}
Vector operator*(double eps, const Vector &b){
  Vector out(b.size(0));
  for(int i=0;i<b.size(0);i++)
    out(i) = eps * b(i);
}
