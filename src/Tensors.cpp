#include <Tensors.hpp>

std::ostream & operator<<(std::ostream &os, const Vector &v){
  if(v.size(0)==0){ os << "()"; return os; }    
  os << "(" << v(0);
  for(int i=1;i<v.size(0);i++) os << ", " << v(i);
  os << ")";
  return os;  
}

void Matrix::pokeColumn(int col, const Vector &data){
    assert(data.size(0) == size0);
    for(int i=0;i<size0;i++)
      this->operator()(i,col) = data(i);
  }
  Vector Matrix::peekColumn(int col) const{
    Vector out(size0);
    for(int i=0;i<size0;i++) out(i)=this->operator()(i,col);
    return out;
  }

  Matrix Matrix::peekColumns(int col_start, int col_end) const{
    Matrix out(size0, col_end-col_start+1);
    for(int i=0;i<size0;i++){
      int jj=0;
      for(int j=col_start;j<=col_end;j++)      
	out(i,jj++)=this->operator()(i,j);
    }
    return out;
  }
  void Matrix::pokeColumns(int col_start, int col_end, const Matrix &cols){
    for(int i=0;i<size0;i++)
      for(int j=col_start;j<=col_end;j++)      
	this->operator()(i,j) = cols(i,j-col_start);
  }


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
Vector & operator+=(Vector &a, const Vector &b){
  for(int i=0;i<a.size(0);i++)
    a(i) += b(i);
  return a;
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
  return out;
}
Vector & operator*=(Vector &a, double eps){
  for(int i=0;i<a.size(0);i++)
    a(i) *= eps;
  return a;
}
