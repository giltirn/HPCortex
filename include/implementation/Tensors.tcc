template<typename FloatType>
std::ostream & operator<<(std::ostream &os, const Vector<FloatType> &v){
  if(v.size(0)==0){ os << "()"; return os; }    
  os << "(" << v(0);
  for(int i=1;i<v.size(0);i++) os << ", " << v(i);
  os << ")";
  return os;  
}

template<typename FloatType>
void Matrix<FloatType>::pokeColumn(int col, const Vector<FloatType> &data){
  assert(data.size(0) == size0);
  for(int i=0;i<size0;i++)
    this->operator()(i,col) = data(i);
}

template<typename FloatType>
Vector<FloatType> Matrix<FloatType>::peekColumn(int col) const{
  Vector<FloatType> out(size0);
  for(int i=0;i<size0;i++) out(i)=this->operator()(i,col);
  return out;
}

template<typename FloatType>
Matrix<FloatType> Matrix<FloatType>::peekColumns(int col_start, int col_end) const{
  Matrix<FloatType> out(size0, col_end-col_start+1);
  for(int i=0;i<size0;i++){
    int jj=0;
    for(int j=col_start;j<=col_end;j++)      
      out(i,jj++)=this->operator()(i,j);
  }
  return out;
}

template<typename FloatType>
void Matrix<FloatType>::pokeColumns(int col_start, int col_end, const Matrix<FloatType> &cols){
  for(int i=0;i<size0;i++)
    for(int j=col_start;j<=col_end;j++)      
      this->operator()(i,j) = cols(i,j-col_start);
}

template<typename FloatType>
std::ostream & operator<<(std::ostream &os, const Matrix<FloatType> &v){
  if(v.size(0)==0 || v.size(1) == 0){ os << "||"; return os; }
  for(int r=0;r<v.size(0);r++){
    os << "|" << v(r,0);
    for(int i=1;i<v.size(1);i++) os << ", " << v(r,i);
    os << "|";
    if(r != v.size(0)-1) os << std::endl;
  }
  return os;  
}

template<typename FloatType>
Vector<FloatType> operator*(const Matrix<FloatType> &A, const Vector<FloatType> &x){
  Vector<FloatType> out(A.size(0), 0.);
  for(int i=0;i<A.size(0);i++)
    for(int j=0;j<A.size(1);j++)
      out(i) += A(i,j) * x(j);
  return out;
}

template<typename FloatType>
Vector<FloatType> operator+(const Vector<FloatType> &a, const Vector<FloatType> &b){
  Vector<FloatType> out(a.size(0));
  for(int i=0;i<a.size(0);i++)
    out(i) = a(i) + b(i);
  return out;
}

template<typename FloatType>
Vector<FloatType> & operator+=(Vector<FloatType> &a, const Vector<FloatType> &b){
  for(int i=0;i<a.size(0);i++)
    a(i) += b(i);
  return a;
}

template<typename FloatType>
Vector<FloatType> operator-(const Vector<FloatType> &a, const Vector<FloatType> &b){
  Vector<FloatType> out(a.size(0));
  for(int i=0;i<a.size(0);i++)
    out(i) = a(i) - b(i);
  return out;
}

template<typename FloatType>
Vector<FloatType> operator*(FloatType eps, const Vector<FloatType> &b){
  Vector<FloatType> out(b.size(0));
  for(int i=0;i<b.size(0);i++)
    out(i) = eps * b(i);
  return out;
}

template<typename FloatType>
Vector<FloatType> & operator*=(Vector<FloatType> &a, FloatType eps){
  for(int i=0;i<a.size(0);i++)
    a(i) *= eps;
  return a;
}
