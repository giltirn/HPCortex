template<typename FloatType>
std::ostream & operator<<(std::ostream &os, const Vector<FloatType> &v){
  autoView(vv,v,HostRead);
  if(vv.size(0)==0){ os << "()"; return os; }    
  os << "(" << vv(0);
  for(int i=1;i<vv.size(0);i++) os << ", " << vv(i);
  os << ")";
  return os;  
}

template<typename FloatType>
void Matrix<FloatType>::pokeColumn(int col, const Vector<FloatType> &data){
  assert(data.size(0) == size0);
  autoView(data_v,data,HostRead);
  autoView(t_v,(*this),HostWrite);
  for(int i=0;i<size0;i++)
    t_v(i,col) = data_v(i);
}

template<typename FloatType>
Vector<FloatType> Matrix<FloatType>::peekColumn(int col) const{
  Vector<FloatType> out(size0);
  autoView(out_v,out,HostWrite);
  autoView(t_v,(*this),HostRead);
  for(int i=0;i<size0;i++) out_v(i)=t_v(i,col);
  return out;
}

template<typename FloatType>
Matrix<FloatType> Matrix<FloatType>::peekColumns(int col_start, int col_end) const{
  Matrix<FloatType> out(size0, col_end-col_start+1);
  autoView(out_v,out,HostWrite);
  autoView(t_v,(*this),HostRead);  
  for(int i=0;i<size0;i++){
    int jj=0;
    for(int j=col_start;j<=col_end;j++)      
      out_v(i,jj++)=t_v(i,j);
  }
  return out;
}

template<typename FloatType>
void Matrix<FloatType>::pokeColumns(int col_start, int col_end, const Matrix<FloatType> &cols){
  autoView(cols_v,cols,HostRead);
  autoView(t_v,(*this),HostWrite);
  
  for(int i=0;i<size0;i++)
    for(int j=col_start;j<=col_end;j++)      
      t_v(i,j) = cols_v(i,j-col_start);
}

template<typename FloatType>
std::ostream & operator<<(std::ostream &os, const Matrix<FloatType> &v){
  if(v.size(0)==0 || v.size(1) == 0){ os << "||"; return os; }
  autoView(v_v,v,HostRead); 
  for(int r=0;r<v.size(0);r++){
    os << "|" << v_v(r,0);
    for(int i=1;i<v.size(1);i++) os << ", " << v_v(r,i);
    os << "|";
    if(r != v.size(0)-1) os << std::endl;
  }
  return os;  
}

template<typename FloatType>
Vector<FloatType> operator*(const Matrix<FloatType> &A, const Vector<FloatType> &x){
  Vector<FloatType> out(A.size(0), 0., MemoryManager::Pool::HostPool);
  autoView(x_v,x,HostRead);
  autoView(out_v,out,HostReadWrite);
  autoView(A_v,A,HostRead);
  
  for(int i=0;i<A.size(0);i++)
    for(int j=0;j<A.size(1);j++)
      out_v(i) += A_v(i,j) * x_v(j);
  return out;
}

template<typename FloatType>
Vector<FloatType> operator+(const Vector<FloatType> &a, const Vector<FloatType> &b){
  Vector<FloatType> out(a.size(0), MemoryManager::Pool::HostPool);
  autoView(out_v,out,HostWrite);
  autoView(a_v,a,HostRead);
  autoView(b_v,b,HostRead);
  for(int i=0;i<a.size(0);i++)
    out_v(i) = a_v(i) + b_v(i);
  return out;
}

template<typename FloatType>
Vector<FloatType> & operator+=(Vector<FloatType> &a, const Vector<FloatType> &b){
  autoView(a_v,a,HostReadWrite);
  autoView(b_v,b,HostRead);
  for(int i=0;i<a.size(0);i++)
    a_v(i) += b_v(i);
  return a;
}

template<typename FloatType>
Vector<FloatType> operator-(const Vector<FloatType> &a, const Vector<FloatType> &b){
  Vector<FloatType> out(a.size(0), MemoryManager::Pool::HostPool);
  autoView(out_v,out,HostWrite);
  autoView(a_v,a,HostRead);
  autoView(b_v,b,HostRead);
  
  for(int i=0;i<a.size(0);i++)
    out_v(i) = a_v(i) - b_v(i);
  return out;
}

template<typename FloatType>
Vector<FloatType> operator*(FloatType eps, const Vector<FloatType> &b){
  Vector<FloatType> out(b.size(0), MemoryManager::Pool::HostPool);
  autoView(out_v,out,HostWrite);
  autoView(b_v,b,HostRead);
  
  for(int i=0;i<b.size(0);i++)
    out_v(i) = eps * b_v(i);
  return out;
}

template<typename FloatType>
Vector<FloatType> & operator*=(Vector<FloatType> &a, FloatType eps){
  autoView(a_v,a,HostReadWrite);
  for(int i=0;i<a.size(0);i++)
    a_v(i) *= eps;
  return a;
}
