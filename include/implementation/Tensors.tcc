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
  autoView(data_v,data,DeviceRead);
  autoView(t_v,(*this),DeviceWrite);
  accelerator_for(i,size0,{
    t_v(i,col) = data_v(i);
    });
}

template<typename FloatType>
Vector<FloatType> Matrix<FloatType>::peekColumn(int col) const{
  Vector<FloatType> out(size0);
  autoView(out_v,out,DeviceWrite);
  autoView(t_v,(*this),DeviceRead);
  accelerator_for(i,size0,{ out_v(i)=t_v(i,col); });
  return out;
}

template<typename FloatType>
Matrix<FloatType> Matrix<FloatType>::peekColumns(int col_start, int col_end) const{
  Matrix<FloatType> out(size0, col_end-col_start+1);
  autoView(out_v,out,DeviceWrite);
  autoView(t_v,(*this),DeviceRead);
  accelerator_for2d(jj,col_end-col_start+1,i,size0,1,{
      int j = jj + col_start;
      out_v(i,jj)=t_v(i,j);
    });
  return out;
}

template<typename FloatType>
void Matrix<FloatType>::pokeColumns(int col_start, int col_end, const Matrix<FloatType> &cols){
  assert(cols.size(0) == this->size(0) && cols.size(1) == col_end-col_start+1);
  autoView(cols_v,cols,DeviceRead);
  autoView(t_v,(*this),DeviceWrite);
  accelerator_for2d(jj,col_end-col_start+1,i,size0,1,{
      int j = jj + col_start;
      t_v(i,j) = cols_v(i,jj);
    });
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
Matrix<FloatType> & operator+=(Matrix<FloatType> &a, const Matrix<FloatType> &b){
  size_t size0 = a.size(0);
  size_t size1 = a.size(1);
  assert(b.size(0)==size0 && b.size(1) == size1);

  autoView(a_v,a,DeviceReadWrite);
  autoView(b_v,b,DeviceRead);
  accelerator_for2d(j,size1,i,size0,1,{
      a_v(i,j) += b_v(i,j);
    });
  return a;
}

template<typename FloatType>
Matrix<FloatType> operator+(const Matrix<FloatType> &a, const Matrix<FloatType> &b){
  size_t size0 = a.size(0);
  size_t size1 = a.size(1);
  assert(b.size(0)==size0 && b.size(1) == size1);
  Matrix<FloatType> out(size0,size1);
    
  autoView(a_v,a,DeviceRead);
  autoView(b_v,b,DeviceRead);
  autoView(out_v,out,DeviceWrite);
  accelerator_for2d(j,size1,i,size0,1,{
      out_v(i,j) = a_v(i,j) + b_v(i,j);
    });
  return out;
}




template<typename FloatType>
Vector<FloatType> operator*(const Matrix<FloatType> &A, const Vector<FloatType> &x){
  size_t size0 = A.size(0), size1 = A.size(1);
  assert(size1 == x.size(0));
  
  Vector<FloatType> out(size0, 0.);
  autoView(x_v,x,DeviceRead);
  autoView(out_v,out,DeviceReadWrite);
  autoView(A_v,A,DeviceRead);

  //simple, inefficient implementation
  accelerator_for(i,size0,{
      for(int j=0;j<size1;j++)
	out_v(i) += A_v(i,j) * x_v(j);
    });
  return out;
}

template<typename FloatType>
Vector<FloatType> operator+(const Vector<FloatType> &a, const Vector<FloatType> &b){
  size_t size = a.size(0);
  assert(b.size(0) == size);
  Vector<FloatType> out(size);
  autoView(out_v,out,DeviceWrite);
  autoView(a_v,a,DeviceRead);
  autoView(b_v,b,DeviceRead);
  accelerator_for(i,size,{
    out_v(i) = a_v(i) + b_v(i);
    });
  return out;
}

template<typename FloatType>
Vector<FloatType> & operator+=(Vector<FloatType> &a, const Vector<FloatType> &b){
  size_t size = a.size(0);
  assert(b.size(0) == size);
  autoView(a_v,a,DeviceReadWrite);
  autoView(b_v,b,DeviceRead);
  accelerator_for(i,size,{
    a_v(i) += b_v(i);
    });
  return a;
}

template<typename FloatType>
Vector<FloatType> operator-(const Vector<FloatType> &a, const Vector<FloatType> &b){
  size_t size = a.size(0);
  assert(b.size(0) == size);
  Vector<FloatType> out(size);
  autoView(out_v,out,DeviceWrite);
  autoView(a_v,a,DeviceRead);
  autoView(b_v,b,DeviceRead);

  accelerator_for(i,size,{
    out_v(i) = a_v(i) - b_v(i);
    });
  return out;
}

template<typename FloatType>
Vector<FloatType> operator*(FloatType eps, const Vector<FloatType> &b){
  size_t size = b.size(0);
  Vector<FloatType> out(size);
  autoView(out_v,out,DeviceWrite);
  autoView(b_v,b,DeviceRead);

  accelerator_for(i,size,{
    out_v(i) = eps * b_v(i);
    });
  return out;
}

template<typename FloatType>
Vector<FloatType> & operator*=(Vector<FloatType> &a, FloatType eps){
  size_t size = a.size(0);
  
  autoView(a_v,a,DeviceReadWrite);
  accelerator_for(i, size, {
    a_v(i) *= eps;
    });
  return a;
}
