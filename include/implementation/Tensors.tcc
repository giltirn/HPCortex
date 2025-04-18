template<typename FloatType, int Dim>
Tensor<FloatType,Dim> Tensor<FloatType,Dim>::sliceLastDimension(int idx_start, int idx_end) const{
  int osize[Dim]; memcpy(osize, this->sizeArray(), Dim*sizeof(int));
  osize[Dim-1] = idx_end-idx_start+1;
  Tensor<FloatType,Dim> out(osize);
  size_t other_size = 1;
  for(int i=0;i<Dim-1;i++) other_size *= osize[i];

  int osize_last = osize[Dim-1];
  int isize_last = this->sizeArray()[Dim-1];
  
  autoView(out_v,out,DeviceWrite);
  autoView(t_v,(*this),DeviceRead);
  accelerator_for2d(jj,idx_end-idx_start+1,i,other_size,1,{
      out_v.data()[jj + osize_last*i] = t_v.data()[jj+idx_start + isize_last*i];
    });
  return out;
}

 //Insert a tensor of Dim-1 such that (*this)(i,j,k,..., idx) = ins(i,j,k,...)
template<typename FloatType, int Dim>
void Tensor<FloatType,Dim>::pokeLastDimension(const Tensor<FloatType,Dim-1> &ins, const int idx){
  size_t other_size = 1;
  for(int i=0;i<Dim-1;i++){
    assert( ins.size(i) == this->size(i) );
    other_size *= ins.size(i);
  }
  size_t size_last = this->size(Dim-1);
    
  autoView(ins_v,ins,DeviceRead);
  autoView(t_v,(*this),DeviceReadWrite);
  accelerator_for2d(dummy1,1, i,other_size,32,{
      t_v.data()[idx + size_last *i] = ins_v.data()[i];
    });
}

template<typename FloatType, int Dim>
Tensor<FloatType,Dim-1> Tensor<FloatType,Dim>::peekLastDimension(const int idx) const{
  size_t other_size = 1;
  int out_size[Dim-1];
  for(int i=0;i<Dim-1;i++){
    out_size[i] = this->size(i);
    other_size *= out_size[i];
  }
  int size_last = this->size(Dim-1);
  
  Tensor<FloatType,Dim-1> out(out_size);
  autoView(out_v,out,DeviceWrite);
  autoView(t_v,(*this),DeviceRead);
  accelerator_for2d(dummy1,1, i,other_size,32,{
      out_v.data()[i] = t_v.data()[idx + size_last *i];
    });
  return out;
}



template<typename FloatType>
std::ostream & operator<<(std::ostream &os, const Vector<FloatType> &v){
  autoView(vv,v,HostRead);
  if(vv.size(0)==0){ os << "()"; return os; }    
  os << "(" << vv(0);
  for(int i=1;i<vv.size(0);i++) os << ", " << vv(i);
  os << ")";
  return os;  
}

//Insert 'data' as column 'col' of this matrix
template<typename FloatType>
void pokeColumn(Matrix<FloatType> &into, int col, const Vector<FloatType> &data){
  assert(data.size(0) == into.size(0));
  autoView(data_v,data,DeviceRead);
  autoView(t_v,into,DeviceWrite);
  accelerator_for(i,into.size(0),{
    t_v(i,col) = data_v(i);
    });
}
  
//Retrieve column 'col' of this matrix
template<typename FloatType>
Vector<FloatType> peekColumn(const Matrix<FloatType> &m, int col){
  Vector<FloatType> out(m.size(0));
  autoView(out_v,out,DeviceWrite);
  autoView(t_v,m,DeviceRead);
  accelerator_for(i,m.size(0),{ out_v(i)=t_v(i,col); });
  return out;
}
  

//Retrieve multiple columns as a new matrix
template<typename FloatType>
Matrix<FloatType> peekColumns(const Matrix<FloatType> &m, int col_start, int col_end){
  Matrix<FloatType> out(m.size(0), col_end-col_start+1);
  autoView(out_v,out,DeviceWrite);
  autoView(t_v,m,DeviceRead);
  accelerator_for2d(jj,col_end-col_start+1,i,m.size(0),1,{
      int j = jj + col_start;
      out_v(i,jj)=t_v(i,j);
    });
  return out;
}

//Insert multiple columns, collected as a matrix 'cols', into this matrix
template<typename FloatType>
void pokeColumns(Matrix<FloatType> &into, int col_start, int col_end, const Matrix<FloatType> &cols){
  assert(cols.size(0) == into.size(0) && cols.size(1) == col_end-col_start+1);
  autoView(cols_v,cols,DeviceRead);
  autoView(t_v,into,DeviceWrite);
  accelerator_for2d(jj,col_end-col_start+1,i,into.size(0),1,{
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
