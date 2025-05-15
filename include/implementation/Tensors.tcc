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

template<typename FloatType>
void pokeRow(Matrix<FloatType> &into, int row, const Vector<FloatType> &data){
  assert(data.size(0) == into.size(1));
  autoView(data_v,data,DeviceRead);
  autoView(t_v,into,DeviceWrite);
  accelerator_for(i,into.size(1),{
    t_v(row,i) = data_v(i);
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

template<typename FloatType,int Dim>
Tensor<FloatType,Dim> & operator+=(Tensor<FloatType,Dim> &a, const Tensor<FloatType,Dim> &b){
  for(int d=0;d<Dim;d++) assert(a.size(d) == b.size(d));
  size_t size = a.data_len();
  
  autoView(a_v,a,DeviceReadWrite);
  autoView(b_v,b,DeviceRead);
  accelerator_for(i,size,{
      a_v.data()[i] += b_v.data()[i];
    });
  return a;
}

template<typename FloatType, int Dim>
Tensor<FloatType,Dim> operator+(const Tensor<FloatType,Dim> &a, const Tensor<FloatType,Dim> &b){
  for(int d=0;d<Dim;d++) assert(a.size(d) == b.size(d));
  size_t size = a.data_len();

  Tensor<FloatType,Dim> out(a.sizeArray());
  autoView(out_v,out,DeviceWrite);
  autoView(a_v,a,DeviceRead);
  autoView(b_v,b,DeviceRead);

  accelerator_for(i,size,{
      out_v.data()[i] = a_v.data()[i] + b_v.data()[i];
    });
  return out;
}

template<typename FloatType,int Dim>
Tensor<FloatType,Dim> & operator-=(Tensor<FloatType, Dim> &a, const Tensor<FloatType,Dim> &b){
  for(int d=0;d<Dim;d++) assert(a.size(d) == b.size(d));
  size_t size = a.data_len();
  
  autoView(a_v,a,DeviceReadWrite);
  autoView(b_v,b,DeviceRead);
  accelerator_for(i,size,{
      a_v.data()[i] -= b_v.data()[i];
    });
  return a;
}

template<typename FloatType, int Dim>
Tensor<FloatType,Dim> operator-(const Tensor<FloatType,Dim> &a, const Tensor<FloatType,Dim> &b){
  for(int d=0;d<Dim;d++) assert(a.size(d) == b.size(d));
  size_t size = a.data_len();

  Tensor<FloatType,Dim> out(a.sizeArray());
  autoView(out_v,out,DeviceWrite);
  autoView(a_v,a,DeviceRead);
  autoView(b_v,b,DeviceRead);

  accelerator_for(i,size,{
      out_v.data()[i] = a_v.data()[i] - b_v.data()[i];
    });
  return out;
}

template<typename FloatType, int Dim>
Tensor<FloatType,Dim> operator*(FloatType eps, const Tensor<FloatType,Dim> &b){
  size_t size = b.data_len();
  Tensor<FloatType,Dim> out(b.sizeArray());
  autoView(out_v,out,DeviceWrite);
  autoView(b_v,b,DeviceRead);

  accelerator_for(i,size,{
      out_v.data()[i] = eps * b_v.data()[i];
    });
  return out;
}

template<typename FloatType, int Dim>
Tensor<FloatType,Dim> & operator*=(Tensor<FloatType,Dim> &a, FloatType eps){
  size_t size = a.data_len();
  
  autoView(a_v,a,DeviceReadWrite);
  accelerator_for(i, size, {
      a_v.data()[i] *= eps;
    });
  return a;
}

template<int Dim, typename FloatType>
Vector<FloatType> flatten(const Tensor<FloatType,Dim> &t){
  size_t out_sz=t.data_len();
  Vector<FloatType> out(out_sz);
  {
    autoView(out_v,out,DeviceWrite);
    autoView(t_v,t,DeviceRead);
    acceleratorCopyDeviceToDevice(out_v.data(),t_v.data(),out_sz*sizeof(FloatType));
  }
  return out;
}

template<int Dim, typename FloatType>
void unflatten(Tensor<FloatType,Dim> &out, const Vector<FloatType> &t){
  size_t sz = t.size(0);
  size_t test_sz= out.data_len();
  assert(sz == test_sz);
  {
    autoView(out_v,out,DeviceWrite);
    autoView(t_v,t,DeviceRead);
    acceleratorCopyDeviceToDevice(out_v.data(),t_v.data(),sz*sizeof(FloatType));
  }
}


template<int Dim1, int Dim2, typename FloatType>
Vector<FloatType> flatten2(const Tensor<FloatType,Dim1> &t1, const Tensor<FloatType,Dim2> &t2){
  size_t t1_lin=t1.data_len();
  size_t t2_lin=t2.data_len();
  size_t out_lin = t1_lin + t2_lin;
  
  Vector<FloatType> out(out_lin);
  {
    autoView(out_v,out,DeviceWrite);
    autoView(t1_v,t1,DeviceRead);
    autoView(t2_v,t2,DeviceRead);
    
    acceleratorCopyDeviceToDevice(out_v.data(),t1_v.data(), t1_lin*sizeof(FloatType));
    acceleratorCopyDeviceToDevice(out_v.data() + t1_lin, t2_v.data(), t2_lin*sizeof(FloatType));
  }
  return out;
}
  

template<int Dim1, int Dim2, typename FloatType>
void unflatten2(Tensor<FloatType,Dim1> &t1,  Tensor<FloatType,Dim2> &t2, const Vector<FloatType> &v){
  size_t t1_lin = t1.data_len();
  size_t t2_lin = t2.data_len();
  
  assert(v.size(0) == t1_lin + t2_lin);
  
  {
    autoView(t1_v,t1,DeviceWrite);
    autoView(t2_v,t2,DeviceWrite);
    autoView(v_v,v,DeviceRead);
    acceleratorCopyDeviceToDevice(t1_v.data(),v_v.data(), t1_lin*sizeof(FloatType));
    acceleratorCopyDeviceToDevice(t2_v.data(),v_v.data() + t1_lin, t2_lin*sizeof(FloatType));
  }
}


template<int Dim, typename FloatType>
Vector<FloatType> flattenNsameDim(Tensor<FloatType,Dim> const* const* tens, int N){
  size_t out_lin=0;
  for(int t=0;t<N;t++)
    out_lin += tens[t]->data_len();
  
  Vector<FloatType> out(out_lin);
  {
    autoView(out_v,out,DeviceWrite);
    size_t off = 0;
    for(int t=0;t<N;t++){
      autoView(t_v, (*tens[t]) ,DeviceRead);
      acceleratorCopyDeviceToDevice(out_v.data() + off, t_v.data(), t_v.data_len()*sizeof(FloatType));
      off += t_v.data_len();
    }
  }
  return out;
}
  

template<int Dim, typename FloatType>
void unflattenNsameDim(Tensor<FloatType,Dim>* const* tens, int N, const Vector<FloatType> &v){
  size_t t_lin=0;
  for(int t=0;t<N;t++)
    t_lin += tens[t]->data_len();
  assert(t_lin == v.size(0));
    
  {
    autoView(v_v,v,DeviceRead);
    size_t off = 0;
    for(int t=0;t<N;t++){
      autoView(t_v, (*tens[t]) ,DeviceWrite);
      acceleratorCopyDeviceToDevice(t_v.data(),v_v.data() + off, t_v.data_len()*sizeof(FloatType));
      off += t_v.data_len();
    }
  }
}



template<int Dim>
accelerator_inline size_t tensorDimensionStride(int iter_dim, int const* size){
  size_t stride = 1;
#pragma unroll
  for(int d=Dim-1;d>iter_dim;d--)
    stride *= size[d];
  return stride;
}
  
template<int Dim>
accelerator_inline size_t tensorDimensionBase(int iter_dim, int const* other_coord, int const *size){
  int coord[Dim];
  coord[iter_dim]=0;
  int i=0;
  for(int d=0;d<Dim;d++)
    if(d!=iter_dim)
      coord[d] = other_coord[i++];
  return tensorOffset<Dim>(coord, size);  
}

template<int Dim>
accelerator_inline size_t batchTensorDimensionBaseLin(int iter_dim, int batch_idx, size_t other_dim_lin, int const *size){
  int coord[Dim];
  coord[iter_dim]=0;
  coord[Dim-1] = batch_idx;
  size_t rem = other_dim_lin;

  //other_dim_lin for, eg 3 dims, mapped as     z + dim3*( y + dim2 * x )
  for(int d=Dim-2;d>=0;d--)
    if(d!=iter_dim){
      coord[d] = rem % size[d];
      rem /= size[d];
    }
  return tensorOffset<Dim>(coord, size);
}

template<int Dim, typename FloatType>
Tensor<FloatType,Dim> batchTensorConcatenate(Tensor<FloatType,Dim> const* const* in, int Ntens, int concat_dim){
  assert(concat_dim < Dim-1 && concat_dim >= 0); 
  int out_sz[Dim] = {0};
  size_t other_dim_len = 1;
  for(int i=0;i<Ntens;i++){
    for(int d=0;d<Dim;d++){
      int isz = in[i]->size(d);
      if(d==concat_dim)    
	out_sz[d] += isz;
      else{
	if(i==0){
	  out_sz[d] = isz;
	  other_dim_len *= isz;
	}else
	  assert(isz == out_sz[d]);
      }
    }
  }
  int batch_size = out_sz[Dim-1];
  size_t out_stride = tensorDimensionStride<Dim>(concat_dim, out_sz);
  
  Tensor<FloatType,Dim> out(out_sz);
  int off = 0;
  for(int i=0;i<Ntens;i++){
    size_t in_stride = tensorDimensionStride<Dim>(concat_dim, in[i]->sizeArray());
    size_t ooff = off * out_stride;
    autoView(out_v,out, i==0 ? DeviceWrite : DeviceReadWrite);
    autoView(in_v, (*in[i]), DeviceRead);
    
    accelerator_for2d(b,batch_size, o, other_dim_len,  1, {
	FloatType* iptr = in_v.data() + batchTensorDimensionBaseLin<Dim>(concat_dim, b, o, in_v.sizeArray());
	FloatType* optr = out_v.data() + batchTensorDimensionBaseLin<Dim>(concat_dim, b, o, out_v.sizeArray()) + ooff;
	for(int k=0;k<in_v.size(concat_dim);k++){
	  *optr = *iptr;
	  iptr += in_stride;
	  optr += out_stride;
	}
      });

    off += in[i]->size(concat_dim);
  }
  return out;  
}

template<int Dim, typename FloatType>
void batchTensorSplit(Tensor<FloatType,Dim>* const* out, int Ntens, const Tensor<FloatType,Dim> &in, int split_dim){
  int split_dim_tot = 0;
  for(int t=0;t<Ntens;t++){
    for(int d=0;d<Dim;d++){
      if(d== split_dim)
	split_dim_tot += out[t]->size(d);
      else assert(out[t]->size(d) == in.size(d));
    }
  }

  assert(in.size(split_dim) == split_dim_tot);
  
  size_t other_dim_len = 1;
  for(int d=0;d<Dim-1;d++)
    if(d!=split_dim)
      other_dim_len *= in.size(d);
  
  int batch_size = in.size(Dim-1);
  size_t in_stride = tensorDimensionStride<Dim>(split_dim, in.sizeArray());
  
  int off = 0;
  for(int i=0;i<Ntens;i++){
    size_t out_stride = tensorDimensionStride<Dim>(split_dim, out[i]->sizeArray());
    size_t ioff = off * in_stride;
    autoView(out_v, (*out[i]), DeviceWrite);
    autoView(in_v, in, DeviceRead);
    
    accelerator_for2d(b,batch_size, o, other_dim_len,  1, {
	FloatType* iptr = in_v.data() + batchTensorDimensionBaseLin<Dim>(split_dim, b, o, in_v.sizeArray()) + ioff;
	FloatType* optr = out_v.data() + batchTensorDimensionBaseLin<Dim>(split_dim, b, o, out_v.sizeArray());
	for(int k=0;k<out_v.size(split_dim);k++){
	  *optr = *iptr;
	  iptr += in_stride;
	  optr += out_stride;
	}
      });

    off += out[i]->size(split_dim);
  }
}
