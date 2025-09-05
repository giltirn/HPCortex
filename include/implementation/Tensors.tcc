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

template<typename FloatType, int Dim>
Tensor<FloatType,Dim> Tensor<FloatType,Dim>::sliceLastDimension(int idx_start, int idx_end) const{
  int osize[Dim]; memcpy(osize, this->sizeArray(), Dim*sizeof(int));
  osize[Dim-1] = idx_end-idx_start+1;
  Tensor<FloatType,Dim> out(osize);
  size_t other_size = 1;
  for(int i=0;i<Dim-1;i++) other_size *= osize[i];

  int osize_last = osize[Dim-1];
  int isize_last = this->sizeArray()[Dim-1];
  assert(idx_start < isize_last && idx_end < isize_last && idx_end >= idx_start);
  
  autoView(out_v,out,DeviceWrite);
  autoView(t_v,(*this),DeviceRead);
  accelerator_for2d(jj,idx_end-idx_start+1,i,other_size,1,{
      out_v.data()[jj + osize_last*i] = t_v.data()[jj+idx_start + isize_last*i];
    });
  return out;
}
template<typename FloatType, int Dim>
void Tensor<FloatType,Dim>::insertSliceLastDimension(const Tensor &ins, int idx_start, int idx_end) const{
  size_t other_size = 1;
  for(int i=0;i<Dim-1;i++) other_size *= this->size(i);

  int osize_last = this->size(Dim-1);
  assert(idx_start < osize_last && idx_end < osize_last && idx_end >= idx_start);
  
  int isize_last = ins.size(Dim-1);
  assert(isize_last == idx_end-idx_start+1);
  
  autoView(t_v,(*this),DeviceReadWrite);
  autoView(ins_v,ins,DeviceRead);
  accelerator_for2d(jj,idx_end-idx_start+1,i,other_size,1,{
      t_v.data()[jj + idx_start + osize_last*i] = ins_v.data()[jj + isize_last*i];
    });
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
  accelerator_for_gen(1,0,splitBlock<32>(), i,other_size,{
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

template<typename FloatType, int Dim>
template<typename FloatTypeOut>
Tensor<FloatTypeOut,Dim> Tensor<FloatType,Dim>::convertFloatType(Locale loc) const{
  if(loc == Auto && deviceResident()) loc = Device;

  Tensor<FloatTypeOut,Dim> out(this->sizeArray(),loc == Device ? MemoryManager::Pool::DevicePool : MemoryManager::Pool::HostPool);
  
  autoView(t_v,(*this),loc == Device ? DeviceRead : HostRead);
  autoView(out_v,out, loc == Device ? DeviceWrite : HostWrite);

#define BODY out_v.data()[i] = (FloatTypeOut)t_v.data()[i];
  if(loc == Device){  
    accelerator_for_gen(0,1,normal(),i,t_v.data_len(),{ BODY; });
  }else{
    thread_for(i,t_v.data_len(),{ BODY; });
  }
#undef BODY
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
Matrix<FloatType> transpose(const Matrix<FloatType> &m, Locale loc){
  if(loc == Auto && m.deviceResident()) loc = Device;

  if(loc == Device){
    int isize = m.size(0);
    int jsize = m.size(1);
  
    Matrix<FloatType> into(jsize,isize);    
    autoView(mat_v,m,DeviceRead);
    autoView(into_v,into,DeviceWrite);

    if(m.size(0) == 1 || m.size(1) == 1){
      acceleratorCopyDeviceToDevice(into_v.data(),mat_v.data(),mat_v.data_len()*sizeof(FloatType));
      return into;
    }
    
    constexpr int iblocksize = 8;
    int iblocks = (isize + iblocksize -1)/iblocksize;

    constexpr int jblocksize = 8;
    int jblocks = (jsize + jblocksize -1)/jblocksize;
  
    accelerator_for_3d_gen(1,2,shm( (jblocksize+1)*iblocksize*sizeof(FloatType)), t, jblocksize,  bi, iblocks,  bj, jblocks, {
	FloatType* bstore = (FloatType*)shared;
      
	int ioff = bi*iblocksize;
	int iblocksize_actual = min(isize - ioff,iblocksize);
	int joff = bj*jblocksize;
	int jblocksize_actual = min(jsize - joff,jblocksize);

	//parallel load jblocksize consecutive elements for fixed i
	FloatType const *mat_p = mat_v.data() + joff + t;
	for(int ii=0;ii<iblocksize_actual;ii++){      
	  int i = ii + ioff;
	  if(t < jblocksize_actual) bstore[t + (jblocksize+1)*ii] = *(mat_p + i*jsize);
	}
	acceleratorSynchronizeBlock();

	//parallel write iblocksize consecutive elements into output for fixed j
	for(int jj=0;jj<jblocksize_actual;jj++){
	  int j = jj+joff;
	  int ii=t;
	  while( ii < iblocksize_actual){
	    into_v.data()[ ii + ioff + isize*j] = bstore[jj + (jblocksize+1)*ii ];	  
	    ii += jblocksize;
	  }
	}

      });
  
    return into;
  }else{
    int out_sz[2] = { m.size(1), m.size(0) };
    Matrix<FloatType> out(m.size(1),m.size(0),MemoryManager::Pool::HostPool);
  
    autoView(in_v,m, HostRead);
    autoView(out_v,out, HostWrite);
  
    thread_for(s,in_v.data_len(),{
	int i = s / in_v.size(1), j = s % in_v.size(1); 
	out_v(j,i) = in_v(i,j);
      });
    return out;
  }
  
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
FloatType * flatten(FloatType* host_ptr, const Tensor<FloatType,Dim> &in){
  size_t in_sz = in.data_len();
  {
    autoView(in_v,in,HostRead);
    memcpy(host_ptr, in_v.data(), in_sz * sizeof(FloatType));
  }
  return host_ptr + in_sz;
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

template<int Dim, typename FloatType>
FloatType const* unflatten(Tensor<FloatType,Dim> &out, FloatType const* host_ptr){
  size_t sz= out.data_len();
  {
    autoView(out_v,out,HostWrite);
    memcpy(out_v.data(), host_ptr, sz*sizeof(FloatType));
  }
  return host_ptr + sz;
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
accelerator_inline size_t tensorDimensionBaseLin(int iter_dim, size_t other_dim_lin, int const *size){
  size_t out = 0;
  size_t coeff = 1;
  size_t rem = other_dim_lin;
#pragma unroll
  for(int d=Dim-1;d>=0;d--){
    int coord_d;
    if(d==iter_dim){
      coord_d = 0;
    }else{
      coord_d = rem % size[d];
      rem /= size[d];
    }
    out += coord_d * coeff;
    coeff *= size[d];
  }
  return out;
}

template<int Dim>
accelerator_inline size_t batchTensorDimensionBaseLin(int iter_dim, int batch_idx, size_t other_dim_lin, int const *size){
  size_t out = batch_idx;
  size_t coeff = size[Dim-1];
  size_t rem = other_dim_lin;
#pragma unroll
  for(int d=Dim-2;d>=0;d--){
    int coord_d;
    if(d==iter_dim){
      coord_d = 0;
    }else{
      coord_d = rem % size[d];
      rem /= size[d];
    }
    out += coord_d * coeff;
    coeff *= size[d];
  }
  return out;
}

template<>
accelerator_inline size_t batchTensorDimensionBaseLin<2>(int iter_dim, int batch_idx, size_t other_dim_lin, int const *size){
  //b + batch_size * u,  return offset for i=0
  return batch_idx;
}

template<>
accelerator_inline size_t batchTensorDimensionBaseLin<3>(int iter_dim, int batch_idx, size_t other_dim_lin, int const *size){
  //[j,k,b] -> b + batch_size * ( k + sizek * j )
  //size = [ sizej, sizek, batch_size ]
  if(iter_dim == 0){ //other_dim_lin = k, return offset for j=0
    return  batch_idx + size[2] * other_dim_lin;
  }else{ //iter_dim == 1,  other_dim_lin = j, return offset for k=0 :  b + batch_size * sizek * j
    return batch_idx + size[2]*size[1] * other_dim_lin;
  }
}


template<int Dim, typename FloatType>
Tensor<FloatType,Dim> batchTensorConcatenate(Tensor<FloatType,Dim> const* const* in, int Ntens, int concat_dim){
  assert(concat_dim < Dim-1 && concat_dim >= 0); 
  int out_sz[Dim] = {0};

  for(int i=0;i<Ntens;i++){
    for(int d=0;d<Dim;d++){
      if(d==concat_dim)    
	out_sz[d] += in[i]->size(d); //sum dimensions along concat_dim
      else{
	if(i==0)
	  out_sz[d] = in[i]->size(d); //other dimensions stay the same and must be equal for all tensors
	else
	  assert(in[i]->size(d) == out_sz[d]);
      }
    }
  }

  size_t other_dim_lin = 1; //product of dimensions != concat_dim && != Dim-1
  for(int d=0;d<Dim-1;d++)
    if(d!=concat_dim)
      other_dim_lin *= in[0]->size(d);
  
  int batch_size = out_sz[Dim-1];
  size_t out_stride = tensorDimensionStride<Dim>(concat_dim, out_sz);
  
  Tensor<FloatType,Dim> out(out_sz);
  int off = 0;
  for(int i=0;i<Ntens;i++){
    size_t in_stride = tensorDimensionStride<Dim>(concat_dim, in[i]->sizeArray());
    size_t ooff = off * out_stride;
    autoView(out_v,out, i==0 ? DeviceWrite : DeviceReadWrite);
    autoView(in_v, (*in[i]), DeviceRead);
    
    accelerator_for2d(b,batch_size, o, other_dim_lin,  1, {
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

template<int Dim, typename FloatType>
double norm2(const Tensor<FloatType,Dim> &T){
#ifndef USE_GPU
  double out = 0.;
  autoView(T_v,T,HostRead);
  for(size_t i=0; i<T.data_len(); i++){
    FloatType v = T_v.data()[i];
    out += v*v;
  }
  return out;
#else 
  int last_dim_sz = T.size(Dim-1);
  double* accum = (double*)acceleratorAllocDevice(sizeof(double));
  acceleratorMemSet(accum,0,sizeof(double));

  int blocksize = 32;
  int blocks = (last_dim_sz + blocksize -1)/blocksize;

  size_t other_dim_lin = 1;
  for(int i=0;i<Dim-1;i++)
    other_dim_lin *= T.size(i);

  autoView(T_v,T,DeviceRead);
  accelerator_for3d_shm(bb,blocksize, b, blocks, o, other_dim_lin, 1, (blocksize*sizeof(FloatType)),
			{
			  FloatType* buf = (FloatType*)shared;
			  int last_dim_idx = bb + blocksize*b;
			  FloatType v = last_dim_idx < last_dim_sz ? *(T_v.data() + last_dim_idx + last_dim_sz*o) : 0.;
			  buf[bb] = v*v;
			  acceleratorSynchronizeBlock();
			  if(bb == 0){
			    double vs = 0.;
			    for(int i=0;i<blocksize;i++)
			      vs += buf[i];
			    atomicAdd(accum, vs);
			  }
			});  
  double out;
  acceleratorCopyFromDevice(&out, accum, sizeof(double));
  acceleratorFreeDevice(accum);
  return out;
#endif  
}

template<int Dim, typename FloatType>
Tensor<FloatType,Dim> dimensionSlice(const Tensor<FloatType,Dim> &from, int const* indices, int nidx, const int dimension, Locale loc){
  if(loc == Auto) loc = from.deviceResident() ? Device : Host;

  int out_size[Dim];
  memcpy(out_size,from.sizeArray(),Dim*sizeof(int));
  out_size[dimension] = nidx;
  
  Tensor<FloatType,Dim> out(out_size, loc == Device ? MemoryManager::Pool::DevicePool : MemoryManager::Pool::HostPool);
  
  size_t other_dim_vol=1;  
  for(int d=0;d<Dim;d++)
    if(d!=dimension)
      other_dim_vol *= from.size(d);

  int istride = tensorDimensionStride<Dim>(dimension,from.sizeArray());
  int ostride = tensorDimensionStride<Dim>(dimension,out.sizeArray());

  int const* indices_p = indices;
  if(loc == Device){
    int *idx_dev = (int *)acceleratorAllocDevice(nidx*sizeof(int));
    acceleratorCopyToDevice(idx_dev, indices, nidx*sizeof(int));
    indices_p = idx_dev;
  }
    
  int slice_dim_size = nidx;
  
  autoView(out_v,out, loc == Device ? DeviceWrite : HostWrite);
  autoView(in_v,from, loc == Device ? DeviceRead : HostRead);

  #define BODY \
    size_t off_i = tensorDimensionBaseLin<Dim>(dimension,o, in_v.sizeArray()); \
    size_t off_o = tensorDimensionBaseLin<Dim>(dimension,o, out_v.sizeArray()); \
    FloatType* to_p = out_v.data() + off_o; \
    FloatType const* from_p = in_v.data() + off_i; \
    for(int i=0;i<slice_dim_size;i++){ \
      *to_p = *(from_p + istride * indices_p[i]); \
      to_p += ostride; \
    }
     
  if(loc == Device){
    //std::cout << "ON DEVICE" << std::endl;
    accelerator_for_gen(0,1,normal(), o, other_dim_vol, { BODY; });
  }else{
    //std::cout << "ON HOST" << std::endl;
    thread_for(o, other_dim_vol, { BODY; });
  }
#undef BODY

  if(loc == Device){
    int *idx_dev = (int*)indices_p;
    acceleratorFreeDevice(idx_dev);
  }
  return out;
}
  


template<int Dim, typename FloatType>
Tensor<FloatType,Dim> dimensionSlice(const Tensor<FloatType,Dim> &from, const std::vector<int> &indices, const int dimension, Locale loc){
  return dimensionSlice(from, indices.data(), indices.size(), dimension, loc);
}
  
template<int Dim, typename FloatType>
normalization<FloatType,Dim-1> normalize(Tensor<FloatType,Dim> &tens, const int dimension, Locale loc, FloatType epsilon){
  int i=0;  
  size_t other_dim_vol=1;
  int nrm_sz[Dim-1];
  for(int d=0;d<Dim;d++)
    if(d!=dimension){
      nrm_sz[i++] = tens.size(d);
      other_dim_vol *= tens.size(d);
    }
  if(loc == Auto) loc = tens.deviceResident() ? Device : Host;
  
  normalization<FloatType,Dim-1> out(nrm_sz, epsilon, loc == Device ? MemoryManager::Pool::DevicePool : MemoryManager::Pool::HostPool);

  int stride = tensorDimensionStride<Dim>(dimension,tens.sizeArray());
  int norm_dim_size = tens.size(dimension);
  
  autoView(tens_v,tens, loc == Device ? DeviceReadWrite : HostReadWrite);
  autoView(mu_v,out.mean, loc == Device ? DeviceWrite : HostWrite);
  autoView(std_v,out.std, loc == Device ? DeviceWrite : HostWrite);

  #define BODY \
    size_t off = tensorDimensionBaseLin<Dim>(dimension,o, tens_v.sizeArray()); \
    FloatType* tens_p = tens_v.data() + off; \
    \
    FloatType mean = 0., std=0.;	         \
    for(int i=0;i<norm_dim_size;i++){ \
       FloatType ii = *tens_p; tens_p += stride; \
       mean += ii; \
       std += ii*ii; \
     } \
     mean = mean / norm_dim_size; \
     std = sqrt( std / norm_dim_size - mean*mean );	\
     \
     tens_p = tens_v.data() + off; \
     for(int i=0;i<norm_dim_size;i++){ \
	*tens_p = ( (*tens_p) - mean ) / std; \
	tens_p += stride; \
     }\
     mu_v.data()[o] = mean; \
     std_v.data()[o] = std;
     
  if(loc == Device){
    //std::cout << "ON DEVICE" << std::endl;
    accelerator_for_gen(0,1,normal(), o, other_dim_vol, { BODY; });
  }else{
    //std::cout << "ON HOST" << std::endl;
    thread_for(o, other_dim_vol, { BODY; });
  }
#undef BODY
  return out;
}

template<int Dim, typename FloatType>
void unnormalize(Tensor<FloatType,Dim> &tens, const int dimension, const normalization<FloatType,Dim-1> &nrm, Locale loc){
  if(loc == Auto) loc = tens.deviceResident() ? Device : Host;

  size_t other_dim_vol=1;  
  for(int d=0;d<Dim;d++)
    if(d!=dimension)
      other_dim_vol *= tens.size(d);

  int stride = tensorDimensionStride<Dim>(dimension,tens.sizeArray());
  int norm_dim_size = tens.size(dimension);
  
  autoView(tens_v,tens, loc == Device ? DeviceReadWrite : HostReadWrite);
  autoView(mu_v,nrm.mean, loc == Device ? DeviceRead : HostRead);
  autoView(std_v,nrm.std, loc == Device ? DeviceRead : HostRead);

  #define BODY \
    size_t off = tensorDimensionBaseLin<Dim>(dimension,o, tens_v.sizeArray()); \
    FloatType* tens_p = tens_v.data() + off; \
    \
    FloatType mean = mu_v.data()[o], std = std_v.data()[o];	\
     \
     tens_p = tens_v.data() + off; \
     for(int i=0;i<norm_dim_size;i++){ \
	*tens_p = (*tens_p) * std + mean; \
	tens_p += stride; \
     }
     
  if(loc == Device){
    //std::cout << "ON DEVICE" << std::endl;
    accelerator_for_gen(0,1,normal(), o, other_dim_vol, { BODY; });
  }else{
    //std::cout << "ON HOST" << std::endl;
    thread_for(o, other_dim_vol, { BODY; });
  }
#undef BODY
}


template<typename FloatType, int Dim>
void untransformBatchMatrix(int rowdim, int coldim, Tensor<FloatType,Dim> &tens, Vector<FloatType> &from){
  assert(rowdim != coldim);
  autoView(tens_v,tens,DeviceWrite);
  autoView(from_v,from,DeviceRead);
  
  accelerator_for_gen(1,0,splitBlock<32>(), i, tens.data_len(),{
      int coord[Dim];
      tensorOffsetUnmap<Dim>(coord, tens_v.sizeArray(), i);

      int o=0;
      for(int d=0;d<Dim;d++) if(d != rowdim && d != coldim) o = o * tens_v.size(d) + coord[d];
    
      int off = coord[coldim] + tens_v.size(coldim)*( coord[rowdim] + tens_v.size(rowdim) * o );
      tens_v.data()[i] = from_v(off);
    });
}
template<typename FloatType>
void untransformBatchMatrix(int rowdim, int coldim, Tensor<FloatType,3> &tens, Vector<FloatType> &from){
  assert(rowdim == !coldim);
  autoView(tens_v,tens,DeviceWrite);
  autoView(from_v,from,DeviceRead);

  int ni = tens.size(0);
  int nj = tens.size(1);
  int nb = tens.size(2);
  int nrows_out = tens.size(rowdim);
  int ncols_out = tens.size(coldim);

  if(rowdim == 0){
    constexpr int jblocksz = 32;
    constexpr int bblocksz = 32;
    constexpr int nthr = jblocksz;
    int nbblocks = (nb + bblocksz -1)/bblocksz;
    int njblocks = (nj + jblocksz -1)/jblocksz;
    
    accelerator_for_4d_gen(1,3,shm(jblocksz*(bblocksz+1)*sizeof(FloatType)), t, nthr,i,ni,bblock,nbblocks,jblock,njblocks,{
	FloatType *bstore = (FloatType*)shared;
	int jbase = jblock * jblocksz;
	int jblocksz_actual = nj - jbase < jblocksz ? nj - jbase : jblocksz;
	int bbase = bblock * bblocksz;
	int bblocksz_actual = nb - bbase < bblocksz ? nb - bbase : bblocksz;
	
	//load a block jsize=nthr  bsize=bblocksz  into shm
	for(int bb=0;bb<bblocksz_actual;bb++){
	  int b = bb + bbase;
	  //load nthr consecutive j for fixed b
	  if(t < jblocksz_actual){
	    FloatType const* fp = from_v.data() + t + jbase + nj*(i + ni*b);  //off = j + nj * (i + ni * b);
	    bstore[t + (nthr+1)*bb] = *fp;
	  }
	}
	acceleratorSynchronizeBlock();

	for(int jj=0;jj<jblocksz_actual;jj++){	    
	  //parallel write of bsize=bblock	  
	  if(t < bblocksz_actual){ //assume nthr >= bblocksz
	    int b = bbase + t;
	    tens_v(i,jbase+jj,b) = bstore[jj + (nthr+1)*t];	     
	  }
	}
      });
  }else{
    constexpr int iblocksz = 32;
    constexpr int bblocksz = 32;
    constexpr int nthr = iblocksz;
    int nbblocks = (nb + bblocksz -1)/bblocksz;
    int niblocks = (ni + iblocksz -1)/iblocksz;
    
    accelerator_for_4d_gen(1,3,shm(iblocksz*(bblocksz+1)*sizeof(FloatType)), t, nthr,j,nj,bblock,nbblocks,iblock,niblocks,{
	FloatType *bstore = (FloatType*)shared;
	int ibase = iblock * iblocksz;
	int iblocksz_actual = ni - ibase < iblocksz ? ni - ibase : iblocksz;
	int bbase = bblock * bblocksz;
	int bblocksz_actual = nb - bbase < bblocksz ? nb - bbase : bblocksz;
	
	//load a block isize=nthr  bsize=bblocksz  into shm
	for(int bb=0;bb<bblocksz_actual;bb++){
	  int b = bb + bbase;
	  //load nthr consecutive i for fixed b
	  if(t < iblocksz_actual){
	    FloatType const* fp = from_v.data() + t + ibase + ni*(j + nj*b);  //off = i + ni * (j + nj * b);
	    bstore[t + (nthr+1)*bb] = *fp;
	  }
	}
	acceleratorSynchronizeBlock();

	for(int ii=0;ii<iblocksz_actual;ii++){	    
	  //parallel write of bsize=bblock	  
	  if(t < bblocksz_actual){ //assume nthr >= bblocksz
	    int b = bbase + t;
	    tens_v(ibase+ii,j,b) = bstore[ii + (nthr+1)*t];	     
	  }
	}
      });
    

  }
}

template<typename FloatType, int Dim>
Vector<FloatType> transformBatchMatrix(int rowdim, int coldim, const Tensor<FloatType,Dim> &tens){
  assert(rowdim != coldim);
  Vector<FloatType> into(tens.data_len());
  autoView(tens_v,tens,DeviceRead);
  autoView(into_v,into,DeviceWrite);
  
  accelerator_for_gen(1,0,splitBlock<32>(), i, tens.data_len(),{
      int coord[Dim];
      tensorOffsetUnmap<Dim>(coord, tens_v.sizeArray(), i);

      int o=0;
      for(int d=0;d<Dim;d++) if(d != rowdim && d != coldim) o = o * tens_v.size(d) + coord[d];
    
      int off = coord[coldim] + tens_v.size(coldim)*( coord[rowdim] + tens_v.size(rowdim) * o );
      into_v(off) = tens_v.data()[i];
    });

  return into;
}
template<typename FloatType>
Vector<FloatType> transformBatchMatrix(int rowdim, int coldim, const Tensor<FloatType,3> &tens){
  assert(rowdim == !coldim);
  Vector<FloatType> into(tens.data_len());
  autoView(tens_v,tens,DeviceRead);
  autoView(into_v,into,DeviceWrite);

  int ni = tens.size(0);
  int nj = tens.size(1);
  int nb = tens.size(2);
  int nrows_out = tens.size(rowdim);
  int ncols_out = tens.size(coldim);

  if(rowdim == 0){
    constexpr int jblocksz = 32;
    constexpr int bblocksz = 32;
    constexpr int nthr = bblocksz;
    int nbblocks = (nb + bblocksz -1)/bblocksz;
    int njblocks = (nj + jblocksz -1)/jblocksz;
    
    accelerator_for_4d_gen(1,3,shm(bblocksz*(jblocksz+1)*sizeof(FloatType)), t, nthr,i,ni,bblock,nbblocks,jblock,njblocks,{
	FloatType *bstore = (FloatType*)shared;
	int jbase = jblock * jblocksz;
	int jblocksz_actual = nj - jbase < jblocksz ? nj - jbase : jblocksz;
	int bbase = bblock * bblocksz;
	int bblocksz_actual = nb - bbase < bblocksz ? nb - bbase : bblocksz;
	
	//load a block bsize=nthr  jsize=jblocksz  into shm
	for(int jj=0;jj<jblocksz_actual;jj++){
	  int j = jj + jbase;
	  //load nthr consecutive b for fixed j
	  if(t < bblocksz_actual){
	    int b = bbase + t;
	    bstore[t + (nthr+1)*jj] = tens_v(i,j,b);
	  }
	}
	acceleratorSynchronizeBlock();

	for(int bb=0;bb<bblocksz_actual;bb++){	    
	  //parallel write of jsize=jblock
	  int b = bb + bbase;
	  if(t < jblocksz_actual){ //assume nthr >= jblocksz
	    into_v.data()[t + jbase + nj*(i + ni*b)] = bstore[bb + (nthr+1)*t]  ;  //off = j + nj * (i + ni * b);
	  }
	}
      });
  }else{
    constexpr int iblocksz = 32;
    constexpr int bblocksz = 32;
    constexpr int nthr = bblocksz;
    int nbblocks = (nb + bblocksz -1)/bblocksz;
    int niblocks = (ni + iblocksz -1)/iblocksz;
    
    accelerator_for_4d_gen(1,3,shm(bblocksz*(iblocksz+1)*sizeof(FloatType)), t, nthr,j,nj,bblock,nbblocks,iblock,niblocks,{
	FloatType *bstore = (FloatType*)shared;
	int ibase = iblock * iblocksz;
	int iblocksz_actual = ni - ibase < iblocksz ? ni - ibase : iblocksz;
	int bbase = bblock * bblocksz;
	int bblocksz_actual = nb - bbase < bblocksz ? nb - bbase : bblocksz;
	
	//load a block bsize=nthr  isize=iblocksz  into shm
	for(int ii=0;ii<iblocksz_actual;ii++){
	  int i = ii + ibase;
	  //load nthr consecutive b for fixed i
	  if(t < bblocksz_actual){
	    int b = bbase + t;
	    bstore[t + (nthr+1)*ii] = tens_v(i,j,b);
	  }
	}
	acceleratorSynchronizeBlock();

	for(int bb=0;bb<bblocksz_actual;bb++){	    
	  //parallel write of isize=iblock
	  int b = bb + bbase;
	  if(t < iblocksz_actual){ //assume nthr >= jblocksz
	    into_v.data()[t + ibase + ni*(j + nj*b)] = bstore[bb + (nthr+1)*t]  ;  //off = i + ni * (j + nj * b);
	  }
	}
      });
  }
  return into;
}

template<typename FloatType, int Dim>
Vector<FloatType> transformBatchVector(int vecdim, const Tensor<FloatType,Dim> &tens){
  assert(vecdim != Dim-1); //batch dim
  Vector<FloatType> into(tens.data_len());
  autoView(tens_v,tens,DeviceRead);
  autoView(into_v,into,DeviceWrite);

  int stride = tensorDimensionStride<Dim>(vecdim, tens.sizeArray());
  int other_size_lin=1;
  for(int i=0;i<Dim-1;i++)
    if(i != vecdim) other_size_lin *= tens.size(i);
  int batch_size = tens.size(Dim-1);

  constexpr int bblocksize = 32;
  int bblocks = (batch_size + bblocksize -1)/bblocksize;

  int isize = tens.size(vecdim);
  constexpr int iblocksize = 32;
  int iblocks = (isize + iblocksize -1)/iblocksize;
  
  accelerator_for_4d_gen(1,3,shm( (bblocksize+1)*iblocksize*sizeof(FloatType)), t, bblocksize, bblock, bblocks, bi, iblocks, o, other_size_lin,{
      FloatType* bstore = (FloatType*)shared;
      int ioff = bi*iblocksize;
      int iblocksize_actual = min(isize - ioff,iblocksize);

      int boff = bblock*bblocksize;
      int bblocksize_actual = min(batch_size - boff,bblocksize);
      
      FloatType* tens_p = tens_v.data() + batchTensorDimensionBaseLin<Dim>(vecdim, t + boff, o, tens_v.sizeArray());
      
      //parallel load batch_size data into shared
      for(int ii=0;ii<iblocksize_actual;ii++){      
	int i = ii + ioff;
	if(t < bblocksize_actual) bstore[t + (bblocksize+1)*ii] = *(tens_p + i*stride);
      }
      acceleratorSynchronizeBlock();

      //parallel write iblocksize into output
      for(int bb=0;bb<bblocksize_actual;bb++){
	int ii=t;
	while( ii < iblocksize_actual){
	  into_v( ii + ioff + isize*(bb + boff + batch_size*o) ) = bstore[bb + (bblocksize+1)*ii ];	  
	  ii += bblocksize;
	}
      }

    });
  
  return into;
}
template<typename FloatType, int Dim>
void untransformBatchVector(int vecdim, Tensor<FloatType,Dim> &tens, const Vector<FloatType> &from){
  assert(vecdim != Dim-1); //batch dim
  autoView(tens_v,tens,DeviceWrite);
  autoView(from_v,from,DeviceRead);

  int stride = tensorDimensionStride<Dim>(vecdim, tens.sizeArray());
  int other_size_lin=1;
  for(int i=0;i<Dim-1;i++)
    if(i != vecdim) other_size_lin *= tens.size(i);
  int batch_size = tens.size(Dim-1);
  int isize = tens_v.size(vecdim);
  
  constexpr int iblocksz = 32;
  int iblocks = (isize + iblocksz -1)/iblocksz;

  constexpr int bblocksz = 32;
  int bblocks = (batch_size + bblocksz - 1)/bblocksz;
  
  accelerator_for_4d_gen(1,3,shm( (iblocksz+1)*bblocksz*sizeof(FloatType)), t, iblocksz, bi,iblocks,  bblock, bblocks, o, other_size_lin,{
      FloatType* bstore = (FloatType*)shared;
      int ioff = bi*iblocksz;
      int iblocksz_actual = min(isize - ioff, iblocksz);

      int boff = bblock*bblocksz;
      int bblocksz_actual = min(batch_size - boff, bblocksz);

      //parallel load iblocksz data into shared
      for(int bb=0;bb<bblocksz_actual;bb++)
	if(t < iblocksz_actual) bstore[t + (iblocksz+1)*bb] = from_v( t + ioff + isize*(bb + boff + batch_size*o ) );

      acceleratorSynchronizeBlock();
      
      FloatType* tens_p = tens_v.data() + batchTensorDimensionBaseLin<Dim>(vecdim, boff, o, tens_v.sizeArray());

      //parallel write bblocksz into output
      for(int ii=0;ii<iblocksz_actual;ii++){
	int bb =t;
	while( bb < bblocksz_actual){
	  *(tens_p + bb + (ii + ioff)*stride) = bstore[ii + (iblocksz+1)*bb];
	  bb+=iblocksz;
	}
      }
	  
    });
}
