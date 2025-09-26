#pragma once

#include<vector>
#include <array>
#include <memory>
#include <cassert>
#include <iostream>
#include <sstream>
#include <ManagedArray.hpp>

/**
 * @brief Compute the linear size of a tensor of dimension "Dim" and the provided dimensions
 * @param dims The tensor dimension (array of size Dim)
 */
template<size_t Dim>
accelerator_inline size_t tensorSize(int const* dims);

/**
 * @brief Compute the linear (pointer) offset of a specific coordate within a tensor of dimension "Dim" and the provided dimensions
 * @param coor The coordinate (array of size Dim) 
 * @param dims The tensor dimension (array of size Dim)
 */
template<size_t Dim>
accelerator_inline size_t tensorOffset(int const* coord, int const* dims);

/**
 * @brief Compute the coordinate associated with a specific linear (pointer) offset for a tensor of dimension "Dim" and the provided dimensions
 * @param [out] coord The tensor coordinate (array of size Dim)
 * @param [in] dims The tensor dimension (array of size Dim)
 * @param [in] offset The input linear offset
 */
template<size_t Dim>
accelerator_inline void tensorOffsetUnmap(int * coord, int const* dims, size_t offset);

/**
 * @brief Compute the stride for iterating over a specific dimension for a tensor of dimension "Dim" with the provided dimensions
 * @param iter_dim The dimension that will be iterated over
 * @param size The tensor dimension (array of size Dim)
 */
template<int Dim>
accelerator_inline size_t tensorDimensionStride(int iter_dim, int const* size);

/**
 * @brief Compute the linear (pointer) offset for the base element for iterating over a specific dimension of a tensor of dimension "Dim"
 * @param iter_dim The dimension that will be iterated over
 * @param other_dim_lin The coordinates in dimensions apart from iter_dim expressed as a lexicographic linear index in descending order, e.g. z + size_z * (y + size_y * x)
 * @param size The tensor dimension (array of size Dim)
 */
template<int Dim>
accelerator_inline size_t tensorDimensionBaseLin(int iter_dim, size_t other_dim_lin, int const *size);

/**
 * @brief Compute the linear (pointer) offset for the base element for iterating over a specific dimension of a tensor of dimension "Dim"
 * @param iter_dim The dimension that will be iterated over
 * @param other_coor The coordinates for the other dimensions (array of size Dim-1)
 * @param size The tensor dimension (array of size Dim)
 */
template<int Dim>
accelerator_inline size_t tensorDimensionBase(int iter_dim, int const* other_coord, int const *size);

/**
 * @brief Compute the linear (pointer) offset for the base element for iterating over a specific dimension for a batch-tensor (last dim is the batch dimension) of dimension "Dim"
 * @param iter_dim The dimension that will be iterated over
 * @param batch_idx The batch index (coordinate in last dimension)
 * @param other_dim_lin The coordinates in dimensions apart from iter_dim and Dim-1 expressed as a lexicographic linear index in descending order, e.g. z + size_z * (y + size_y * x)
 * @param size The tensor dimension (array of size Dim)
 */
template<int Dim>
accelerator_inline size_t batchTensorDimensionBaseLin(int iter_dim, int batch_idx, size_t other_dim_lin, int const *size);

     
 
/**
 * @brief A class for tensors of arbitrary dimension and floating point type
 */
template<typename _FloatType, int Dim>
struct Tensor{
public:
  typedef _FloatType FloatType; /**< The floating point type*/
private:
  ManagedArray<FloatType> vals; /**< Memory-contiguous container for tensor data*/
  int _size[Dim]; /**< Tensor dimensions*/
public:  
  typedef const int * Dims; /**< Array type for tensor dimensions*/
  typedef const int * Coord; /**< Array type for tensor coordinates */
  enum { Dimension = Dim }; /**< The tensor dimension*/

#define _1D_TENSOR_ONLY template<int D=Dim, typename std::enable_if<D==1,int>::type = 0>
#define _2D_TENSOR_ONLY template<int D=Dim, typename std::enable_if<D==2,int>::type = 0>
#define _3D_TENSOR_ONLY template<int D=Dim, typename std::enable_if<D==3,int>::type = 0>
#define _4D_TENSOR_ONLY template<int D=Dim, typename std::enable_if<D==4,int>::type = 0>

  /**
   * @brief Return the tensor dimension
   */
  static constexpr int dimension(){ return Dim; }

  /**
   * @brief Default constructor for a zero-size tensor
   */
  Tensor(): _size{0}{}

  /**
   * @brief Construct a tensor with the provided dimensions with the initial memory allocation in the provided pool
   * @param dims The tensor dimensions
   * @param alloc_pool The memory pool for the initial allocatio (default: device)
   */
  Tensor(Dims dims, MemoryManager::Pool alloc_pool = MemoryManager::Pool::DevicePool): vals(tensorSize<Dim>(dims),alloc_pool){ memcpy(_size,dims,Dim*sizeof(int));  }

  /**
   * @brief Construct a tensor with the provided dimensions uniformly initialized with the provided value with the initial memory allocation in the provided pool
   * @param dims The tensor dimensions
   * @param init The initial value for all elements
   * @param alloc_pool The memory pool for the initial allocatio (default: device)
   */ 
  Tensor(Dims dims, FloatType init, MemoryManager::Pool alloc_pool = MemoryManager::Pool::DevicePool): vals(tensorSize<Dim>(dims),init,alloc_pool){ memcpy(_size,dims,Dim*sizeof(int));  }

  /**
   * @brief Construct a tensor with the provided dimensions initialized from the provided array. The initial memory allocation will be in the host pool.
   * @param dims The tensor dimensions
   * @param init_vals The initial values with lexicographic mapping in descending order, e.g. z + size_z * (y + size_y * x)
   */ 
  Tensor(Dims dims, const std::vector<FloatType> &init_vals): vals(init_vals){
    memcpy(_size,dims,Dim*sizeof(int));
    assert(tensorSize<Dim>(dims) == init_vals.size());
  }  

  /**
   * @brief Construct a tensor with the provided dimensions initialized from the provided array. The initial memory allocation will be in the host pool.
   * @param dims The tensor dimensions
   * @param init_vals The initial values with lexicographic mapping in descending order, e.g. z + size_z * (y + size_y * x)
   */ 
  Tensor(Dims dims, FloatType const* init_vals): vals(tensorSize<Dim>(dims)){
    memcpy(_size,dims,Dim*sizeof(int));    
    autoView(vals_v,vals,HostWrite);
    memcpy(vals_v.data(), init_vals, vals.size()*sizeof(FloatType));
  }  

  /**
   * @brief Construct a 1D tensor (vector) with the provided length with the initial memory allocation in the provided pool
   * @param len The vector length
   * @param alloc_pool The memory pool for the initial allocatio (default: device)
   */
  _1D_TENSOR_ONLY
  Tensor(int len, MemoryManager::Pool alloc_pool = MemoryManager::Pool::DevicePool): vals(len,alloc_pool){ _size[0]=len; }

  /**
   * @brief Construct a 1D tensor (vector) with the provided length uniformly initialized with the provided value with the initial memory allocation in the provided pool
   * @param len The vector length
   * @param init The initial value for all elements
   * @param alloc_pool The memory pool for the initial allocatio (default: device)
   */ 
  _1D_TENSOR_ONLY
  Tensor(int len, FloatType init, MemoryManager::Pool alloc_pool = MemoryManager::Pool::DevicePool): vals(len,init,alloc_pool){ _size[0]=len; }

  /**
   * @brief Construct a 1D tensor (vector) initialized from the provided array. The initial memory allocation will be in the host pool.
   * @param init_vals The initial values
   */
  _1D_TENSOR_ONLY
  Tensor(const std::vector<FloatType> &init_vals): vals(init_vals){ _size[0]=init_vals.size(); }  

  /**
   * @brief Construct a 2D tensor (matrix) with the provided dimensions with the initial memory allocation in the provided pool
   * @param size0 The size of the 1st dimension (number of rows)
   * @param size1 The size of the 2nd dimension (number of columns)
   * @param alloc_pool The memory pool for the initial allocatio (default: device)
   */
  _2D_TENSOR_ONLY
  Tensor(int size0, int size1, MemoryManager::Pool alloc_pool = MemoryManager::Pool::DevicePool): vals(size0*size1,alloc_pool){ _size[0]=size0; _size[1]=size1; }

  /**
   * @brief Construct a 2D tensor (matrix) with the provided dimensions, uniformly initialized with the provided value with the initial memory allocation in the provided pool
   * @param size0 The size of the 1st dimension (number of rows)
   * @param size1 The size of the 2nd dimension (number of columns)
   * @param init The initial value for all elements
   * @param alloc_pool The memory pool for the initial allocation (default: device)
   */ 
  _2D_TENSOR_ONLY
  Tensor(int size0, int size1, FloatType init, MemoryManager::Pool alloc_pool = MemoryManager::Pool::DevicePool): vals(size0*size1,init,alloc_pool){ _size[0]=size0; _size[1]=size1; }

  /**
   * @brief Construct a 2D tensor (matrix) with the provided dimensions, initialized from the provided array. The initial memory allocation will be in the host pool.
   * @param size0 The size of the 1st dimension (number of rows)
   * @param size1 The size of the 2nd dimension (number of columns)
   * @param init_vals The initial values with lexicographic mapping y + size1*x for coord (x,y)
   */
  _2D_TENSOR_ONLY
  Tensor(int size0, int size1, const std::vector<FloatType> &init_vals): vals(init_vals){ _size[0]=size0; _size[1]=size1; assert(init_vals.size() == size0*size1); }  

  /**
   * @brief Construct a 3D tensor with the provided dimensions with the initial memory allocation in the provided pool
   * @param size0 The size of the 1st dimension
   * @param size1 The size of the 2nd dimension
   * @param size2 The size of the 3rd dimension
   * @param alloc_pool The memory pool for the initial allocatio (default: device)
   */
  _3D_TENSOR_ONLY
  Tensor(int size0, int size1, int size2, MemoryManager::Pool alloc_pool = MemoryManager::Pool::DevicePool): vals(size0*size1*size2,alloc_pool){ _size[0]=size0; _size[1]=size1; _size[2]=size2; }

  /**
   * @brief Construct a 3D tensor with the provided dimensions, uniformly initialized with the provided value with the initial memory allocation in the provided pool
   * @param size0 The size of the 1st dimension
   * @param size1 The size of the 2nd dimension
   * @param size2 The size of the 3rd dimension
   * @param init The initial value for all elements
   * @param alloc_pool The memory pool for the initial allocation (default: device)
   */ 
  _3D_TENSOR_ONLY
  Tensor(int size0, int size1, int size2, FloatType init, MemoryManager::Pool alloc_pool = MemoryManager::Pool::DevicePool): vals(size0*size1*size2,init,alloc_pool){ _size[0]=size0; _size[1]=size1; _size[2]=size2; }

  /**
   * @brief Construct a 3D tensor with the provided dimensions, initialized from the provided array. The initial memory allocation will be in the host pool.
   * @param size0 The size of the 1st dimension
   * @param size1 The size of the 2nd dimension
   * @param size2 The size of the 3rd dimension
   * @param init_vals The initial values with lexicographic mapping z + size2*(y + size1*x) for coord (x,y,z)
   */
  _3D_TENSOR_ONLY
  Tensor(int size0, int size1, int size2, const std::vector<FloatType> &init_vals): vals(init_vals){ _size[0]=size0; _size[1]=size1; _size[2]=size2; assert(init_vals.size() == size0*size1*size2); }  


  /**
   * @brief Construct a 4D tensor with the provided dimensions with the initial memory allocation in the provided pool
   * @param size0 The size of the 1st dimension
   * @param size1 The size of the 2nd dimension
   * @param size2 The size of the 3rd dimension
   * @param size3 The size of the 4th dimension
   * @param alloc_pool The memory pool for the initial allocatio (default: device)
   */
  _4D_TENSOR_ONLY
  Tensor(int size0, int size1, int size2, int size3, MemoryManager::Pool alloc_pool = MemoryManager::Pool::DevicePool): vals(size0*size1*size2*size3,alloc_pool){ _size[0]=size0; _size[1]=size1; _size[2]=size2; _size[3]=size3; }

  /**
   * @brief Construct a 4D tensor with the provided dimensions, uniformly initialized with the provided value with the initial memory allocation in the provided pool
   * @param size0 The size of the 1st dimension
   * @param size1 The size of the 2nd dimension
   * @param size2 The size of the 3rd dimension
   * @param size3 The size of the 4th dimension
   * @param init The initial value for all elements
   * @param alloc_pool The memory pool for the initial allocation (default: device)
   */ 
  _4D_TENSOR_ONLY
  Tensor(int size0, int size1, int size2, int size3, FloatType init, MemoryManager::Pool alloc_pool = MemoryManager::Pool::DevicePool): vals(size0*size1*size2*size3,init,alloc_pool){
    _size[0]=size0; _size[1]=size1; _size[2]=size2; _size[3]=size3; }

  /**
   * @brief Construct a 4D tensor with the provided dimensions, initialized from the provided array. The initial memory allocation will be in the host pool.
   * @param size0 The size of the 1st dimension
   * @param size1 The size of the 2nd dimension
   * @param size2 The size of the 3rd dimension
   * @param size3 The size of the 4th dimension
   * @param init_vals The initial values with lexicographic mapping t + size3*(z + size2*(y + size1*x) ) for coord (x,y,z,t)
   */
  _4D_TENSOR_ONLY
  Tensor(int size0, int size1, int size2, int size3, const std::vector<FloatType> &init_vals): vals(init_vals){
    _size[0]=size0; _size[1]=size1; _size[2]=size2; _size[3]=size3; assert(init_vals.size() == size0*size1*size2*size3); }  

  /**
   * @brief Return the tensor dimensions as an array pointer
   */
  inline int const* sizeArray() const{ return _size; }

  /**
   * @brief Return the tensor size along a specific dimension
   * @param i The dimension
   */
  inline int size(int i) const{ return _size[i]; }

  /**
   * @brief Return the tensor dimensions as a string
   */
  std::string sizeArrayString() const{
    std::ostringstream os; os << "(";
    for(int i=0;i<Dim-1;i++) os << this->size(i) << " ";
    os << this->size(Dim-1) << ")";
    return os.str();
  }

  /**
   * @brief Return the linear dimension (flattened size) of the tensor, or equivalently, the total number of elements
   */
  inline size_t data_len() const{ return vals.size(); }

  /**
   * The tensor View accessor class
   */
  class View: private ManagedArray<FloatType>::View{
    typedef typename ManagedArray<FloatType>::View Base;
    int _size[Dim];
  public:
    /**
     * @brief Construct a view with a specific view mode and parent object
     * @param mode The view mode
     * @param parent The parent object
     */
    inline View(ViewMode mode, const Tensor<FloatType,Dim> &parent): Base(mode, parent.vals){
      memcpy(_size,parent._size,Dim*sizeof(int)); //note, this constructor will only ever be called on the *host* so it's safe to use memcpy    
    }

    /**
     * @brief Free the view. This *must* be called explicitly once the view is no longer needed
     */
    inline void free(){   
      return this->Base::free();
    }

    /**
     * @brief Access the tensor at the provided coordinate
     */
    accelerator_inline FloatType & operator()(const Coord coord) const{
      return this->Base::operator[](tensorOffset<Dim>(coord, _size));
    }
    
    /**
     * @brief Access the 1D tensor at the index (i)
     */
    _1D_TENSOR_ONLY
    accelerator_inline FloatType & operator()(int i) const{
      return this->Base::operator[](i);
    }

    /**
     * @brief Access the 2D tensor at the coordinate (i,j)
     */
    _2D_TENSOR_ONLY
    accelerator_inline FloatType & operator()(int i,int j) const{
      return this->Base::operator[](j+_size[1]*i);
    }

    /**
     * @brief Access the 3D tensor at the coordinate (i,j,k)
     */
    _3D_TENSOR_ONLY
    accelerator_inline FloatType & operator()(int i,int j,int k) const{
      return this->Base::operator[](k + _size[2]*(j+_size[1]*i));
    }

    /**
     * @brief Access the 4D tensor at the coordinate (i,j,k,l)
     */
    _4D_TENSOR_ONLY
    accelerator_inline FloatType & operator()(int i,int j,int k,int l) const{
      return this->Base::operator[](l+_size[3]*(k + _size[2]*(j+_size[1]*i)));
    }

    /**
     * @brief Return a pointer to the underlying array
     */
    accelerator_inline FloatType* data() const{ return this->Base::data(); }

    /**
     * @brief Return the linear dimension (flattened size) of the tensor, or equivalently, the total number of elements
     */
    accelerator_inline size_t data_len() const{ return this->Base::size(); }

    /**
     * @brief Return the tensor size along a specific dimension
     * @param i The dimension
     */
    accelerator_inline size_t size(int i) const{ return _size[i]; }

    /**
     * @brief Return the tensor dimensions as an array pointer
     */
    accelerator_inline int const* sizeArray() const{ return _size; }

    /**
     * @brief Access a tensor element at a coordinate expressed such that the first Dim-2 dimensions are expressed lexicographically
     * @param i The first Dim-2 dimensions expressed lexicographically in descending order (e.g. z+sizez*(y+sizey*x))
     * @param j The index of dimension Dim-2
     * @param k The index of dimension Dim-1
     */
    accelerator_inline FloatType & compact3(int i,int j,int k) const{
      return *( this->Base::data() + k + _size[Dim-1]*( j + _size[Dim-2]*i ) );
    }
  };

  /**
   * @brief Return a view to this tensor opened with a specific view mode
   */
  View view(ViewMode mode) const{
    return View(mode, *this);
  }

  /**
   * @brief "Lock" the memory region associated with this object such that it cannot be auto-evicted to free space in a memory pool. A possible use case is to ensure a memory region remains valid while performing an asynchronouse background copy
   */
  inline void lock() const{ vals.lock(); }
  
  /** 
   * @brief "Unlock the memory region, allowing it to be evicted. This is the default state.
   */
  inline void unlock() const{ vals.unlock(); }

  /**
   * @brief Return a tensor where the last dimension contains the slice between idx_start and idx_end (inclusive). E.g., for a 3D tensor T, return T(:,:,idx_start:idx_end+1)
   */
  Tensor sliceLastDimension(int idx_start, int idx_end) const;

  /**
   * @brief Insert a tensor for which the last dimension contains a slice inserted between idx_start and idx_end (inclusive). E.g., for a 3D tensor T,  T(:,:,idx_start:idx_end+1) = ins(:,:,:)
   */
  void insertSliceLastDimension(const Tensor &ins, int idx_start, int idx_end) const;
  
  /**
   * @brief Insert a tensor of Dim-1 such that (*this)(i,j,k,..., idx) = ins(i,j,k,...). E.g., for a 3D tensor T and 2D input I, set T[:,:,idx] = I[:,:]
   * @param ins The Dim-1 dimensional tensor to insert
   * @param idx The index in the last dimension on which to insert the tensor
   */
  void pokeLastDimension(const Tensor<FloatType,Dim-1> &ins, const int idx);

  /**
   * @brief Return a tensor of dimension Dim-1 such that out(i,j,k,...) = (*this)(i,j,k,..., idx). E.g., for a 3D tensor T, return T[:,:,idx]
   * @param idx The index in the last dimension on which to insert the tensor
   */  
  Tensor<FloatType,Dim-1> peekLastDimension(const int idx) const;

  /**
   * @brief Return true if the data is resident and up-to-date on the device
   */
  inline bool deviceResident() const{ return vals.deviceResident(); }

  /**
   * @brief Up/down-cast the floating point type. If loc == Auto and this tensor is device-resident, the copy will be made on the device, else on the host
   */
  template<typename FloatTypeOut>
  Tensor<FloatTypeOut,Dim> convertFloatType(Locale loc = Auto) const;
};

#undef _1D_TENSOR_ONLY
#undef _2D_TENSOR_ONLY
#undef _3D_TENSOR_ONLY
#undef _4D_TENSOR_ONLY

/**
 * @brief Alias vector to 1D tensor
 */
template<typename FloatType>
using Vector = Tensor<FloatType,1>;

/**
 * @brief Alias matrix to 2D tensor
 */
template<typename FloatType>
using Matrix = Tensor<FloatType,2>;

/**
 * @brief Insert a vector as particular column of a matrix, i.e.  into(:,col) = data(:)
 * @param The target matrix
 * @param col The column index
 * @param data The input column
 */
template<typename FloatType>
void pokeColumn(Matrix<FloatType> &into, int col, const Vector<FloatType> &data);

/**
 * @brief Insert a vector as particular row of a matrix, i.e.  into(row,:) = data(:)
 * @param The target matrix
 * @param row The row index
 * @param data The input row
 */
template<typename FloatType>
void pokeRow(Matrix<FloatType> &into, int row, const Vector<FloatType> &data);


/** 
 * @brief Retrieve a specific column of a matrix m, i.e. return m(:,col)
 * @param m The matrix
 * @param col The column index
 */
template<typename FloatType>
Vector<FloatType> peekColumn(const Matrix<FloatType> &m, int col);

/** 
 * @brief Retrieve multiple consecutive columns of a matrix m, i.e. return m(:,col_start:col_end+1)
 * @param m The matrix
 * @param col_start The first column index
 * @param col_end The last column index
 */
template<typename FloatType>
Matrix<FloatType> peekColumns(const Matrix<FloatType> &m, int col_start, int col_end);

/** 
 * @brief Insert multiple consecutive columns of a matrix m, i.e. into(:,col_start:col_end+1) = cols(:,:)
 * @param into The matrix in which to insert the columns
 * @param col_start The first column index
 * @param col_end The last column index
 * @param cols The matrix containing the columns (#cols = col_end-col_start+1) 
 */
template<typename FloatType>
void pokeColumns(Matrix<FloatType> &into, int col_start, int col_end, const Matrix<FloatType> &cols);

/**
 * @brief Extract a slice/subset of a tensor based on indices in a given dimension, 
 *        e.g. for a 3-tensor X and dimension=1, return  X[:,indices,:]
 * @param from The tensor to slice
 * @param indices The indices along the slice dimension to retain
 * @param dimension The dimension along which to slice
 * @param loc The locale in which the operation is performed. If set to Auto (default) it will be performed on the device if from is device-resident, else on the host
 */
template<int Dim, typename FloatType>
Tensor<FloatType,Dim> dimensionSlice(const Tensor<FloatType,Dim> &from, const std::vector<int> &indices, const int dimension, Locale loc = Auto);

/**
 * @brief Extract a slice/subset of a tensor based on indices in a given dimension, 
 *        e.g. for a 3-tensor X and dimension=1, return  X[:,indices,:]
 * @param from The tensor to slice
 * @param indices A *host* pointer to the array of indices along the slice dimension to retain
 * @param nidx The number of indices / size of the sliced output dimension
 * @param dimension The dimension along which to slice
 * @param loc The locale in which the operation is performed. If set to Auto (default) it will be performed on the device if from is device-resident, else on the host
 */
template<int Dim, typename FloatType>
Tensor<FloatType,Dim> dimensionSlice(const Tensor<FloatType,Dim> &from, int const* indices, int nidx, const int dimension, Locale loc = Auto);


/**
 * @brief A struct to contain normalization factors, allowing for unnormalization
 */
template<typename FloatType, int Dim>
struct normalization{
  Tensor<FloatType,Dim> mean;
  Tensor<FloatType,Dim> std;
  FloatType epsilon;

  normalization(int const* tens_sz, FloatType epsilon, MemoryManager::Pool pool): mean(tens_sz,0.,pool), std(tens_sz,0.,pool), epsilon(epsilon) {}    
};

/**
 * @brief Normalize a tensor along a specific dimension. The normalization factors (mean, std) will be returned for each orthogonal dimension
 * @param tens The tensor to normalize
 * @param dimension The dimension along which to normalize
 * @param loc The locale on which the operation is performed. If set to Auto (default), it will be performed on the device if the tensor is device-resident, else on the host
 * @param epsilon A small offset for numerical stability
 */
template<int Dim, typename FloatType>
normalization<FloatType,Dim-1> normalize(Tensor<FloatType,Dim> &tens, const int dimension, Locale loc = Auto, FloatType epsilon = FloatType(0.));

/**
 * @brief Unnormalize a tensor along a specific dimension using the pre-computed normalization factors (mean, std)
 * @param tens The tensor to normalize
 * @param dimension The dimension along which to normalize
 * @param nrm The precomputed normalization factors
 * @param loc The locale on which the operation is performed. If set to Auto (default), it will be performed on the device if the tensor is device-resident, else on the host
 */
template<int Dim, typename FloatType>
void unnormalize(Tensor<FloatType,Dim> &tens, const int dimension, const normalization<FloatType,Dim-1> &nrm, Locale loc = Auto);

/**
 * @brief Transpose a matrix. If loc = Auto (default), the operation will be performed on the device, else on the host
 */
template<typename FloatType>
Matrix<FloatType> transpose(const Matrix<FloatType> &m, Locale loc = Auto);

/**
 * @brief Output a vector to a stream
 */
template<typename FloatType>
std::ostream & operator<<(std::ostream &os, const Vector<FloatType> &v);

/**
 * @brief Output a matrix to a stream
 */
template<typename FloatType>
std::ostream & operator<<(std::ostream &os, const Matrix<FloatType> &v);

/**
 * @brief Perform the matrix-vector product of A and x
 */
template<typename FloatType>
Vector<FloatType> operator*(const Matrix<FloatType> &A, const Vector<FloatType> &x);

/**
 * @brief Addition-assignment operator for tensors
 */
template<typename FloatType,int Dim>
Tensor<FloatType,Dim> & operator+=(Tensor<FloatType,Dim> &a, const Tensor<FloatType,Dim> &b);

/**
 * @brief Addition operator for tensors
 */
template<typename FloatType, int Dim>
Tensor<FloatType,Dim> operator+(const Tensor<FloatType,Dim> &a, const Tensor<FloatType,Dim> &b);

/**
 * @brief Subtraction-assignment operator for tensors
 */
template<typename FloatType,int Dim>
Tensor<FloatType,Dim> & operator-=(Tensor<FloatType,Dim> &a, const Tensor<FloatType,Dim> &b);

/**
 * @brief Subtraction operator for tensors
 */
template<typename FloatType, int Dim>
Tensor<FloatType,Dim> operator-(const Tensor<FloatType,Dim> &a, const Tensor<FloatType,Dim> &b);

/**
 * @brief Scalar multiplication-assignment operator for tensors
 */
template<typename FloatType, int Dim>
Tensor<FloatType,Dim> & operator*=(Tensor<FloatType,Dim> &a, FloatType eps);

/**
 * @brief Scalar left-multiplication operator for tensors
 */
template<typename FloatType, int Dim>
Tensor<FloatType,Dim> operator*(FloatType eps, const Tensor<FloatType,Dim> &b);

/**
 * @brief Scalar right-multiplication operator for tensors
 */
template<typename FloatType, int Dim>
inline Tensor<FloatType,Dim> operator*(const Tensor<FloatType,Dim> &b, FloatType eps){ return eps*b; }

/**
 * @brief "Flatten" a tensor into a vector. The output mapping is lexicographic in descending order, e.g. (x,y,z) -> z + sizez*(y + sizey*x)
 */
template<int Dim, typename FloatType>
Vector<FloatType> flatten(const Tensor<FloatType,Dim> &t);

/**
 * @brief "Flatten" a tensor into a pre-allocated *host* array and return the pointer to the element of the array one past the flattened tensor. 
 * @param host_ptr The host array destination. The output mapping is lexicographic in descending order, e.g. (x,y,z) -> z + sizez*(y + sizey*x)
 * @param in The input tensor
 * @return A pointer to the element of the array one past the flattened tensor. 
 * note, the copy is performed on the host side
 */
template<int Dim, typename FloatType>
FloatType * flatten(FloatType* host_ptr, const Tensor<FloatType,Dim> &in);

/**
 * @brief "Unflatten" vector into tensor
 * @param out The output tensor. Its dimensions should be set correctly prior to calling this function
 * @param t The input vector. The input mapping is lexicographic in descending order, e.g. (x,y,z) -> z + sizez*(y + sizey*x)
 */
template<int Dim, typename FloatType>
void unflatten(Tensor<FloatType,Dim> &out, const Vector<FloatType> &t);

/**
 * @brief "Unflatten" a tensor from a pre-allocated *host* array and return the pointer to the element of the array one past the flattened tensor. 
 * @param out The output tensor. Its dimensions should be set correctly prior to calling this function
 * @param host_ptr The input array pointer. The input mapping is lexicographic in descending order, e.g. (x,y,z) -> z + sizez*(y + sizey*x)
 * @return A pointer to the element of the array one past the flattened tensor. 
 * note, the copy is performed on the host side
 */
template<int Dim, typename FloatType>
FloatType const* unflatten(Tensor<FloatType,Dim> &out, FloatType const* host_ptr);

/**
 * @brief Flatten two tensors into a single contiguous array. 
 * @param t1 The first tensor
 * @param t2 The second tensor
 * @return An output vector of length t1.data_len() + t2.data_len(), where the elements within the sub-arrays are obtained from their corresponding tensor via a lexicographic mapping in descending order, e.g. (x,y,z) -> z + sizez*(y + sizey*x)
 */
template<int Dim1, int Dim2, typename FloatType>
Vector<FloatType> flatten2(const Tensor<FloatType,Dim1> &t1, const Tensor<FloatType,Dim2> &t2);

/**
 * @brief Unflatten two tensors from a single contiguous array
 * @param[out] t1 The first tensor
 * @param[out] t2 The first tensor
 * @param[in] v An input vector of length t1.data_len() + t2.data_len(), where the elements within the sub-arrays map to their corresponding tensor coordinates via a lexicographic mapping in descending order, e.g. (x,y,z) -> z + sizez*(y + sizey*x)
 *
 * The output tensor dimensions should be set appropriately prior to calling this function
 */
template<int Dim1, int Dim2, typename FloatType>
void unflatten2(Tensor<FloatType,Dim1> &t1,  Tensor<FloatType,Dim2> &t2, const Vector<FloatType> &v);

/**
 * @brief Flatten N tensors of the same dimension into a single contiguous array
 * @param tens An array of pointers to input tensors
 * @param N The number of tensors
 * @return An output vector of length \sum_i tens[i].data_len(), where the elements within the sub-arrays are obtained from their corresponding tensor via a lexicographic mapping in descending order, e.g. (x,y,z) -> z + sizez*(y + sizey*x)
 */
template<int Dim, typename FloatType>
Vector<FloatType> flattenNsameDim(Tensor<FloatType,Dim> const* const* tens, int N);

/**
 * @brief Unflatten N tensors of the same dimension from a single contiguous array. 
 * @param tens The output tensor array. The tensor dimensions should be set appropriately prior to calling this function. 
 * @param N The number of tensors
 * @param v The input vector. This must have length \sum_i tens[i].data_len(), where the elements within the sub-arrays map to their corresponding tensor coordinates via a lexicographic mapping in descending order, e.g. (x,y,z) -> z + sizez*(y + sizey*x)
 */
template<int Dim, typename FloatType>
void unflattenNsameDim(Tensor<FloatType,Dim>* const* tens, int N, const Vector<FloatType> &v);


/**
 * @brief Flatten a batched tensor to a batch-vector, Tensor<FloatType,Dim> -> Matrix<FloatType> where the matrix column count is the batch size and the row count is the product of the size of all tensor dimensions other than the batch size
 */
template<int Dim, typename FloatType>
Matrix<FloatType> flattenToBatchVector(const Tensor<FloatType,Dim> &tens);

/**
 * @brief Unflatten a batched vector to a batch-vector, Matrix<FloatType> -> Tensor<FloatType,Dim> where the matrix column count is the batch size and the row count is the product of the size of all tensor dimensions other than the batch size.
 */
template<int Dim, typename FloatType>
Tensor<FloatType,Dim> unflattenFromBatchVector(const Matrix<FloatType> &vec, int const *tens_dim);

/**
 * @brief Concatenate (stack) Ntens tensors along a dimension concat_dim < Dim-1 (last dim is assumed to be the batch index). 
 * @param in The input tensor array
 * @param Ntens The number of tensors
 * @param concat_dim The dimension along which the concatenation is performed
 *
 * Dimensions other than concat_dim must all have the same size.
 */
template<int Dim, typename FloatType>
Tensor<FloatType,Dim> batchTensorConcatenate(Tensor<FloatType,Dim> const* const* in, int Ntens, int concat_dim);

/**
 * @brief Split a tensor along a dimension split_dim < Dim-1  (last dim is the batch index) into multiple tensors. 
 * @param out The output tensors. These should be pre-initialized to the appropriate sizes. 
 * @param Ntens The number of output tensors
 * @param in The input tensor
 * @param split_dim The dimension along which to split
 *
 * Dimensions other than split_dim must all have the same size.
 */
template<int Dim, typename FloatType>
void batchTensorSplit(Tensor<FloatType,Dim>* const* out, int Ntens, const Tensor<FloatType,Dim> &in, int split_dim);

/**
 * @brief Return the tensor norm^2, i.e.  \sum_{i,j,k,...} T[i,j,k,...]^2
 */
template<int Dim, typename FloatType>
double norm2(const Tensor<FloatType,Dim> &T);


/**
 * @brief Interpret a batched-tensor (last dim is the batch index) as an array of matrices with the provided row and column dimensions 
 *        The output data are rearranged such that these matrices are contiguous in row-major, suitable for BLAS libraries
 * @param rowdim The dimension of the input tensor that is interpreted as the output matrix row dimension
 * @param coldim The dimension of the input tensor that is interpreted as the output matrix column dimension
 * @param tens The input tensor
 * @return An array of contiguous matrices in row-major format
 */
template<typename FloatType, int Dim>
Vector<FloatType> transformBatchMatrix(int rowdim, int coldim, const Tensor<FloatType,Dim> &tens);

/**
 * @brief Perform the inverse operation of transformBatchMatrix, taking an array of matrices with the provided row and column dimensions and interpreting them as a batched-tensor (last dim is the batch index)
 * @param rowdim The dimension of the output tensor that is interpreted as the output matrix row dimension
 * @param coldim The dimension of the output tensor that is interpreted as the output matrix column dimension
 * @param tens The output tensor, setup prior to the call to the appropriate dimension
 * @param from An array of contiguous matrices in row-major format
 */
template<typename FloatType, int Dim>
void untransformBatchMatrix(int rowdim, int coldim, Tensor<FloatType,Dim> &tens, Vector<FloatType> &from);

/**
 * @brief Interpret a batched-tensor (last dim is the batch index) as an array of vectors with the provided vector dimension
 *        The output data are rearranged such that these matrices are contiguous in row-major, suitable for BLAS libraries
 * @param vecdim The dimension of the input tensor that is interpreted as the output vector dimension
 * @param tens The input tensor
 * @return An array of contiguous vectors
 */
template<typename FloatType, int Dim>
Vector<FloatType> transformBatchVector(int vecdim, const Tensor<FloatType,Dim> &tens);


/**
 * @brief Perform the inverse operation of transformBatchVector, taking an array of vectors with the provided vector dimension and interpreting them as a batched-tensor (last dim is the batch index)
 * @param vecdim The dimension of the output tensor that is interpreted as the vector dimension
 * @param tens The output tensor, setup prior to the call to the appropriate dimension
 * @param from An array of contiguous vectors
 */
template<typename FloatType, int Dim>
void untransformBatchVector(int vecdim, Tensor<FloatType,Dim> &tens, const Vector<FloatType> &from);
  
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
