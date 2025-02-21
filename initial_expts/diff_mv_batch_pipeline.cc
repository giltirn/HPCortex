#include<iostream>
#include<cmath>
#include<vector>
#include <random>
#include <algorithm>
#include <cassert>
#include <array>
#include <memory>
#include <mpi.h>
#include "RingBuffer.h"

//ML interface with batching
struct LeafTag{};

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


class InputLayer{
public:
  typedef LeafTag tag;
  
  inline InputLayer(){}
  inline InputLayer(InputLayer &&r) = default;
  inline InputLayer(const InputLayer &r) = delete;
  
  inline const Matrix &value(const Matrix &x){
    return x;
  }

  inline void deriv(Vector &cost_deriv, int off, const Matrix &above_deriv, Matrix* input_above_deriv_copyback = nullptr) const{
    if(input_above_deriv_copyback) *input_above_deriv_copyback = above_deriv;
  }
  
  inline void update(int off, const Vector &new_params){}

  inline void step(int off, const Vector &derivs, double eps){}
  
  inline int nparams() const{ return 0; }

  inline void getParams(Vector &into, int off){}

  //For pipelining
  inline void resizeInputBuffer(size_t to){}
};

inline InputLayer input_layer(){ return InputLayer(); }

struct StorageTag{};

template<typename T>
struct LeafStore{
  T v;
  typedef StorageTag tag;
  typedef T type;
  
  LeafStore(T && v): v(std::move(v)){
    //std::cout << "STORE" << std::endl;
  }
  LeafStore(const LeafStore &r) = delete;
  LeafStore(LeafStore &&r): v(std::move(r.v)){}
  
};
template<typename T>
struct LeafRef{
  T &v;
  typedef StorageTag tag;
  typedef T type;
  
  LeafRef(T &v): v(v){
    //std::cout << "REF" << std::endl;
  }
  LeafRef(const LeafRef &r) = delete;
  LeafRef(LeafRef &&r): v(r.v){}

};



template<typename T>
struct deduceStorage{};
template<typename T>
struct deduceStorage<T&>{
  typedef LeafRef<T> type;
};

template<typename T>
struct deduceStorage<T&&>{
  typedef LeafStore<T> type;
};

#define DDST(a) typename deduceStorage<decltype(a)>::type

#define ISLEAF(a) std::is_same<typename std::decay<a>::type::tag,LeafTag>::value

#define ISSTORAGE(a) std::is_same<typename std::decay<a>::type::tag,StorageTag>::value

class ReLU{
public: 
  inline Matrix operator()(const Matrix &x) const{
    int dim = x.size(0);
    int batch_size = x.size(1);
    Matrix out(dim,batch_size,1.0);
    //f(x)_i = max(x_i, 0)
    for(int i=0;i<dim;i++)
      for(int b=0;b<batch_size;b++)
	if(x(i,b) <= 0.) out(i,b) = 0.;
    return out;
  }
};

class noActivation{
public:
  inline Matrix operator()(const Matrix &x) const{
    return Matrix(x.size(0),x.size(1),1.0);
  }
};

  


template<typename Store, typename ActivationFunc>
class DNNlayer{
  Matrix weights;
  Vector bias;  
  Store leaf;
  int size0;
  int size1;

  ActivationFunc activation_func;

  //Storage from last call to "value"
  //Buffer size > 1 depending on rank if doing pipelining
  RingBuffer<Matrix> leaf_buf;
  RingBuffer<Matrix> activation_buf;
  size_t calls;

  int rank;
  bool pipeline_mode;
public:
  typedef LeafTag tag;
  
  DNNlayer(Store &&leaf, const Matrix &weights,const Vector &bias, const ActivationFunc &activation_func):
    leaf(std::move(leaf)), weights(weights), bias(bias),
    size0(weights.size(0)), size1(weights.size(1)),
    activation_func(activation_func), leaf_buf(1), calls(0), pipeline_mode(false)
  {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  }
  DNNlayer(const DNNlayer &r) = delete;
  DNNlayer(DNNlayer &&r) = default;
  
  //Forward pass
  Matrix value(const Matrix &x){
    ++calls;
    
    Matrix in = leaf.v.value(x);
    int batch_size = x.size(1);   
    assert(in.size(0) == size1);
    assert(in.size(1) == batch_size);

    leaf_buf.push(in);
    //if(pipeline_mode) std::cout << "RANK " << rank << " " << this << " CALL " << calls << " INPUT " << x << " VALUE BUFFERED INPUT " << in << std::endl;
    //else std::cout << "RANK " << rank << " " << this << " UNPIPELINED CALL " << calls << " INPUT " << x << " VALUE BUFFERED INPUT " << in << std::endl;
    
    Matrix out(size0,batch_size,0.0);

    for(int i=0;i<size0;i++){
      for(int b=0;b<batch_size;b++){
	out(i,b) = bias(i);
	for(int j=0;j<size1;j++)
	  out(i,b) += weights(i,j)* in(j,b);
      }
    }
	
    Matrix activation = activation_func(out);
    assert(activation.size(0) == size0);
    assert(activation.size(1) == batch_size);
    
    for(int i=0;i<size0;i++)
      for(int b=0;b<batch_size;b++)
	out(i,b) *= activation(i,b);    

    activation_buf.push(activation);
    
    return out;
  }
 
  void deriv(Vector &cost_deriv, int off, const Matrix &above_deriv, Matrix* input_above_deriv_copyback = nullptr) const{
    assert(above_deriv.size(0) == size0);
    Matrix in = leaf_buf.pop();
    Matrix activation = activation_buf.pop();
    int batch_size = in.size(1);

    //if(pipeline_mode) std::cout << "RANK " << rank << " " << this << " CALL " << calls << " DERIV USING BUFFERED INPUT " << in << " ABOVE_DERIV " << above_deriv << " WITH INPUT COST DERIV " << cost_deriv;
    //else std::cout << "RANK " << rank << " " << this << " UNPIPELINED CALL " << calls << " DERIV USING BUFFERED INPUT " << in << " ABOVE_DERIV " << above_deriv << " WITH INPUT COST DERIV " << cost_deriv;
    
    //for reverse differentiation, we pass down the derivatives with respect to our inputs
    //f(x)_i = act_i b_i + \sum_j act_i w_ij x_j
    //dcost / dx_j = \sum_i dcost/df_i df_i/dx_j
    //df_i/dx_j = act_i w_ij
    Matrix layer_deriv(size1,batch_size,0.);
    for(int j=0;j<size1;j++)
      for(int i=0;i<size0;i++)
	for(int b=0;b<batch_size;b++)
	  layer_deriv(j,b) += above_deriv(i,b) * activation(i,b) * weights(i,j);

    //Now we finish up the derivs wrt our parameters
    //df(x)_i / d w_jk = delta_ij act_j x_k
    //df(x)_i / d b_j = delta_ij act_j
    //dcost / dw_jk = \sum_i dcost/df_i df_i/dw_jk = dcost/df_j * act_j * x_k
    //dcost / db_j = \sum_i dcost/df_i df_i/db_j = dcost/df_j * act_j
    int p=off;
    for(int j=0;j<size0;j++)
      for(int k=0;k<size1;k++){
	for(int b=0;b<batch_size;b++)
	  cost_deriv(p) += above_deriv(j,b) * activation(j,b) * in(k,b); //batch reduction! (assume zero-initialized)
	++p;
      }
	
    for(int j=0;j<size0;j++){
      for(int b=0;b<batch_size;b++)
	cost_deriv(p) += above_deriv(j,b) * activation(j,b);
      ++p;
    }

    //std::cout << " AND RESULT " << cost_deriv << std::endl;
    
    leaf.v.deriv(cost_deriv, p, layer_deriv, input_above_deriv_copyback);
  }

  void update(int off, const Vector &new_params){
    int p=off;
    for(int i=0;i<size0;i++)
      for(int j=0;j<size1;j++)
	weights(i,j) = new_params(p++);
    for(int i=0;i<size0;i++)
      bias(i) = new_params(p++);
    leaf.v.update(p, new_params);
  }
  void step(int off, const Vector &derivs, double eps){
    int p=off;
    for(int i=0;i<size0;i++)
      for(int j=0;j<size1;j++){
	//std::cout << "Weights " << i << " " << j << " " << weights(i,j) << " -= " << derivs(p) << "*" << eps;
	weights(i,j) -= derivs(p++) * eps;
	//std::cout << " = " <<  weights(i,j) << std::endl;
      }
    for(int i=0;i<size0;i++){
      //std::cout << "Bias " << i << " " << bias(i) << " -= " << derivs(p) << "*" << eps;
      bias(i) -= derivs(p++) * eps;
      //std::cout << " = " << bias(i) << std::endl;
    }
    leaf.v.step(p, derivs, eps);
  }

  //accumulated #params for layers here and below
  inline int nparams() const{ return size0*size1 + size0 + leaf.v.nparams(); }

  //off measured from *end*, return new off
  void getParams(Vector &into, int off){
    int p = off;
    for(int i=0;i<size0;i++)
      for(int j=0;j<size1;j++)
	into(p++) = weights(i,j);
    for(int i=0;i<size0;i++)
      into(p++) = bias(i);
    leaf.v.getParams(into, p);
  }

  //For pipelining
  inline void resizeInputBuffer(size_t to){
    //std::cout << "RANK " << rank << " " << this << " RESIZING RING BUFFERS TO " << to << std::endl;
    pipeline_mode = true;
    leaf_buf.resize(to);
    activation_buf.resize(to);
    leaf.v.resizeInputBuffer(to);
  }

};

template<typename U, typename ActivationFunc, typename std::enable_if<ISLEAF(U), int>::type = 0>
auto dnn_layer(U &&u, const Matrix &weights,const Vector &bias, const ActivationFunc &activation)->DNNlayer<DDST(u),ActivationFunc>{
  return DNNlayer<DDST(u),ActivationFunc>(std::forward<U>(u), weights, bias, activation);
}
template<typename U, typename std::enable_if<ISLEAF(U), int>::type = 0>
auto dnn_layer(U &&u, const Matrix &weights,const Vector &bias)->DNNlayer<DDST(u),noActivation>{
  return DNNlayer<DDST(u),noActivation>(std::forward<U>(u), weights, bias, noActivation());
}

class MSEcostFunc{
public:
  inline static double loss(const Matrix &y, const Matrix &ypred){
    int dim = y.size(0);
    int batch_size = y.size(1);
    
    double out = 0.;
    for(int i=0;i<dim;i++)
      for(int b=0;b<batch_size;b++)
	out += pow(ypred(i,b) - y(i,b),2);
    out /= (dim * batch_size);
    return out;
  }
  inline static Matrix layer_deriv(const Matrix &y, const Matrix &ypred){
    //for reverse differentiation, we pass down the derivatives with respect to our inputs
    //dout / dparam(i) = \sum_j 2*(ypred(j) - y(j)) * dypred(j)/dparam(i)

    //dout / dypred(i) = 2*(ypred(i) - y(i)) /dim
    int dim = y.size(0);
    int batch_size = y.size(1);
    
    Matrix layer_deriv(dim,batch_size);
    for(int i=0;i<dim;i++)
      for(int b=0;b<batch_size;b++)
	layer_deriv(i,b) = 2*(ypred(i,b) - y(i,b)) / (dim*batch_size);
    return layer_deriv;
  }
};


template<typename Store, typename CostFunc>
class CostFuncWrapper{
  Store leaf;
  Matrix ypred; //dim * batch_size
  Matrix yval;
  CostFunc cost;
  int nparam;
public:
  CostFuncWrapper(Store &&leaf, const CostFunc &cost = CostFunc()): cost(cost), leaf(std::move(leaf)), nparam(leaf.v.nparams()){}
  
  double loss(const Matrix &x, const Matrix &y){
    ypred = leaf.v.value(x);
    int dim = y.size(0);
    int batch_size = y.size(1);
    assert(ypred.size(0) == dim);
    assert(ypred.size(1) == batch_size);
    
    yval = y;
    return cost.loss(y,ypred);
  }
  Vector deriv() const{
    Matrix layer_deriv = cost.layer_deriv(yval, ypred);

    Vector cost_deriv(nparam,0.);    //zero initialize
    leaf.v.deriv(cost_deriv, 0, layer_deriv);
    return cost_deriv;
  }

  Matrix predict(const Matrix &x){
    return leaf.v.value(x);
  }

  void update(const Vector &new_params){
    leaf.v.update(0, new_params);
  }
  void step(const Vector &derivs, double eps){
    leaf.v.step(0,derivs,eps);
  }
  int nparams(){ return nparam; }

  Vector getParams(){
    Vector out(nparams());
    leaf.v.getParams(out,0);
    return out;
  }
};
  
    
template<typename U, typename std::enable_if<ISLEAF(U), int>::type = 0>
auto mse_cost(U &&u)->CostFuncWrapper<DDST(u), MSEcostFunc>{
  return CostFuncWrapper<DDST(u), MSEcostFunc>(std::forward<U>(u));
}


template<typename PipelineBlockType, typename CostFunc>
class PipelineCostFuncWrapper{
  PipelineBlockType &block;

  RingBuffer<Matrix> yval_buf_v;//buffered yvalues for calls to "loss"
  //RingBuffer<Matrix> yval_buf_d;//buffered yvalues for calls to "deriv" (these have a longer lag time)
  //RingBuffer<Matrix> ypred_buf_d;//buffered ypred for calls to "deriv" 
  size_t calls;

  Matrix ypred; //dim * batch_size
  Matrix yval; //yval associated with ypred
  
  CostFunc cost;
  int nparam;
  int value_lag;
  int deriv_lag;

  int rank;
public:
  PipelineCostFuncWrapper(PipelineBlockType &block, const CostFunc &cost = CostFunc()): cost(cost), block(block), nparam(block.nparams()),
											value_lag(block.valueLag()), deriv_lag(block.derivLag()),
											yval_buf_v(block.valueLag()),
											calls(0){
    //yval_buf_off(0),
    //yval_buf_d(block.derivLag()-block.valueLag()+1),
    //ypred_buf_d(block.derivLag()-block.valueLag()+1),

    
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  }


  
  double loss(const Matrix &x, const Matrix &y){
    ++calls;
    int dim = y.size(0);
    int batch_size = y.size(1);
    
    //<i- (<0-, <1- etc): item i in 'prev_in' at start of iteration, perform action and send left in this iter
    //<i|                : take item i from input 'in', perform action and send left in this iter
    //iter    rank->
    //        0     1      2
    //0                   <0|
    //1            <0-    <1|
    //2      <0-   <1-    <2|
    //3      <1-   <2-    <3|
    //etc
    //value_lag = 3 = nrank

    yval_buf_v.push(y);
   
    ypred = block.value(x);
    assert(ypred.size(0) == dim);
    assert(ypred.size(1) == batch_size);
    
    if(calls < value_lag) return -1;
    else{ //yval not initialized until ring buffer is full
      yval = yval_buf_v.pop();
      assert(yval.size(0) == dim);
      assert(yval.size(1) == batch_size);
      
      return cost.loss(yval,ypred);
    }
  }
  
  Vector deriv() const{
    //3 ranks
    //Value:
    //iter    rank->
    //        0     1      2
    //0                   <0|
    //1            <0-    <1|
    //2      <0-   <1-    <2|
    //3      <1-   <2-    <3|
    //etc
    
    //value lag: 3

    //Deriv:
    //iter    rank->
    //        0     1      2
    //0       
    //1       
    //2      |0>   
    //3      |1>   -0>
    //4      |2>   -1>    -0>    
    //etc

    //deriv lag: 5

    //Notice rank 0 consumes y,ypred[i] on the same iteration it receives ypred[i], so we don't need a buffer
    if(calls >= value_lag){ //can't call before value_lag because yval uninitialized
      Matrix layer_deriv = cost.layer_deriv(yval, ypred);
      Vector cost_deriv(nparam,0.);    //zero initialize
      block.deriv(cost_deriv, layer_deriv);
      if(calls < deriv_lag) return Vector(nparam,-1.); //indicate that these derivs are invalid
      else return cost_deriv;
    }else return Vector(nparam,-1.); //indicate that these derivs are invalid
  }
  
  
};

template<typename BlockStore>
class PipelineBlock{
  BlockStore block; //this chain should terminate on an InputLayer. This represents the work done on this rank

  int rank; //current rank
  int next_rank; //rank of next comm layer, -1 indicates this is the last
  int prev_rank; //rank of previous comm layer, -1 indicates this is the first
  int pipeline_depth; //number of ranks in the pipeline
  bool is_first;
  bool is_last;
  
  int batch_size;
  int input_features; //features of data
  int this_features; //features of this layer
  int next_features; //features of output of layer to the right
  
  int nparam; //total #params
  int stage_off; //offset within parameter vector associated with this block
  
  Matrix prev_in; //input for forwards propagation

  //storage for backpropagation
  Matrix prev_above_deriv;
  Vector prev_cost_deriv_passright;
  
  inline static MPI_Request send(const Matrix &mat, int to){
    MPI_Request req;		
    MPI_Isend(mat.data(), mat.data_len(), MPI_DOUBLE, to, 0, MPI_COMM_WORLD, &req);
    return req;
  }
  inline static MPI_Request recv(Matrix &mat, int from){
    MPI_Request req;		
    MPI_Irecv(mat.data(), mat.data_len(), MPI_DOUBLE, from, 0, MPI_COMM_WORLD, &req);
    return req;
  }
  inline static MPI_Request send(const Vector &mat, int to){
    MPI_Request req;		
    MPI_Isend(mat.data(), mat.data_len(), MPI_DOUBLE, to, 0, MPI_COMM_WORLD, &req);
    return req;
  }
  inline static MPI_Request recv(Vector &mat, int from){
    MPI_Request req;		
    MPI_Irecv(mat.data(), mat.data_len(), MPI_DOUBLE, from, 0, MPI_COMM_WORLD, &req);
    return req;
  }

  template<typename T>
  void passLeft(std::vector<MPI_Request> &reqs,
		T const* send_bulk, T const *send_last,
		T* recv_first, T* recv_bulk) const{
    if(pipeline_depth == 1){
      *recv_first = *send_last; return;
    }
    
    if(is_last) reqs.push_back(send(*send_last, prev_rank));
    else if(!is_first) reqs.push_back(send(*send_bulk, prev_rank));

    if(is_first) reqs.push_back(recv(*recv_first, next_rank));
    else if(!is_last) reqs.push_back(recv(*recv_bulk, next_rank));
  }
  template<typename T>
  void passRight(std::vector<MPI_Request> &reqs,
		T const* send_first, T const *send_bulk,
		T* recv_bulk, T* recv_last) const{
    if(pipeline_depth == 1){
      *recv_last = *send_first; return;
    }
    
    if(is_first) reqs.push_back(send(*send_first, next_rank));
    else if(!is_last) reqs.push_back(send(*send_bulk, next_rank));

    if(is_last) reqs.push_back(recv(*recv_last, prev_rank));
    else if(!is_first) reqs.push_back(recv(*recv_bulk, prev_rank));
  }

  template<typename T>
  void passLeftLastToFirst(std::vector<MPI_Request> &reqs,
			   T const* send_last, T *recv_first){
    if(pipeline_depth == 1){
      *recv_first = *send_last; return;
    }
    if(is_last) reqs.push_back(send(*send_last,0));
    else if(is_first) reqs.push_back(recv(*recv_first,pipeline_depth-1));
  }

  int dcalls;
  
public:
  
  PipelineBlock(BlockStore &&_block, int batch_size, int input_features, int this_features, int next_features): block(std::move(_block)), batch_size(batch_size), input_features(input_features), this_features(this_features), next_features(next_features), dcalls(0){
    
    //Prepare rank information
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    int nranks;
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    next_rank = rank+1;
    prev_rank = rank-1;
    if(next_rank == nranks) next_rank = -1;
    pipeline_depth = nranks;

    is_first = prev_rank == -1;
    is_last = next_rank == -1;

    //Compute parameter information
    std::vector<int> block_params(nranks, 0);
    block_params[rank] = block.v.nparams();
    
    MPI_Allreduce(MPI_IN_PLACE, block_params.data(), nranks, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    //Compute block param offset
    stage_off = 0;
    for(int i=0;i<rank;i++) stage_off += block_params[i];

    //Compute total params
    nparam = 0;
    for(int i=0;i<nranks;i++) nparam += block_params[i];
    
    //Setup storage
    prev_in = Matrix(this_features, batch_size, 0.);
    prev_above_deriv = Matrix(this_features, batch_size, 0.); //dcost/dout_i  from layer above
    prev_cost_deriv_passright = Vector(nparam,0.); //partially-populated cost deriv from layer above
    
    //Tell the block to resize its input buffers accordingly (cf below)
    block.v.resizeInputBuffer(2*rank+1);
  }

  PipelineBlock(const PipelineBlock &r) = delete;
  PipelineBlock(PipelineBlock &&r) = default;
  
  int nparams() const{ return nparam; }

  int pipelineDepth() const{ return pipeline_depth; }

  //Amount of iterations before you get the return value for the first item back
  int valueLag() const{ return pipeline_depth; }
  //Amount of iterations before you get the derivative for the first item back
  int derivLag() const{ return 2*pipeline_depth - 1; }

  //We assume every node in the group has access to the same x value called in the same order
  Matrix value(const Matrix &in){
    std::vector<MPI_Request> reqs;

    //<i- (<0-, <1- etc): item i in 'prev_in' at start of iteration, perform action and send left in this iter
    //<i|                : take item i from input 'in', perform action and send left in this iter

    //iter    rank->
    //        0     1      2
    //0                   <0|
    //1            <0-    <1|
    //2      <0-   <1-    <2|
    //3      <1-   <2-    <3|
    //etc

    Matrix out = block.v.value(is_last ? in : prev_in);
    passLeft(reqs,          &out,     &out,
	          &prev_in, &prev_in);    
    MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
    
    return out; //note, output only meaningful on first node   
  }

  void deriv(Vector &cost_deriv, const Matrix &above_deriv){
    ++dcalls;
    std::vector<MPI_Request> reqs;

    //For reverse differentiation we need to wait for a full value pipeline
    //As each layer needs the input value from the layer below to compute its derivative we need to buffer these appropriately

    //2 ranks
    //Value:
    //iter    rank->
    //        0     1
    //0            <0|
    //1      <0-   <1|
    //2      <1-   <2|
    //3      <2-   <3|
    //etc

    //value lag: 2
    
    //Deriv:
    //iter    rank->
    //        0     1    
    //0       
    //1      |0>   
    //2      |1>   -0>
    //3      |2>   -1>   
    //etc

    //deriv lag: 3
    
    //3 ranks
    //Value:
    //iter    rank->
    //        0     1      2
    //0                   <0|
    //1            <0-    <1|
    //2      <0-   <1-    <2|
    //3      <1-   <2-    <3|
    //etc
    
    //value lag: 3

    //Deriv:
    //iter    rank->
    //        0     1      2
    //0       
    //1       
    //2      |0>   
    //3      |1>   -0>
    //4      |2>   -1>    -0>    
    //etc

    //deriv lag: 5


    //4 ranks
    //Value:
    //iter    rank->
    //        0     1      2     3
    //0                         <0|
    //1                  <0-    <1|
    //2            <0-   <1-    <2|
    //3      <0-   <1-   <2-    <3|
    //4      <1-   <2-   <3-    <4|
    //etc
    
    //value lag: 4

    //Deriv:
    //iter    rank->
    //        0     1      2     3
    //0       
    //1
    //2
    //3      |0>   
    //4      |1>   -0>
    //5      |2>   -1>    -0>
    //6      |3>   -2>    -1>    -0>
    //7      |4>   -3>    -2>    -1>
    //etc

    //deriv lag: 7

    //value lag: nrank
    //deriv lag: 2*nrank - 1

    
    
    //To work out the buffering reqs we track what iter the derivs and values are computed for a particular item
    //For item 0:
    //Layer    Val-iter  Deriv-iter   
    // 0          2          2        
    // 1          1          3        
    // 2          0          4

    //For item 1:
    //Layer    Val-iter  Deriv-iter   
    // 0          3          3        
    // 1          2          4        
    // 2          1          5


    //Layer 0
    //Iter     Append    Consume
    // 0         
    // 1         
    // 2         0        0
    // 3         1        1
    // 4         2        2
    //lag = 0
    //buf_sz = 1
    
    //Layer 1
    //Iter     Append    Consume
    // 0         
    // 1         0
    // 2         1
    // 3         2         0
    // 4         3         1
    //lag = 2
    //buf_sz = 3
    
    //Layer 2
    //Iter     Append    Consume
    // 0         0
    // 1         1
    // 2         2
    // 3         3         
    // 4         4         0
    // 5         5         1
    //lag = 4
    //buf_sz = 5

    //buf_sz = 2*layer_idx + 1
    
    
    //compute layer derivative and fill in cost derivative vector
    //if this is the first rank, fill the input cost_deriv, else we append it to the deriv vector received last call
    Matrix layer_deriv(next_features, batch_size); //layer deriv to send right
    Vector pass_cost_deriv(is_first ? cost_deriv : prev_cost_deriv_passright); //cost deriv to send right

    //std::cout << "RANK " << rank << " CALL " << dcalls << " INPUT COST DERIV: " << pass_cost_deriv << " and base offset " << stage_off << std::endl;
    
    block.v.deriv(pass_cost_deriv, stage_off, is_first ? above_deriv : prev_above_deriv, &layer_deriv);

    //std::cout << "RANK " << rank << " CALL " << dcalls << " RESULT COST DERIV: " << pass_cost_deriv << std::endl;
    
    //send layer deriv to right if !last
    passRight(reqs, &layer_deriv, &layer_deriv,
                                 &prev_above_deriv, &prev_above_deriv); 
    
    //send new cost deriv to right if !last
    passRight(reqs, &pass_cost_deriv, &pass_cost_deriv,
	                           &prev_cost_deriv_passright, &prev_cost_deriv_passright);

    //if last our cost derivative is fully populated, so we send it back to rank 0 for output
    passLeftLastToFirst(reqs, &pass_cost_deriv, &cost_deriv);
    
    MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);

    //if(rank == 0) std::cout << "RANK 0 CALL " << dcalls << " RECEIVED COST DERIV: " << cost_deriv << std::endl;
  }

  // inline void update(int off, const Vector &new_params){}

  // inline void step(int off, const Vector &derivs, double eps){}
  
  // inline int nparams(){ return 0; }

  // inline void getParams(Vector &into, int off){}  
  
};

template<typename U, typename std::enable_if<ISLEAF(U), int>::type = 0>
auto pipeline_block(U &&u, int batch_size, int input_features, int this_features, int next_features)->PipelineBlock<DDST(u)>{
  return PipelineBlock<DDST(u)>(std::forward<U>(u),batch_size,input_features, this_features, next_features);
}


  
struct XYpair{
  Matrix x;
  Matrix y;
};


template<typename T, typename LRscheduler>
void optimizeGradientDescent(T &model, const std::vector<XYpair> &data, const LRscheduler &lr, int nepoch){
  std::default_random_engine gen(1234);
  std::uniform_int_distribution<int> dist(0,data.size());

  int ndata = data.size();
  
  std::vector<int> didx(ndata);
  for(int i=0;i<ndata;i++) didx[i] = i;
  
  for(int epoch=0;epoch<nepoch;epoch++){
    std::random_shuffle ( didx.begin(), didx.end(), [&](const int l){ return dist(gen); }  );
    double eps = lr(epoch);
    std::cout << "Epoch " << epoch << " learning rate " << eps << std::endl;
    
    for(int ii=0;ii<ndata;ii++){
      int i = didx[ii];
      double loss = model.loss(data[i].x, data[i].y);
      std::cout << epoch << "-" << ii << " : "<< loss << std::endl;
      model.step( model.deriv(), eps );
    }
  }

}



struct AdamParams{ //NB, alpha comes from the learning scheduler
  double beta1;
  double beta2;
  double eps;
  AdamParams( double beta1=0.99, double beta2=0.999, double eps=1e-8): beta1(beta1), beta2(beta2), eps(eps){}
};
  
template<typename T, typename LRscheduler>
void optimizeAdam(T &model, const std::vector<XYpair> &data, const LRscheduler &lr, const AdamParams &ap, int nepoch){
  std::default_random_engine gen(1234);
  std::uniform_int_distribution<int> dist(0,data.size());

  int nparam = model.nparams();
  Vector m(nparam,0.0);
  Vector v(nparam,0.0);
  int t=0;
  
  int ndata = data.size();
  
  std::vector<int> didx(ndata);
  for(int i=0;i<ndata;i++) didx[i] = i;
  
  for(int epoch=0;epoch<nepoch;epoch++){
    std::random_shuffle ( didx.begin(), didx.end(), [&](const int l){ return dist(gen); }  );
    double alpha = lr(epoch);
    std::cout << "Epoch " << epoch << " learning rate " << alpha << std::endl;
    
    for(int ii=0;ii<ndata;ii++){
      int i = didx[ii];
      double loss = model.loss(data[i].x, data[i].y);
      auto g = model.deriv();

      double delta = t>0 ? alpha * sqrt(1. - pow(ap.beta2,t))  / (1. - pow(ap.beta1,t) ) : alpha;
      for(int p=0;p<nparam;p++){
	double gp_init = g(p);
	m(p) = ap.beta1 * m(p) + (1.-ap.beta1)*g(p);
	v(p) = ap.beta2 * v(p) + (1.-ap.beta2)*pow(g(p),2);

	g(p) = m(p)/(sqrt(v(p)) + ap.eps);
	//std::cout << "p="<< p << " m=" << m(p) << " v=" << v(p) << " g:" << gp_init << "->" <<  g(p) << std::endl;
      }
      ++t;      
      std::cout << epoch << "-" << ii << " : "<< loss << " update model with step size " << delta << std::endl;
      model.step( g , delta );
    }
  }

}


//TODO: Optimizer can be separate, needs to be passed just the gradient and return an ascent vector and step size
//TODO: Consider how to distribute layers over MPI. Each rank has a batch of layers. We need to keep every rank busy
//      Need distributed vectors and operations thereon

class DecayScheduler{
  double eps;
  double decay_rate;
public:
  DecayScheduler(double eps, double decay_rate): eps(eps), decay_rate(decay_rate){}
  double operator()(const int epoch) const{ return eps * 1./(1. + decay_rate * epoch); }
};


template<typename Store, typename ActivationFunc>
struct TestLeaf{
  Store leaf;
  ActivationFunc activation_func;
  typedef LeafTag tag;
  
  TestLeaf(Store &&leaf, const ActivationFunc &activation_func): leaf(std::move(leaf)), activation_func(activation_func){}
  TestLeaf(const TestLeaf &r) = delete;
  TestLeaf(TestLeaf &&r): leaf(std::move(r.leaf)){}
};
template<typename U>
auto test(U &&u)->TestLeaf<DDST(u),noActivation>{
  return TestLeaf<DDST(u),noActivation>(std::forward<U>(u),noActivation());
}


void basicTests(){
  typedef std::vector<double> vecD;
  
  Matrix w1_init(3,2, vecD({0.1,0.2,
	                   -0.1,-0.2,
			    0.7,0.7}));
  Vector b1_init( vecD({0.5,0.7,0.9}));		    
  
  auto f = mse_cost( dnn_layer(input_layer(), w1_init, b1_init) );

  //NB batch size 2, batches in different *columns*
  Matrix x1(2,2,vecD({1.3, 0.6,
	             -0.3, -1.7}));
  
  Matrix y1(3,2,std::vector<double>({-0.5, -0.5,
	                             1.7, 1.3
				     -0.7, -0.5}));

  double expect = 0.;
  for(int i=0;i<2;i++){  
    Vector y1pred = w1_init * x1.peekColumn(i) + b1_init;
    Vector y1_b = y1.peekColumn(i);
    expect += pow(y1pred(0)-y1_b(0),2)/3. + pow(y1pred(1)-y1_b(1),2)/3. + pow(y1pred(2)-y1_b(2),2)/3.;
  }
  expect /= 2.;
    
  double got=  f.loss(x1,y1);
  std::cout << "Test loss : got " << got << " expect " << expect << std::endl;


  Vector dexpect(9);
  int p=0;
  for(int i=0;i<3;i++){
    for(int j=0;j<2;j++){
      Matrix w1_p = w1_init;
      w1_p(i,j) += 1e-7;
      auto f2 = mse_cost( dnn_layer(input_layer(), w1_p, b1_init) );
      dexpect(p++) = (f2.loss(x1,y1) - got)/1e-7;
    }
  }
  for(int i=0;i<3;i++){
    Vector b1_p = b1_init;
    b1_p(i) += 1e-7;
    auto f2 = mse_cost( dnn_layer(input_layer(), w1_init, b1_p) );
    dexpect(p++) = (f2.loss(x1,y1) - got)/1e-7;    
  }

  Vector dgot = f.deriv();
  for(int i=0;i<9;i++){
    std::cout << "Test deriv wrt param " << i <<  ": got " << dgot(i) << " expect " << dexpect(i) << std::endl;
  }
    
  //test update
  Matrix w1_new(3,2, std::vector<double>({-0.5,0.4,
					  0.8,1.2,
					  2.1,-3.0}));
  Vector b1_new( std::vector<double>({-0.5,0.7,-1.1}));	

  auto ftest = mse_cost( dnn_layer(input_layer(), w1_new, b1_new) );
  f.update(ftest.getParams());
  
  std::cout << "Update check : expect " << ftest.loss(x1,y1) << " got " <<  f.loss(x1,y1) << std::endl;

}


void testSimpleLinear(){
  //Test f(x) = 0.2*x + 0.3;

  Matrix winit(1,1,0.0);
  Vector binit(1,0.0);

  int ndata = 100;
  std::vector<XYpair> data(ndata);
  for(int i=0;i<ndata;i++){
    double eps = 2.0/(ndata - 1);
    double x = -1.0 + i*eps; //normalize x to within +-1

    data[i].x = Matrix(1,1,x);
    data[i].y = Matrix(1,1,0.2*x + 0.3);
  }
    
  auto model = mse_cost( dnn_layer(input_layer(), winit, binit) );
  DecayScheduler lr(0.01, 0.1);
  optimizeGradientDescent(model, data, lr, 200);

  std::cout << "Final params" << std::endl;
  Vector final_p = model.getParams();
  for(int i=0;i<final_p.size(0);i++)
    std::cout << i << " " << final_p(i) << std::endl;

}


void testOneHiddenLayer(){
  //Test f(x) = 0.2*x + 0.3;
  int nbatch = 100;
  int batch_size = 4;
  std::vector<XYpair> data(nbatch);

  int ndata = batch_size * nbatch;

  for(int i=0;i<ndata;i++){ //i = b + batch_size * B
    double eps = 2.0/(ndata - 1);
    double x = -1.0 + i*eps; //normalize x to within +-1

    int b = i % batch_size;
    int B = i / batch_size;
    if(b==0){
      data[B].x = Matrix(1,batch_size);
      data[B].y = Matrix(1,batch_size);
    }
    
    data[B].x(0,b) = x;
    data[B].y(0,b) = 0.2*x + 0.3;
  }

  int nhidden = 5;

  Matrix winit_out(1,nhidden,0.01);
  Matrix winit_h(nhidden,1,0.01);

  Vector binit_out(1,0.01);
  Vector binit_h(nhidden, 0.01);

  auto hidden_layer( dnn_layer(input_layer(), winit_h, binit_h, ReLU()) );
  auto model = mse_cost( dnn_layer(hidden_layer, winit_out, binit_out) );

  //Test derivative
  {
    Vector p = model.getParams();
    
    for(int d=1;d<5;d++){ //first 5 data
    
      double c1 = model.loss(data[d].x,data[d].y);
      Vector pd = model.deriv();
      
      auto hidden_layer2 = dnn_layer(input_layer(), winit_h, binit_h, ReLU());  
      auto model2 = mse_cost( dnn_layer(hidden_layer2, winit_out, binit_out) );

      std::cout << "Test derivs " << d << " x=" << data[d].x(0,0) << " " << data[d].x(0,1) << std::endl;
      for(int i=0;i<p.size(0);i++){
	Vector pp(p);
	pp(i) += 1e-9;
	model2.update(pp);
      
	double c2 = model2.loss(data[d].x,data[d].y);
	std::cout << i << " got " << pd(i) << " expect " << (c2-c1)/1e-9 << std::endl;
      }
    }
  }


  DecayScheduler lr(0.001, 0.1);
  AdamParams ap;
  optimizeAdam(model, data, lr, ap, 200);

  std::cout << "Final params" << std::endl;
  Vector final_p = model.getParams();
  for(int i=0;i<final_p.size(0);i++)
    std::cout << i << " " << final_p(i) << std::endl;

  std::cout << "Test on some data" << std::endl;
  for(int d=0;d<data.size();d++){ //first 5 data, batch idx 0
    auto got = model.predict(data[d].x);
    std::cout << data[d].x(0,0) << " got " << got(0,0) << " expect " << data[d].y(0,0) << std::endl;
  }

}


//This functionality allows dynamic rather than compile time composition of layers
class LayerWrapperInternalBase{
public:
  virtual Matrix value(const Matrix &x) = 0;
  virtual void deriv(Vector &cost_deriv, int off, const Matrix &above_deriv, Matrix* input_above_deriv_copyback = nullptr) const = 0;
  virtual int nparams() const = 0;
  virtual ~LayerWrapperInternalBase(){}
};
template<typename Store, typename std::enable_if<ISSTORAGE(Store), int>::type = 0 >
class LayerWrapperInternal: public LayerWrapperInternalBase{
  Store layer;
public:
  LayerWrapperInternal(Store &&layer): layer(std::move(layer)){}
  
  Matrix value(const Matrix &x) override{
    return layer.v.value(x);
  }
  void deriv(Vector &cost_deriv, int off, const Matrix &above_deriv, Matrix* input_above_deriv_copyback = nullptr) const override{
    layer.v.deriv(cost_deriv,off,above_deriv, input_above_deriv_copyback);
  }
  int nparams() const{ return layer.v.nparams(); }
};
class LayerWrapper{
  std::unique_ptr<LayerWrapperInternalBase> layer;
public:
  typedef LeafTag tag;

  LayerWrapper(LayerWrapper &&r) = default;
  LayerWrapper & operator=(LayerWrapper &&r) = default;
  
  template<typename Store, typename std::enable_if<ISSTORAGE(Store), int>::type = 0 >
  LayerWrapper(Store &&layer): layer( new LayerWrapperInternal<Store>(std::move(layer)) ){}

  inline Matrix value(const Matrix &x){
    return layer->value(x);
  }
  inline void deriv(Vector &cost_deriv, int off, const Matrix &above_deriv, Matrix* input_above_deriv_copyback = nullptr) const{
    layer->deriv(cost_deriv,off,above_deriv, input_above_deriv_copyback);
  }
  inline int nparams() const{ return layer->nparams(); }
};

template<typename U, typename std::enable_if<ISLEAF(U), int>::type = 0>
LayerWrapper enwrap(U &&u){
  return LayerWrapper(DDST(u)(std::forward<U>(u)));
}



void testWrapping(){
  double B=0;
  double A=2;
  Matrix winit(1,1,A);
  Vector binit(1,B);

  auto model = dnn_layer(
			 dnn_layer(
				   dnn_layer(
      				             input_layer(),
					     winit,binit),
				   winit, binit),
			 winit,binit);
 
  LayerWrapper composed = enwrap( input_layer() );
  for(int i=0;i<3;i++)
     composed = enwrap( dnn_layer(std::move(composed), winit,binit) );
 
  
  int iters=10;
  for(int i=0;i<iters;i++){
      Matrix x(1,1, i+1);
      Matrix vexpect = model.value(x);

      Matrix vgot = composed.value(x);

      std::cout << i << " got " << vgot(0,0) << " expect " << vexpect(0,0) << std::endl;
  }
}

void testPipeline(){
  int nranks;
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  int batch_size = 1;
  int input_features = 1;
  

  double B=0.15;
  double A=3.14;
  
  Matrix winit(1,1,A);
  Vector binit(1,B);
  typedef decltype( dnn_layer(input_layer(), winit,binit) ) Ltype;


  if(1){ //test model
    if(!rank) std::cout << "Testing model value pipeline" << std::endl;
    //auto b = dnn_layer(input_layer(), winit,binit);    
    //auto p = pipeline_block( b, batch_size, input_features, 1, rank == nranks -1 ? 0 : 1);

    auto p = pipeline_block( dnn_layer(input_layer(), winit,binit), batch_size, input_features, 1, rank == nranks -1 ? 0 : 1);
    
    int value_lag = p.valueLag(); //iterations before first complete cycle of forwards differentiation
    int deriv_lag = p.derivLag(); //iterations before first complete cycle of backwards differentiation
    
    int iters=20;

    //Build the same model on just this rank
    LayerWrapper test_model = enwrap( input_layer() );
    for(int r=0;r<nranks;r++) test_model = enwrap( dnn_layer(std::move(test_model), winit,binit) ); 

    if(!rank) std::cout << "Computing expectations" << std::endl;
    std::vector<Matrix> expect_v(iters);
    std::vector<Vector> expect_d(iters, Vector(test_model.nparams()) );
    
    std::vector<Matrix> input_deriv(iters);
    for(int i=0;i<iters;i++){
      input_deriv[i] = Matrix(1,batch_size, 2.13*(i+1)); 
      Matrix x(1,1, i+1);
      expect_v[i] = test_model.value(x);
      test_model.deriv(expect_d[i],0,input_deriv[i]);
    }
    int nparams = test_model.nparams();

    if(!rank) std::cout << "Starting test loop" << std::endl;
    for(int i=0;i<iters;i++){
      Matrix x(1,1, i+1);
      Matrix v = p.value(x);
      Vector d(nparams,0.);

      int i_vpipe = i-(value_lag-1); //lag=3    2->0  3->1
      int i_dpipe = i-(deriv_lag-1);
      p.deriv(d,i_vpipe >= 0 ? input_deriv[i_vpipe] : Matrix(1,batch_size,-1)); //use the input deriv appropriate to the item index!
      
      if(!rank){

	if(i_vpipe >=0 ){
	  double ev = expect_v[i_vpipe](0,0); 
	  std::cout << i << "\tval expect " << ev << " got "<<  v(0,0) << std::endl;
	}
	if(i_dpipe >=0 ){
	  Vector ed = expect_d[i_dpipe];	
	  std::cout << "\tderiv expect " << ed << " got " << d << std::endl;
	}
      }
    }
  }
  if(1){ //test cost
    if(!rank) std::cout << "Testing loss pipeline" << std::endl;
    auto p = pipeline_block( dnn_layer(input_layer(), winit,binit) , batch_size, input_features, 1, rank == nranks -1 ? 0 : 1);
    PipelineCostFuncWrapper<decltype(p),MSEcostFunc> pc(p);
    int value_lag = p.valueLag();
    int deriv_lag = p.derivLag();
    
    //Build the same model on just this rank
    LayerWrapper test_model = enwrap( input_layer() );
    for(int r=0;r<nranks;r++) test_model = enwrap( dnn_layer(std::move(test_model), winit,binit) ); 
    auto test_cost = mse_cost(test_model);

    int nparams = p.nparams();
    
    int iters=20;

    std::vector<Matrix> x(iters);
    std::vector<Matrix> y(iters);
    
    for(int i=0;i<iters;i++){
      x[i] = Matrix(1,1, i+1);

      double ival = i+1;
      for(int r=0;r<nranks;r++)
	ival = B + A*ival;

      //Add noise
      y[i] = Matrix(1,1, 1.05*ival);
    }

    //Get expectation loss and derivatives
    if(!rank) std::cout << "Computing expectations" << std::endl;
    std::vector<double> expect_l(iters);
    std::vector<Vector> expect_d(iters, Vector(test_model.nparams()) );
    for(int i=0;i<iters;i++){
      expect_l[i] = test_cost.loss(x[i],y[i]);
      expect_d[i] = test_cost.deriv();
    }
    
    if(!rank) std::cout << "Starting test loop" << std::endl;
    for(int i=0;i<iters;i++){
      int i_vpipe = i-(value_lag-1);
      double loss = pc.loss(x[i],y[i]);     
      double loss_expect = i_vpipe < 0 ? -1. : expect_l[i_vpipe];

      int i_dpipe = i-(deriv_lag-1); //item index associated with derivative
      Vector deriv = pc.deriv();
      Vector deriv_expect = i_dpipe < 0 ? Vector(nparams,-1.) : expect_d[i_dpipe];
      
      if(!rank){
	std::cout << i << "\tvalue expect " << loss_expect << " got "<<  loss << std::endl;
	std::cout << "\tderiv expect " << deriv_expect << " got " << deriv << std::endl;
      }
    }
  }

}


int main(int argc, char** argv){
  MPI_Init(&argc, &argv);
  
  //basicTests();
  //testSimpleLinear();
  //testOneHiddenLayer();
  //testWrapping();
  testPipeline();

  MPI_Finalize();
  return 0;
}

