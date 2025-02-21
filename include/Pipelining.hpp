#pragma once
#include<mpi.h>
#include<Tensors.hpp>
#include<Layers.hpp>

template<typename PipelineBlockType, typename CostFunc>
class PipelineCostFuncWrapper{
  PipelineBlockType &block;

  RingBuffer<Matrix> yval_buf_v;//buffered yvalues for calls to "loss"
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
