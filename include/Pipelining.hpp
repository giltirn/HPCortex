#pragma once
#include<Tensors.hpp>
#include<Layers.hpp>
#include<Comms.hpp>

//This is a pipeline parallelized cost function wrapper. The user must provide a PipelineBlock containing the layers on the current rank, each terminating
//with an InputLayer.
//call_batch_size must be provided indicating the batch size acted on by each rank in each cycle; it must be a divisor of the total batch size
template<typename PipelineBlockType, typename CostFunc>
class BatchPipelineCostFuncWrapper{
  PipelineBlockType &block;
  
  CostFunc cost;
  int nparam;
  int value_lag;
  int deriv_lag;

  int rank;

  int call_batch_size;
  
  Vector deriv_store;
public:
  BatchPipelineCostFuncWrapper(PipelineBlockType &block, int call_batch_size, const CostFunc &cost = CostFunc()): cost(cost), block(block), nparam(block.nparams()),
														  value_lag(block.valueLag()), deriv_lag(block.derivLag()),
														  call_batch_size(call_batch_size)
  {
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  }


  
  double loss(const Matrix &x, const Matrix &y){
    int dim = y.size(0);
    int global_batch_size = y.size(1);
    int nrank = block.pipelineDepth();

    assert( global_batch_size % call_batch_size == 0);
    int navg = global_batch_size / call_batch_size;

    Matrix x_dummy(x.size(0),call_batch_size,0.0);
    Matrix y_dummy(y.size(0),call_batch_size,0.0);
    
    deriv_store = Vector(nparam, 0.);    

    double out = 0.;

    //compute the total required number of calls
    //3 ranks
    //Value:
    //iter    rank->
    //        0     1      2     ret
    //0                   <0|     -
    //1            <0-    <1|     -
    //2      <0-   <1-    <2|     0
    //3      <1-   <2-    <3|     1
    //etc
    
    //value lag: 3

    //Deriv:
    //iter    rank->
    //        0     1      2     ret
    //0       
    //1       
    //2      |0>   
    //3      |1>   -0>
    //4      |2>   -1>    -0>    0
    //5            -2>    -1>    1
    //6                   -2>    2
    //etc

    //deriv lag: 5

    int dcount = 0; //number of derivatives that have been computed. When this is equal to navg we terminate
    int iter =0;
    while(dcount < navg){
      Matrix x_iter = iter < navg ? x.peekColumns(iter * call_batch_size, (iter+1)* call_batch_size - 1) : x_dummy;
      Matrix ypred = block.value(x_iter);
      
      int i_vpipe = iter-(value_lag-1);
      int i_dpipe = iter-(deriv_lag-1);

      //start recording loss
      Matrix y_iter(y_dummy);
      if(i_vpipe >= 0 && i_vpipe < navg){
	y_iter = y.peekColumns(i_vpipe * call_batch_size, (i_vpipe+1) * call_batch_size - 1);
	double dloss = cost.loss(y_iter, ypred);
	//if(!rank) std::cout << i_vpipe << " dloss=" << dloss << std::endl;
	out += dloss;
      }
	
      //once we start getting values back we need to feed them back to the derivs
      if(i_vpipe >= 0){
	Matrix layer_deriv = cost.layer_deriv(y_iter , ypred);
	Vector cost_deriv(nparam,0.);    //zero initialize
	block.deriv(cost_deriv, layer_deriv);
	
	if(i_dpipe >= 0){
	  ++dcount;
	  deriv_store += cost_deriv;
	}
      }

      ++iter;
    }

    //Make sure all the ranks get the right output and derivative
    assert( MPI_Bcast(deriv_store.data(), deriv_store.data_len(), MPI_DOUBLE, 0, communicators().pipelineCommunicator()) == MPI_SUCCESS );
    assert( MPI_Bcast(&out, 1, MPI_DOUBLE, 0, communicators().pipelineCommunicator()) == MPI_SUCCESS );

    deriv_store *= 1./navg;
    return out / navg;
  }
  
  Vector deriv() const{ return deriv_store; }

  void update(const Vector &new_params){
    block.update(new_params);
  }
  void step(const Vector &derivs, double eps){
    block.step(derivs,eps);
  }
  int nparams(){ return nparam; }

  Vector getParams(){
    Vector out(nparams());
    block.getParams(out);
    return out;
  }

  Matrix predict(const Matrix &x){
    int global_batch_size = x.size(1);
    int nrank = block.pipelineDepth();

    assert( global_batch_size % call_batch_size == 0);
    int navg = global_batch_size / call_batch_size;

    Matrix x_dummy(x.size(0),call_batch_size,0.0);

    int iters = navg + value_lag -1; //exit when i_vpipe = iter-(value_lag-1) = navg, i.e  iter == navg + (value_lag-1)

    Matrix out;
    
    for(int iter=0;iter<iters;iter++){
      Matrix x_iter = iter < navg ? x.peekColumns(iter * call_batch_size, (iter+1)* call_batch_size - 1) : x_dummy;
      Matrix ypred = block.value(x_iter);

      if(iter == 0 && !rank) out = Matrix(ypred.size(0),global_batch_size);
      
      int i_vpipe = iter-(value_lag-1);      
      if(i_vpipe >= 0 && !rank)
	out.pokeColumns(i_vpipe * call_batch_size, (i_vpipe+1) * call_batch_size - 1, ypred);
    }

    int osize[2] = {out.size(0),out.size(1)};
    assert( MPI_Bcast(osize, 2, MPI_INT, 0, communicators().pipelineCommunicator()) == MPI_SUCCESS );
    if(rank != 0) out = Matrix(osize[0],osize[1]);    
    
    //Make sure all the ranks get the right output
    assert( MPI_Bcast(out.data(), out.data_len(), MPI_DOUBLE, 0, communicators().pipelineCommunicator()) == MPI_SUCCESS );

    return out;
  }
  //note this call is inefficient as we need to do call_batch_size work for only one data. Use the matrix version if this matters
  Vector predict(const Vector &x){
    Matrix b(x.size(0),call_batch_size,0.);  
    b.pokeColumn(0,x);
    return predict(b).peekColumn(0);    
  }
  
};



//This is a pipelined loss function wrapper. Using it requires careful coordination as the returned value is not the loss for the provided inputs but a delayed result
//from an earlier iteration. Specifically, the output of loss will be the loss for the call valueLag() calls previously, and that for deriv derivLag() calls previously
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
    rank = communicators().pipelineRank();
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
  
  void update(const Vector &new_params){
    block.update(new_params);
  }
  void step(const Vector &derivs, double eps){
    block.step(derivs,eps);
  }
  int nparams(){ return nparam; }

  Vector getParams(){
    Vector out(nparams());
    block.getParams(out);
    return out;
  } 
};

class PipelineCommunicator{
protected:
  int rank; //current rank
  int next_rank; //rank of next comm layer, -1 indicates this is the last
  int prev_rank; //rank of previous comm layer, -1 indicates this is the first
  int pipeline_depth; //number of ranks in the pipeline
  bool is_first;
  bool is_last;
public:
  PipelineCommunicator(){
    //Prepare rank information
    rank = communicators().pipelineRank();
    
    int nranks = communicators().pipelineNrank();

    next_rank = rank+1;
    prev_rank = rank-1;
    if(next_rank == nranks) next_rank = -1;
    pipeline_depth = nranks;

    is_first = prev_rank == -1;
    is_last = next_rank == -1;
  }

  int pipelineDepth() const{ return pipeline_depth; }

  template<typename T>
  inline static MPI_Request send(const T &mat, int to){
    MPI_Request req;		
    assert( MPI_Isend(mat.data(), mat.data_len(), MPI_DOUBLE, to, 0, communicators().pipelineCommunicator(), &req) == MPI_SUCCESS );
    return req;
  }
  template<typename T>
  inline static MPI_Request recv(T &mat, int from){
    MPI_Request req;		
    assert( MPI_Irecv(mat.data(), mat.data_len(), MPI_DOUBLE, from, 0, communicators().pipelineCommunicator(), &req) == MPI_SUCCESS );
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

};
  

template<typename BlockStore>
class PipelineBlock: public PipelineCommunicator{
  BlockStore block; //this chain should terminate on an InputLayer. This represents the work done on this rank
  
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

  int dcalls;
  
public:
  
  PipelineBlock(BlockStore &&_block, int batch_size, int input_features, int this_features, int next_features): block(std::move(_block)), batch_size(batch_size), input_features(input_features), this_features(this_features), next_features(next_features), dcalls(0){

    //Compute parameter information
    std::vector<int> block_params(pipeline_depth, 0);
    block_params[rank] = block.v.nparams();
    
    assert( MPI_Allreduce(MPI_IN_PLACE, block_params.data(), pipeline_depth, MPI_INT, MPI_SUM, communicators().pipelineCommunicator()) == 0);

    //Compute block param offset
    stage_off = 0;
    for(int i=0;i<rank;i++) stage_off += block_params[i];

    //Compute total params
    nparam = 0;
    for(int i=0;i<pipeline_depth;i++) nparam += block_params[i];
    
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

  //Amount of iterations at which you get the return value for the first item back
  //i.e.  iteration  i -> i-(value_lag - 1)    with i=0,1,2...
  int valueLag() const{ return pipeline_depth; }
  //Amount of iterations at which you get the derivative for the first item back
  //i.e.  iteration  i -> i-(deriv_lag - 1)    with i=0,1,2...
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
    assert( MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE) == MPI_SUCCESS );
    
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
    
    assert( MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE) == MPI_SUCCESS );

    //if(rank == 0) std::cout << "RANK 0 CALL " << dcalls << " RECEIVED COST DERIV: " << cost_deriv << std::endl;
  }

  //Update the parameters. It is assumed that all ranks have the same input 'new_params'
  inline void update(const Vector &new_params){
    block.v.update(stage_off, new_params);
  }

  //Step down the gradient. Assumed that all ranks have the same input 'new_params'
  inline void step(const Vector &derivs, double eps){
    block.v.step(stage_off, derivs, eps);
  }    
  //Get the parameters for the complete model. Each rank will get the same result
  inline void getParams(Vector &into){
    into = Vector(nparam, 0.);
    block.v.getParams(into, stage_off);
    assert( MPI_Allreduce(MPI_IN_PLACE, into.data(),into.data_len(), MPI_DOUBLE, MPI_SUM, communicators().pipelineCommunicator()) == MPI_SUCCESS );
  }  
  
};

template<typename U, typename std::enable_if<ISLEAF(U), int>::type = 0>
auto pipeline_block(U &&u, int batch_size, int input_features, int this_features, int next_features)->PipelineBlock<DDST(u)>{
  return PipelineBlock<DDST(u)>(std::forward<U>(u),batch_size,input_features, this_features, next_features);
}

