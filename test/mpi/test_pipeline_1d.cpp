#include <HPCortex.hpp>
#include <Testing.hpp>

void testPipeline(){
  typedef confSinglePipeline PipelineConfig;
  typedef confSingle StdConfig;
  typedef float FloatType;
  
  communicators().enableGlobalPipelining(); //put all the ranks into a single pipeline
    
  int nranks = communicators().pipelineNrank();
  int rank = communicators().pipelineRank();
  
  int batch_size = 1;
  int input_features = 1;
  int input_dims[2] = {input_features, batch_size};  

  FloatType B=0.15;
  FloatType A=3.14;
  
  Matrix<FloatType> winit(1,1,A);
  Vector<FloatType> binit(1,B);
  int block_output_dims[2] = {1, batch_size};

  if(1){ //test model
    if(!rank) std::cout << "Testing model value pipeline" << std::endl;

    auto p = pipeline_block< Matrix<FloatType>, Matrix<FloatType> >( dnn_layer(winit,binit,input_layer<PipelineConfig>()), block_output_dims, rank == nranks - 1  ? input_dims : block_output_dims);
    
    int value_lag = p.valueLag(); //iterations before first complete cycle of forwards differentiation
    int deriv_lag = p.derivLag(); //iterations before first complete cycle of backwards differentiation
    
    int iters=20;

    //Build the same model on just this rank
    auto test_model = enwrap( input_layer<StdConfig>() );
    for(int r=0;r<nranks;r++) test_model = enwrap( dnn_layer(winit,binit,std::move(test_model)) ); 

    if(!rank) std::cout << "Computing expectations" << std::endl;
    std::vector<Matrix<FloatType> > expect_v(iters);
    std::vector<Vector<FloatType> > expect_d(iters, Vector<FloatType>(test_model.nparams()) );
    
    std::vector<Matrix<FloatType> > input_deriv(iters);
    for(int i=0;i<iters;i++){
      input_deriv[i] = Matrix<FloatType>(1,batch_size, 2.13*(i+1)); 
      Matrix<FloatType> x(1,1, i+1);
      expect_v[i] = test_model.value(x,DerivYes);

      Matrix<FloatType> idcp(input_deriv[i]);
      test_model.deriv(expect_d[i],0,std::move(idcp));
    }
    int nparams = test_model.nparams();

    if(!rank) std::cout << "Starting test loop" << std::endl;
    for(int i=0;i<iters;i++){
      Matrix<FloatType> x(1,1, i+1);
      Matrix<FloatType> v = p.value(x,DerivYes);
      Vector<FloatType> d(nparams,0.);

      int i_vpipe = i-(value_lag-1); //lag=3    2->0  3->1
      int i_dpipe = i-(deriv_lag-1);
      p.deriv(d,i_vpipe >= 0 ? input_deriv[i_vpipe] : Matrix<FloatType>(1,batch_size,-1)); //use the input deriv appropriate to the item index!
      
      if(!rank){

	if(i_vpipe >=0 ){
	  autoView(ev_i_v, expect_v[i_vpipe], HostRead);
	  autoView(v_v,v,HostRead);
	  
	  FloatType ev = ev_i_v(0,0); 
	  std::cout << i << "\tval expect " << ev << " got "<<  v_v(0,0) << std::endl;
	  assert(near(ev,v_v(0,0),FloatType(1e-4)));
	}
	if(i_dpipe >=0 ){
	  Vector<FloatType> ed = expect_d[i_dpipe];	
	  std::cout << "\tderiv expect " << ed << " got " << d << std::endl;
	  assert(near(d,ed,FloatType(1e-4),true));
	}
      }
    }
  }
  if(1){ //test cost
    if(!rank) std::cout << "Testing loss pipeline" << std::endl;
    auto p = pipeline_block< Matrix<FloatType>, Matrix<FloatType> >( dnn_layer(winit,binit,input_layer<PipelineConfig>()) , block_output_dims, rank == nranks - 1  ? input_dims : block_output_dims);
    PipelineCostFuncWrapper<decltype(p),MSEcostFunc<Matrix<FloatType>> > pc(p);
    int value_lag = p.valueLag();
    int deriv_lag = p.derivLag();
    
    //Build the same model on just this rank
    auto test_model = enwrap( input_layer<StdConfig>() );
    for(int r=0;r<nranks;r++) test_model = enwrap( dnn_layer(winit,binit,std::move(test_model)) ); 
    auto test_cost = mse_cost(test_model);

    int nparams = p.nparams();
    
    int iters=20;

    std::vector<Matrix<FloatType> > x(iters);
    std::vector<Matrix<FloatType>> y(iters);
    
    for(int i=0;i<iters;i++){
      x[i] = Matrix<FloatType>(1,1, i+1);

      FloatType ival = i+1;
      for(int r=0;r<nranks;r++)
	ival = B + A*ival;

      //Add noise
      y[i] = Matrix<FloatType>(1,1, 1.05*ival);
    }

    //Get expectation loss and derivatives
    if(!rank) std::cout << "Computing expectations" << std::endl;
    std::vector<FloatType> expect_l(iters);
    std::vector<Vector<FloatType> > expect_d(iters, Vector<FloatType>(test_model.nparams()) );
    for(int i=0;i<iters;i++){
      expect_l[i] = test_cost.loss(x[i],y[i],DerivYes);
      expect_d[i] = test_cost.deriv();
    }
    
    if(!rank) std::cout << "Starting test loop" << std::endl;
    for(int i=0;i<iters;i++){
      int i_vpipe = i-(value_lag-1);
      FloatType loss = pc.loss(x[i],y[i],DerivYes).first;
      FloatType loss_expect = i_vpipe < 0 ? -1. : expect_l[i_vpipe];

      int i_dpipe = i-(deriv_lag-1); //item index associated with derivative
      Vector<FloatType> deriv = pc.deriv().first;
      Vector<FloatType> deriv_expect = i_dpipe < 0 ? Vector<FloatType>(nparams,-1.) : expect_d[i_dpipe];
      
      if(!rank){
	std::cout << i << "\tvalue expect " << loss_expect << " got "<<  loss << std::endl;
	std::cout << "\tderiv expect " << deriv_expect << " got " << deriv << std::endl;
	assert(near(loss_expect,loss,FloatType(1e-4)));
	assert(near(deriv_expect,deriv,FloatType(1e-4),true));
      }
    }
  }


  if(1){ //test batched cost
    if(!rank) std::cout << "Testing batch loss pipeline" << std::endl;

    int glob_batch_size = 6*nranks;
    int call_batch_size = 2;

    int input_dims_2[2] = {input_features, call_batch_size};  
    int block_output_dims_2[2] = {1, call_batch_size};
    
    auto p = pipeline_block<Matrix<FloatType>, Matrix<FloatType> >( dnn_layer(winit,binit,input_layer<PipelineConfig>()) , block_output_dims_2, rank == nranks -1 ? input_dims_2 : block_output_dims_2);
    BatchPipelineCostFuncWrapper<decltype(p),MSEcostFunc<Matrix<FloatType>> > pc(p, call_batch_size);

    Matrix<FloatType> x(input_features, glob_batch_size);
    Matrix<FloatType> y(1, glob_batch_size);

    for(int i=0;i<glob_batch_size;i++){
      pokeColumn(x,i,Vector<FloatType>(1,i+1));

      FloatType ival = i+1;
      for(int r=0;r<nranks;r++)
	ival = B + A*ival;

      //Add noise
      pokeColumn(y, i, Vector<FloatType>(1, 1.05*ival) );
    }
    
    //Build the same model on just this rank
    auto test_model = enwrap( input_layer<StdConfig>() );
    for(int r=0;r<nranks;r++) test_model = enwrap( dnn_layer(winit,binit,std::move(test_model)) ); 
    auto test_cost = mse_cost(test_model);


    FloatType loss_expect = test_cost.loss(x,y,DerivYes);
    Vector<FloatType> deriv_expect = test_cost.deriv();

    FloatType loss_got = pc.loss(x,y,DerivYes);
    Vector<FloatType> deriv_got = pc.deriv();

    if(!rank){
      std::cout << "Loss - got " << loss_got << " expect " << loss_expect << std::endl;
      std::cout << "Deriv - got " << deriv_got << " expect " << deriv_expect << std::endl;
      assert(near(loss_expect,loss_got,FloatType(1e-4)));
      assert(near(deriv_expect,deriv_got,FloatType(1e-4),true));
    }
  }
  std::cout << "testPipeline passed" << std::endl;
}


struct PostCommActionCallback{
  virtual void performAction() = 0;
  virtual ~PostCommActionCallback(){}
};

struct CommsRequest{
  MPI_Request req;
  std::unique_ptr<PostCommActionCallback> post;
};

template<typename T>
struct PostCommActionCallbackUnlock: public PostCommActionCallback{
  T const* v;
  
  PostCommActionCallbackUnlock(T const* v): v(v){}
  
  void performAction() override{
    v->unlock();
  }
};

void waitAll(std::vector<CommsRequest> &reqs){
  std::vector<MPI_Request> rm(reqs.size());
  for(int i=0;i<reqs.size();i++)
    rm[i] = reqs[i].req;
  assert( MPI_Waitall(rm.size(), rm.data(), MPI_STATUSES_IGNORE) == MPI_SUCCESS );
  for(int i=0;i<reqs.size();i++){
    if(reqs[i].post) reqs[i].post->performAction();
  }
  reqs.clear();
}

struct PipelineLayerIOcontainerInitializer{
  virtual ~PipelineLayerIOcontainerInitializer(){}
};

class PipelineLayerIOcontainerBase{
public:
  virtual CommsRequest send(int to) = 0;
  virtual CommsRequest recv(int from) = 0;
  virtual CommsRequest sendInitializer(int to) = 0;
  virtual CommsRequest recvInitializer(int from) = 0;

  virtual std::unique_ptr<PipelineLayerIOcontainerInitializer> getInitializer() const = 0;
  virtual void initialize(const std::unique_ptr<PipelineLayerIOcontainerInitializer> &init) = 0;
  
  virtual PipelineLayerIOcontainerBase* copy() const = 0;
  virtual PipelineLayerIOcontainerBase * getMicroBatch(int ubatch_idx, int ubatch_size) const = 0;
  virtual void insertMicroBatch(int ubatch_idx, int ubatch_size, PipelineLayerIOcontainerBase *from) = 0;
  virtual void insertFirstMicroBatch(int ubatch_size, PipelineLayerIOcontainerBase *from, int nubatch) = 0;
  virtual int batchDimSize() const = 0;
  virtual ~PipelineLayerIOcontainerBase(){}
};


template<typename T>
class PipelineLayerIOcontainer: public PipelineLayerIOcontainerBase{};

template<typename FloatType, int Dim>
struct PostCommActionCallbackTensorInitialize: public PostCommActionCallback{
  std::unique_ptr<Tensor<FloatType,Dim> > &tens;
  int size[Dim];
  PostCommActionCallbackTensorInitialize(std::unique_ptr<Tensor<FloatType,Dim> > &tens): tens(tens){}
  
  void performAction() override{
    tens.reset(new Tensor<FloatType,Dim>(size));
  }
};

template<int Dim>
struct PipelineLayerIOcontainerTensorInitializer: public PipelineLayerIOcontainerInitializer{
  int dims[Dim];
  PipelineLayerIOcontainerTensorInitializer(int const* in_dim){
    memcpy(dims, in_dim, Dim*sizeof(int));
  }
    
};

template<typename FloatType, int Dim>
class PipelineLayerIOcontainer<Tensor<FloatType,Dim> >: public PipelineLayerIOcontainerBase{
  std::unique_ptr<Tensor<FloatType,Dim> > tens;
public:
  PipelineLayerIOcontainer(){}
  PipelineLayerIOcontainer(const Tensor<FloatType,Dim> &t): tens(new Tensor<FloatType,Dim>(t)){}
  PipelineLayerIOcontainer(Tensor<FloatType,Dim> &&t): tens(new Tensor<FloatType,Dim>(std::move(t))){}

  std::unique_ptr<PipelineLayerIOcontainerInitializer> getInitializer() const override{
    assert(tens);
    return std::unique_ptr<PipelineLayerIOcontainerInitializer>(new PipelineLayerIOcontainerTensorInitializer<Dim>(tens->sizeArray()));
  }
  void initialize(const std::unique_ptr<PipelineLayerIOcontainerInitializer> &init) override{
    PipelineLayerIOcontainerTensorInitializer<Dim> const* ip = dynamic_cast<PipelineLayerIOcontainerTensorInitializer<Dim> const*>(init.get());
    tens.reset(new Tensor<FloatType,Dim>(ip->dims));
  }
  
  CommsRequest send(int to) override{
    assert(tens);
    tens->lock();
    CommsRequest out;
    autoView(tens_v,(*tens),HostRead);   
    assert( MPI_Isend(tens_v.data(), tens_v.data_len(), getMPIdataType<FloatType>(), to, 0, communicators().pipelineCommunicator(), &out.req) == MPI_SUCCESS );
    out.post.reset(new PostCommActionCallbackUnlock(tens.get()));    
    return out;
  }
  CommsRequest recv(int from) override{
    assert(tens);
    tens->lock();
    CommsRequest out;
    autoView(tens_v,(*tens),HostWrite);	
    assert( MPI_Irecv(tens_v.data(), tens_v.data_len(), getMPIdataType<FloatType>(), from, 0, communicators().pipelineCommunicator(), &out.req) == MPI_SUCCESS );
    out.post.reset(new PostCommActionCallbackUnlock(tens.get()));
    return out;
  }
  CommsRequest sendInitializer(int to) override{
    if(!tens) std::cout << "Empty tensor on rank " << communicators().pipelineRank() << " ptr " << tens.get() << std::endl;
    assert(tens);
    CommsRequest out;
    assert( MPI_Isend(tens->sizeArray(), Dim, getMPIdataType<int>(), to, 0, communicators().pipelineCommunicator(), &out.req) == MPI_SUCCESS );
    return out;
  }
  
  CommsRequest recvInitializer(int from) override{
    CommsRequest out;
    PostCommActionCallbackTensorInitialize<FloatType,Dim> *callback = new PostCommActionCallbackTensorInitialize<FloatType,Dim>(tens);
  
    assert( MPI_Irecv(callback->size, Dim, getMPIdataType<int>(), from, 0, communicators().pipelineCommunicator(), &out.req) == MPI_SUCCESS );
    out.post.reset(callback);
    return out;
  }

  PipelineLayerIOcontainerBase* copy() const override{
    if(tens) return new PipelineLayerIOcontainer<Tensor<FloatType,Dim> >(*tens);
    else return new PipelineLayerIOcontainer<Tensor<FloatType,Dim> >();      
  }
  
  Tensor<FloatType,Dim>& get(){
    assert(tens);
    return *tens;
  }
  const Tensor<FloatType,Dim>& get() const{
    assert(tens);
    return *tens;
  }

  Tensor<FloatType,Dim> remove(){
    assert(tens);
    Tensor<FloatType,Dim>* p = tens.release();
    Tensor<FloatType,Dim> out(std::move(*p));
    delete p;
    return out;
  }   

  PipelineLayerIOcontainerBase * getMicroBatch(int ubatch_idx, int ubatch_size) const override{
    assert(tens);
    Tensor<FloatType,Dim> slice = tens->sliceLastDimension(ubatch_idx*ubatch_size, (ubatch_idx+1)*ubatch_size - 1);
    return new PipelineLayerIOcontainer<Tensor<FloatType,Dim> >(std::move(slice));
  }
  void insertMicroBatch(int ubatch_idx, int ubatch_size, PipelineLayerIOcontainerBase *from) override{
    assert(tens);
    Tensor<FloatType,Dim> &slice = *(dynamic_cast<PipelineLayerIOcontainer<Tensor<FloatType,Dim> >* >(from)->tens);
    tens->insertSliceLastDimension(slice, ubatch_idx*ubatch_size, (ubatch_idx+1)*ubatch_size - 1);
  }
  void insertFirstMicroBatch(int ubatch_size, PipelineLayerIOcontainerBase *from, int nubatch) override{
    Tensor<FloatType,Dim> &slice = *(dynamic_cast<PipelineLayerIOcontainer<Tensor<FloatType,Dim> >* >(from)->tens);
    int size[Dim]; memcpy(size,slice.sizeArray(),Dim*sizeof(int));
    size[Dim-1] = nubatch*ubatch_size;
    tens.reset(new Tensor<FloatType,Dim>(size));
    tens->insertSliceLastDimension(slice, 0, ubatch_size - 1);
  }
  int batchDimSize() const override{
    assert(tens);
    return tens->size(Dim-1);
  }
};

struct PipelineLayerIOcontainer_p{
  std::unique_ptr<PipelineLayerIOcontainerBase> p;

  PipelineLayerIOcontainer_p(){}
  PipelineLayerIOcontainer_p(PipelineLayerIOcontainerBase *pin): p(pin){} //takes ownership
  
  template<typename T, typename std::enable_if<!std::is_same<typename std::decay<T>::type,PipelineLayerIOcontainer_p>::value, int>::type = 0>
  PipelineLayerIOcontainer_p(T &&v): p(new PipelineLayerIOcontainer<typename std::decay<T>::type>(std::forward<T>(v))){}

  PipelineLayerIOcontainer_p(const PipelineLayerIOcontainer_p &to_copy): p(to_copy.p->copy()){}
  PipelineLayerIOcontainer_p(PipelineLayerIOcontainer_p &&to_move): p(std::move(to_move.p)){}  

  PipelineLayerIOcontainer_p & operator=(PipelineLayerIOcontainer_p &&to_move){
    p.reset(to_move.p.release());
    return *this;
  }
  template<typename T, typename std::enable_if<!std::is_same<typename std::decay<T>::type,PipelineLayerIOcontainer_p>::value, int>::type = 0>
  PipelineLayerIOcontainer_p & operator=(T &&v){
    p.reset(new PipelineLayerIOcontainer<typename std::decay<T>::type>(std::forward<T>(v)));
    return *this;
  }  
  
  template<typename T>
  void setType(){ p.reset(new PipelineLayerIOcontainer<T>()); }
  
  template<typename T>
  void insert(T &&v){
    p.reset(new PipelineLayerIOcontainer<typename std::decay<T>::type>(std::forward<T>(v)));
  }
  
  template<typename T>
  T & as(){  assert(p); return dynamic_cast<PipelineLayerIOcontainer<T> *>(p.get())->get(); }

  template<typename T>
  const T & as() const{ assert(p); return dynamic_cast<PipelineLayerIOcontainer<T> const*>(p.get())->get(); }

  std::unique_ptr<PipelineLayerIOcontainerInitializer> getInitializer() const{ assert(p); return p->getInitializer(); }

  void initialize(const std::unique_ptr<PipelineLayerIOcontainerInitializer> &init){
    assert(p); p->initialize(init);
  }
  
  template<typename T>
  T remove(){
    assert(p);
    PipelineLayerIOcontainerBase *pp = p.release();
    auto out = dynamic_cast<PipelineLayerIOcontainer<T> *>(pp)->remove();
    delete pp;
    return out;
  }
    
  CommsRequest send(int to){ return p->send(to); }
  CommsRequest recv(int from){ return p->recv(from); }
  CommsRequest sendInitializer(int to){ return p->sendInitializer(to); }
  CommsRequest recvInitializer(int from){ return p->recvInitializer(from); }

  PipelineLayerIOcontainer_p getMicroBatch(int ubatch_idx, int ubatch_size) const{
    assert(p);
    return PipelineLayerIOcontainer_p(p->getMicroBatch(ubatch_idx,ubatch_size));
  }
  void insertMicroBatch(int ubatch_idx, int ubatch_size, PipelineLayerIOcontainer_p &from){
    assert(p);
    p->insertMicroBatch(ubatch_idx,ubatch_size,from.p.get());
  }
  void insertFirstMicroBatch(int ubatch_size, PipelineLayerIOcontainer_p &from, int nubatch){
    assert(p);
    p->insertFirstMicroBatch(ubatch_size,from.p.get(),nubatch);
  }
  int batchDimSize() const{
    assert(p);
    return p->batchDimSize();
  }
  
};

void sendRecv(std::vector<CommsRequest> &reqs,
	      PipelineLayerIOcontainer_p &to, PipelineLayerIOcontainer_p &from,
	      int rank_to, int rank_from){
    int me = communicators().pipelineRank();
    if(me == rank_from) reqs.push_back(from.send(rank_to));
    else if(me == rank_to) reqs.push_back(to.recv(rank_from));
  }
template<typename Cond>
void passRightConditional(std::vector<CommsRequest> &reqs,
	       PipelineLayerIOcontainer_p &to, PipelineLayerIOcontainer_p &from, const Cond &send_rank_cond){
  int me = communicators().pipelineRank();
  if(me != communicators().pipelineNrank()-1 && send_rank_cond(me))
    reqs.push_back(from.send(me+1));
  
  if(me != 0 && send_rank_cond(me-1))
    reqs.push_back(to.recv(me-1));
}
void passRight(std::vector<CommsRequest> &reqs,
	       PipelineLayerIOcontainer_p &to, PipelineLayerIOcontainer_p &from){
  int me = communicators().pipelineRank();
  if(me != communicators().pipelineNrank()-1)
    reqs.push_back(from.send(me+1));
  
  if(me != 0)
    reqs.push_back(to.recv(me-1));
}
template<typename Cond>
void passLeftConditional(std::vector<CommsRequest> &reqs,
	       PipelineLayerIOcontainer_p &to, PipelineLayerIOcontainer_p &from, const Cond &send_rank_cond){
  int me = communicators().pipelineRank();
  if(me != 0 && send_rank_cond(me))
    reqs.push_back(from.send(me-1));
  
  if(me != communicators().pipelineNrank()-1 && send_rank_cond(me+1))
    reqs.push_back(to.recv(me+1));
}
void passLeft(std::vector<CommsRequest> &reqs,
	       PipelineLayerIOcontainer_p &to, PipelineLayerIOcontainer_p &from){
  int me = communicators().pipelineRank();
  if(me != 0)
    reqs.push_back(from.send(me-1));
  
  if(me != communicators().pipelineNrank()-1)
    reqs.push_back(to.recv(me+1));
}

void sendRecvInitializer(std::vector<CommsRequest> &reqs,
			 PipelineLayerIOcontainer_p &to, PipelineLayerIOcontainer_p &from,
			 int rank_to, int rank_from){
  int me = communicators().pipelineRank();
  if(me == rank_from) reqs.push_back(from.sendInitializer(rank_to));
  else if(me == rank_to) reqs.push_back(to.recvInitializer(rank_from));
}



template<typename FloatType>
class PipelineBlockContainerBase{
public:
  virtual PipelineLayerIOcontainer_p blockValue(const PipelineLayerIOcontainer_p &block_input, EnableDeriv enable_deriv) = 0;
  virtual void blockDeriv(Vector<FloatType> &cost_deriv, const PipelineLayerIOcontainer_p &_above_deriv, PipelineLayerIOcontainer_p &layer_input_deriv) = 0;
  virtual void blockUpdate(int off, const Vector<FloatType> &new_params) = 0;
  virtual void blockStep(int off, const Vector<FloatType> &derivs, FloatType eps) = 0;
  virtual void blockGetParams(Vector<FloatType> &rank_params) = 0;
  virtual size_t blockFLOPS(int value_or_deriv) const = 0;
  virtual void resizeInputBuffer(int to) = 0;
  virtual void setInputType(PipelineLayerIOcontainer_p &con) const = 0;
  virtual void setOutputType(PipelineLayerIOcontainer_p &con) const = 0;
  virtual int nparams() const = 0;
  virtual ~PipelineBlockContainerBase(){}
};

template<typename BlockStore>
class PipelineBlockContainer: public PipelineBlockContainerBase<typename BlockStore::type::FloatType>{
  BlockStore block;
  typedef typename BlockStore::type BlockType;
  typedef typename BlockType::InputType BlockInputType;
  typedef typename BlockType::FloatType FloatType;
  typedef LAYEROUTPUTTYPE(BlockType) BlockOutputType;
public:
  PipelineBlockContainer(BlockStore &&block): block(std::move(block)){}
  
  PipelineLayerIOcontainer_p blockValue(const PipelineLayerIOcontainer_p &block_input, EnableDeriv enable_deriv) override{
    const BlockInputType &block_input_tens = block_input.as<BlockInputType>();
    return PipelineLayerIOcontainer_p(block.v.value(block_input_tens,enable_deriv));
  }
  void blockDeriv(Vector<FloatType> &cost_deriv, const PipelineLayerIOcontainer_p &_above_deriv, PipelineLayerIOcontainer_p &_layer_input_deriv) override{
    BlockOutputType above_deriv(_above_deriv.as<BlockOutputType>());
    BlockInputType layer_input_deriv_tmp;
    block.v.deriv(cost_deriv, 0, std::move(above_deriv), &layer_input_deriv_tmp);
    _layer_input_deriv = std::move(layer_input_deriv_tmp);
  }
  void blockUpdate(int off, const Vector<FloatType> &new_params) override{
    block.v.update(off,new_params);
  }
  void blockStep(int off, const Vector<FloatType> &derivs, FloatType eps) override{
    block.v.step(off, derivs, eps);
  }
  void blockGetParams(Vector<FloatType> &rank_params) override{
    block.v.getParams(rank_params, 0);
  }
  void resizeInputBuffer(int to) override{
    block.v.resizeInputBuffer(to);
  }
  size_t blockFLOPS(int value_or_deriv) const override{
    return block.v.FLOPS(value_or_deriv);
  }    
  void setInputType(PipelineLayerIOcontainer_p &con) const override{
    con.setType<BlockInputType>();
  }
  void setOutputType(PipelineLayerIOcontainer_p &con) const override{
    con.setType<BlockOutputType>();
  }
  int nparams() const override{
    return block.v.nparams();
  }
};

//Note: the pipeline starts with rank 0!
template<typename LayerOutputType, typename BelowStore>
class PipelineBlockLayer{
public:
  typedef typename BelowStore::type BelowType;
  typedef typename BelowType::ModelConfig Config;
  EXTRACT_CONFIG_TYPES;
  typedef LAYEROUTPUTTYPE(BelowType) LayerInputType;
  typedef typename BelowType::InputType InputType;
  
private: 
  BelowStore below;
  
  int ubatch_size;
  std::unique_ptr<PipelineBlockContainerBase<FloatType> > rank_block;
  bool initialized;

  int rank;
  int pipeline_depth;
  bool is_first;
  bool is_last;

  typedef std::unique_ptr<PipelineLayerIOcontainerInitializer> initStore;
  initStore rank_block_in_init;
  initStore pipeline_out_init;

  mutable initStore rank_above_deriv_init;
  mutable initStore pipeline_input_deriv_init;
  
  int nparam;
  std::vector<int> rank_block_nparam;
  int rank_param_offset;

  size_t value_flops;
  mutable size_t deriv_flops;
  
  void initialize(){
    if(initialized) return;
    assert(rank_block);
    rank_block_nparam[rank] = rank_block->nparams();
    commsReduce(rank_block_nparam.data(),rank_block_nparam.size(), communicators().pipelineCommunicator());
    
    for(int r=0;r<pipeline_depth;r++){
      if(r==rank) rank_param_offset = nparam;
      
      if(!rank) std::cout << "Rank params " << r << " : " << rank_block_nparam[r] << std::endl;
      nparam += rank_block_nparam[r];
    }
    if(!rank) std::cout << "Total params : " << nparam << std::endl;

    //note: the ordering of parameters is *top-down*, so the rank's parameter offset counts down
    rank_param_offset = nparam;
    for(int r=0;r<=rank;r++)
      rank_param_offset -= rank_block_nparam[r];
    
    initialized = true;
  }
  //gather a parameter vector to rank 0
  void gatherParameterVector(int to_off, Vector<FloatType> &vec_to, Vector<FloatType> &vec_from) const{
    if(rank != 0){
      autoView(vec_from_v, vec_from, HostRead);
      MPI_Request req;
      assert( MPI_Isend(vec_from_v.data(), vec_from_v.data_len(), getMPIdataType<FloatType>(),
			0, 0, communicators().pipelineCommunicator(), &req) == MPI_SUCCESS );
      assert(MPI_Wait(&req, MPI_STATUS_IGNORE) == MPI_SUCCESS);
    }else{
      //note: the ordering of parameters is *top-down*, so for the pipeline we need to put the last rank first in the output      
      autoView(vec_to_v, vec_to, HostReadWrite);
      std::vector<MPI_Request> mr(pipeline_depth-1);
      FloatType* cdp = vec_to_v.data() + to_off;
      for(int r=1;r<pipeline_depth;r++){
	int from_rank = pipeline_depth-r;
	assert( MPI_Irecv(cdp , rank_block_nparam[from_rank],
			  getMPIdataType<FloatType>(), from_rank, 0, communicators().pipelineCommunicator(), &mr[r-1]) == MPI_SUCCESS );
	cdp += rank_block_nparam[from_rank];
      }
      assert(MPI_Waitall(pipeline_depth-1, mr.data(), MPI_STATUSES_IGNORE) == MPI_SUCCESS);

      autoView(vec_from_v, vec_from, HostRead);
      memcpy(cdp, vec_from_v.data(), vec_from_v.data_len()*sizeof(FloatType));
    }
  }
  
public:
  typedef LeafTag tag;
  
  PipelineBlockLayer(BelowStore &&below, int ubatch_size): below(std::move(below)), ubatch_size(ubatch_size), initialized(false), nparam(0), rank(communicators().pipelineRank()),
							   pipeline_depth(communicators().pipelineNrank()), is_first(rank == 0), is_last(rank == pipeline_depth -1), 
							   rank_block_nparam(communicators().pipelineNrank(),0){}

  template<typename Block>
  void setRankBlock(Block &&block){
    if(initialized) throw std::runtime_error("Cannot change model once initialized");
    rank_block.reset(new PipelineBlockContainer<DDST(block)>(std::forward<Block>(block)));
  }

  int nparams(){
    initialize();
    return nparam + below.v.nparams();
  }
  
  LayerOutputType value(const InputType &x, EnableDeriv enable_deriv = DerivNo){
    initialize();
   
    PipelineLayerIOcontainer_p below_in(below.v.value(x,enable_deriv)); //everyone call below.value but it need only be valid content on rank 0
    int input_batch_dim_size = below_in.batchDimSize();
    
    assert(input_batch_dim_size % ubatch_size == 0);
    
    int nubatch = input_batch_dim_size/ubatch_size;    
    rank_block->resizeInputBuffer(nubatch);
    
    assert(nubatch >= pipeline_depth);

    std::vector<CommsRequest> reqs;

    PipelineLayerIOcontainer_p rank_block_in, rank_block_out, pipeline_out;
    if(!is_first) rank_block->setInputType(rank_block_in);
    pipeline_out.setType<LayerOutputType>();
    
    int ubatch_idx_in=0;
    //"prime" the pipeline
    for(int prime=0; prime < pipeline_depth; prime++){
      if(is_first)
	rank_block_in = below_in.getMicroBatch(ubatch_idx_in, ubatch_size);
      ++ubatch_idx_in;
      
      //Which rank has data depends on the iteration
      //eg for 4 ranks
      //0  :   0 <--from below
      //1  :   0,1
      //2  :   0,1,2
      //3  :   0,1,2,3
      if(rank <= prime) rank_block_out = rank_block->blockValue(rank_block_in,enable_deriv);

      //Ranks that were just called for the first time need to send their output initializers to the right
      if(prime != pipeline_depth-1){
	if(!rank_block_in_init){	
	  sendRecvInitializer(reqs, rank_block_in, rank_block_out, prime+1, prime);
	  waitAll(reqs);
	}else if(rank == prime+1) rank_block_in.initialize(rank_block_in_init);	  
      }

      //All active ranks need to send their output to the right
      passRightConditional(reqs, rank_block_in, rank_block_out, [=](int send_rank){ return send_rank <= prime; });
      waitAll(reqs);
    }
    //after priming, rank_block_out on the last rank should be the first pipeline output
    if(is_last) pipeline_out.insertFirstMicroBatch(ubatch_size, rank_block_out, nubatch);    

    //record initializer for next call
    if(!rank_block_in_init) rank_block_in_init = rank_block_in.getInitializer();
    
    //steady state
    int ubatch_idx_out = 1;
    for(int iter = pipeline_depth; iter < nubatch; iter++){
      if(is_first)
	rank_block_in = below_in.getMicroBatch(ubatch_idx_in, ubatch_size);
      ++ubatch_idx_in;
      
      rank_block_out = rank_block->blockValue(rank_block_in, enable_deriv);
      passRight(reqs, rank_block_in, rank_block_out);
      waitAll(reqs);
      if(is_last) pipeline_out.insertMicroBatch(ubatch_idx_out, ubatch_size, rank_block_out);
      ++ubatch_idx_out;
    }

    //at this point we should have no further input batches
    assert(ubatch_idx_in == nubatch);
    
    //"drain" the pipeline
    for(int drain=0;drain < pipeline_depth-1; drain++){
      if(rank > drain) rank_block_out = rank_block->blockValue(rank_block_in, enable_deriv);
      passRightConditional(reqs, rank_block_in, rank_block_out, [=](int send_rank){ return send_rank > drain; }  );
      waitAll(reqs);
      if(is_last) pipeline_out.insertMicroBatch(ubatch_idx_out, ubatch_size, rank_block_out);
      ++ubatch_idx_out;
    }
    //should now be done
    assert(ubatch_idx_out == nubatch);

    value_flops = rank_block->blockFLOPS(0) * nubatch;
    
    //Communicate result back to pipeline leader and return
    if(!pipeline_out_init){
      for(int r=0;r<pipeline_depth-1;r++)
	sendRecvInitializer(reqs, pipeline_out, pipeline_out, r, pipeline_depth-1); //all ranks have output of correct *size*
      waitAll(reqs);
      pipeline_out_init = pipeline_out.getInitializer();
    }else if(!is_last) pipeline_out.initialize(pipeline_out_init);
    
    sendRecv(reqs, pipeline_out, pipeline_out, 0, pipeline_depth-1); //only rank 0 has actual output
    waitAll(reqs);
    return pipeline_out.remove<LayerOutputType>();
  }
  int deriv(Vector<FloatType> &cost_deriv, int off, LayerOutputType &&_above_deriv, InputType* input_above_deriv_return = nullptr) const{    
    PipelineLayerIOcontainer_p pipeline_above_deriv(std::move(_above_deriv));  //all ranks store the rvalue but only the pipeline leader is required to have valid input
    Vector<FloatType> rank_block_cost_derivs(rank_block->nparams(), 0.); //accumulate
   
    int batch_dim_size = pipeline_above_deriv.batchDimSize();
    assert(batch_dim_size % ubatch_size == 0);
    
    int nubatch = batch_dim_size/ubatch_size;
    
    //communicate above deriv to top of pipeline
    std::vector<CommsRequest> reqs;      
    sendRecv(reqs, pipeline_above_deriv, pipeline_above_deriv, pipeline_depth-1, 0); //already initialized to correct size by move
    waitAll(reqs);

    PipelineLayerIOcontainer_p rank_above_deriv, rank_input_deriv, pipeline_input_deriv;
    pipeline_input_deriv.setType<LayerInputType>();
    rank_block->setOutputType(rank_above_deriv);
    
    //prime the backwards pass
    int ubatch_idx_in = 0;
    for(int prime=0; prime < pipeline_depth; prime++){
      if(is_last) rank_above_deriv = pipeline_above_deriv.getMicroBatch(ubatch_idx_in, ubatch_size);
      ++ubatch_idx_in;
      if(rank >= pipeline_depth - prime -1){
	Vector<FloatType> cd(rank_block->nparams(),0.);
	rank_block->blockDeriv(cd, rank_above_deriv, rank_input_deriv);
	rank_block_cost_derivs += cd;
      }
      //Ranks that were just called for the first time need to send their output initializers to the left
      if(prime != pipeline_depth-1){
	if(!rank_above_deriv_init){
	  sendRecvInitializer(reqs, rank_above_deriv, rank_input_deriv, pipeline_depth-prime-2, pipeline_depth-prime-1);
	  waitAll(reqs);
	}else if(rank == pipeline_depth-prime-2)
	  rank_above_deriv.initialize(rank_above_deriv_init);	  
      }
      //All active ranks need to send their output to the left
      passLeftConditional(reqs, rank_above_deriv, rank_input_deriv, [=](int send_rank){ return send_rank >= pipeline_depth - prime -1; });
      waitAll(reqs);
    }

    //At this point the first batch of input derivs should have reached rank 0
    if(is_first)
      pipeline_input_deriv.insertFirstMicroBatch(ubatch_size, rank_input_deriv, nubatch);      

    if(!rank_above_deriv_init) rank_above_deriv_init = rank_above_deriv.getInitializer();
    
    //steady state
    int ubatch_idx_out = 1;
    for(int iter = pipeline_depth; iter < nubatch; iter++){
      if(is_last) rank_above_deriv = pipeline_above_deriv.getMicroBatch(ubatch_idx_in, ubatch_size);
      ++ubatch_idx_in;
      Vector<FloatType> cd(rank_block->nparams(),0.);
      rank_block->blockDeriv(cd, rank_above_deriv, rank_input_deriv);
      rank_block_cost_derivs += cd;
      
      passLeft(reqs, rank_above_deriv, rank_input_deriv);
      waitAll(reqs);
      if(is_first) pipeline_input_deriv.insertMicroBatch(ubatch_idx_out, ubatch_size, rank_input_deriv);
      ++ubatch_idx_out;
    }

    //at this point we should have no further input batches
    assert(ubatch_idx_in == nubatch);

    //"drain" the pipeline
    for(int drain=0;drain < pipeline_depth-1; drain++){
      if(rank < pipeline_depth-1-drain){
	Vector<FloatType> cd(rank_block->nparams(),0.);
	rank_block->blockDeriv(cd, rank_above_deriv, rank_input_deriv);
	rank_block_cost_derivs += cd;
      }
      passLeftConditional(reqs, rank_above_deriv, rank_input_deriv, [=](int send_rank){ return send_rank < pipeline_depth-1-drain; });
      waitAll(reqs);
      
      if(is_first) pipeline_input_deriv.insertMicroBatch(ubatch_idx_out, ubatch_size, rank_input_deriv);
      ++ubatch_idx_out;
    }
    //should now be done
    assert(ubatch_idx_out == nubatch);

    deriv_flops = rank_block->blockFLOPS(1) * nubatch;
    
    //gather the cost derivs to rank 0
    gatherParameterVector(off, cost_deriv, rank_block_cost_derivs);

    //ensure the input deriv has the right size for all ranks
    if(!pipeline_input_deriv_init){
      for(int r=1;r<pipeline_depth;r++)
	sendRecvInitializer(reqs, pipeline_input_deriv, pipeline_input_deriv, r, 0);
      waitAll(reqs);
      pipeline_input_deriv_init = pipeline_input_deriv.getInitializer();
    }else if(rank !=0) pipeline_input_deriv.initialize(pipeline_input_deriv_init);   

    return below.v.deriv(cost_deriv, off + nparam, pipeline_input_deriv.remove<LayerInputType>(), input_above_deriv_return);
  }

  inline void resizeInputBuffer(size_t to){
    if(to != 1) throw std::runtime_error("Using a pipeline layer inside other pipelines is not currently supported");
    below.v.resizeInputBuffer(to);
  }
  int update(int off, const Vector<FloatType> &new_params){
    Vector<FloatType> np(new_params);
    commsBroadcast(np, 0, communicators().pipelineCommunicator());
    commsBroadcast(&off, 1, 0, communicators().pipelineCommunicator());
    rank_block->blockUpdate(off + rank_param_offset, np);
    return below.v.update(off + nparam, new_params);
  }
  int step(int off, const Vector<FloatType> &derivs, FloatType eps){
    Vector<FloatType> np(derivs);
    commsBroadcast(np, 0, communicators().pipelineCommunicator());
    commsBroadcast(&off, 1, 0, communicators().pipelineCommunicator());
    rank_block->blockStep(off + rank_param_offset, np, eps);
    return below.v.step(off + nparam, derivs, eps);
  }
  int getParams(Vector<FloatType> &into, int off) const{
    Vector<FloatType> rank_params(rank_block->nparams());
    rank_block->blockGetParams(rank_params);
    commsBroadcast(&off, 1, 0, communicators().pipelineCommunicator());
    gatherParameterVector(off, into, rank_params);
    return below.v.getParams(into, off + nparam);
  }
  size_t FLOPS(int value_or_deriv) const{
    uint64_t fl = value_or_deriv == 0 ? value_flops : deriv_flops;
    commsReduce(&fl,1, communicators().pipelineCommunicator());
    fl += below.v.FLOPS(value_or_deriv);
    return fl;
  }
    
};

/**
 * @brief Create a pipeline block layer object
 * @param ubatch_size The microbatch size. Must be a divisor of the global batch size
 * @param below The layer below
 */
template<typename LayerOutputType, typename U, typename std::enable_if<ISLEAF(U), int>::type = 0>
auto pipeline_block_layer(int ubatch_size, U &&below){
  return PipelineBlockLayer<LayerOutputType, DDST(below)>(std::forward<U>(below), ubatch_size);
}



  
void testPipelineLayer(){
  std::mt19937 rng(1234);

  typedef ModelConfiguration<float, FillEmptyRingBuffer> pconfSingle;
  typedef ModelConfiguration<double, FillEmptyRingBuffer> pconfDouble;
  
  communicators().enableGlobalPipelining(); //put all the ranks into a single pipeline
  assert(communicators().pipelineNrank() > 1);
  int rank = communicators().pipelineRank();
  int nrank = communicators().pipelineNrank();
  
  //Check we can wrap a layer and correctly call value
  {
    int dim[2] = {2,3};
    Tensor<float,2> tens(dim);
    uniformRandom(tens,rng);    
    
    PipelineLayerIOcontainer_p tc(tens);

    auto layer = dnn_layer(4,2,			 
			   input_layer<confSingle>()
			   );
    PipelineBlockContainer<LeafRef<decltype(layer)> > con(layer);
  
    PipelineLayerIOcontainer_p got = con.blockValue(tc,DerivNo);
    auto expect = layer.value(tens);
    assert(equal(got.as<Tensor<float,2> >(), expect,true));
  }
  //Check comms of io container wrapper
  {
    std::cout << "Checking comms of io container wrapper" << std::endl;
    PipelineLayerIOcontainer_p tc;
    if(rank == 0){
      int dim[2] = {2,3};
      Tensor<float,2> tens(dim);
      doHost(tens,{
	  for(int i=0;i<2;i++)
	    for(int j=0;j<3;j++)
	      tens_v(i,j) = j+3*i;
	});
      tc.insert(std::move(tens));
    }else if(rank == 1){
      tc.setType< Tensor<float,2> >();
    }
    std::vector<CommsRequest> reqs;
    sendRecvInitializer(reqs, tc, tc, 1, 0);
    waitAll(reqs);
    if(rank == 1){
      int const* sizes = tc.as< Tensor<float,2> >().sizeArray();
      assert(sizes[0] == 2 && sizes[1] == 3);
    }
    sendRecv(reqs, tc, tc, 1, 0);
    waitAll(reqs);
    if(rank == 1){
      int dim[2] = {2,3};
      Tensor<float,2> expect(dim);
      doHost(expect,{
	  for(int i=0;i<2;i++)
	    for(int j=0;j<3;j++)
	      expect_v(i,j) = j+3*i;
	});
      assert(equal(expect, tc.as< Tensor<float,2> >(), true));      
    
      //check copy, move
      {
	PipelineLayerIOcontainer_p tc_cp(tc);
	assert(equal(tc.as<Tensor<float,2>>(), tc_cp.as<Tensor<float,2>>(), true));

	PipelineLayerIOcontainer_p tc_mv(std::move(tc_cp));
	assert(equal(tc.as<Tensor<float,2>>(), tc_mv.as<Tensor<float,2>>(), true));

	PipelineLayerIOcontainer_p tc_mv2;
	tc_mv2 = std::move(tc_mv);
	assert(equal(tc.as<Tensor<float,2>>(), tc_mv2.as<Tensor<float,2>>(), true));
      }	
      
      //check remove
      Tensor<float,2> rm = tc.remove< Tensor<float,2> >();
      assert(equal(expect, rm, true));
    } //rank==1
  }

    
  //Check layer functionality
  {
    std::cout << "Checking layer functionality" << std::endl;
    int ubatch_size = 2;
    int batch_size = nrank * ubatch_size * 3;

    int fan_in = 3;
    int fan_out = 3;

    //have a non-trivial model below
    Matrix<double> bweight(fan_out,fan_in);
    Vector<double> bbias(fan_out);
    uniformRandom(bweight,rng);
    uniformRandom(bbias,rng);
    
    auto pbelow = dnn_layer(bweight, bbias, input_layer<pconfDouble>());
    // PipelineBlockLayer< Matrix<double>, LeafRef<decltype(pbelow)> > player(pbelow, ubatch_size);
    auto player = pipeline_block_layer<Matrix<double> >(ubatch_size, pbelow);
    
    //For covenience use a uniform fan_in, fan_out
    std::vector<Matrix<double> > weights(nrank, Matrix<double>(fan_out,fan_in));
    std::vector<Vector<double> > biases(nrank, Vector<double>(fan_out));
    for(int r=0;r<nrank;r++){ 
      uniformRandom(weights[r],rng);
      uniformRandom(biases[r],rng);
    }
      
    player.setRankBlock(dnn_layer(weights[rank],biases[rank], input_layer<pconfDouble>()));

    //have a non-trivial model above too
    Matrix<double> aweight(fan_out,fan_in);
    Vector<double> abias(fan_out);
    uniformRandom(aweight,rng);
    uniformRandom(abias,rng);
    auto got_model = dnn_layer(aweight,abias, player);
    
    //generate the equivalent model on each rank separately
    auto expect_model = enwrap(dnn_layer(bweight,bbias,input_layer<confDouble>()));
    for(int r=0;r<nrank;r++)
      expect_model = enwrap(dnn_layer(weights[r],biases[r], std::move(expect_model)));
    expect_model = enwrap(dnn_layer(aweight,abias,std::move(expect_model)));
    
    assert(got_model.nparams() == expect_model.nparams());
    
    for(int i=0;i<2;i++){ //run twice to ensure consistency as we store initializers on the first call
      Matrix<double> input(fan_in, batch_size);
      uniformRandom(input, rng);
      
      //value
      Matrix<double> expect = expect_model.value(input,DerivYes);
      Matrix<double> got = got_model.value(input,DerivYes);
      if(rank == 0) assert(abs_near(expect,got,1e-6,true));

      //deriv
      Matrix<double> above_deriv(fan_out, batch_size);
      uniformRandom(above_deriv,rng);
      
      Vector<double> got_der(got_model.nparams(),0.);
      Matrix<double> got_in_der(fan_in, batch_size);

      Vector<double> expect_der(expect_model.nparams(),0.);
      Matrix<double> expect_in_der(fan_in, batch_size);

      int dout_got = got_model.deriv(got_der, 0, Matrix<double>(above_deriv), &got_in_der);
      int dout_expect = expect_model.deriv(expect_der, 0, Matrix<double>(above_deriv), &expect_in_der);
      std::cout << "Offset got " << dout_got << " expect " << dout_expect << std::endl;
      assert(dout_got == dout_expect);

      if(rank == 0){
	std::cout << "Got der:\n" << got_der << "\nExpect der:\n" << expect_der << std::endl;
	std::cout << "Got input der:\n" << got_in_der << "\nExpect in der:\n" << expect_in_der << std::endl;

	assert(near(got_der,expect_der,1e-6,true));
	assert(near(got_in_der,expect_in_der,1e-6,true));
      }
    }

    //check update
    Vector<double> new_params(got_model.nparams());
    uniformRandom(new_params,rng);
    Vector<double> dummy_params(got_model.nparams(), 0.);
    
    int poff = got_model.update(0, rank == 0 ? new_params : dummy_params); //ensure params are passed from rank 0
    int eoff = expect_model.update(0, new_params);
    assert(poff == eoff);

    Matrix<double> input(fan_in, batch_size);
    uniformRandom(input, rng);
     
    Matrix<double> expect = expect_model.value(input);
    Matrix<double> got = got_model.value(input);
    if(rank == 0) assert(abs_near(expect,got,1e-6,true));

    //check step
    poff = got_model.step(0, rank == 0 ? new_params : dummy_params, 0.567); //ensure params are passed from rank 0
    eoff = expect_model.step(0, new_params, 0.567);
    assert(poff == eoff);
    
    expect = expect_model.value(input);
    got = got_model.value(input);
    if(rank == 0) assert(abs_near(expect,got,1e-6,true));

    //check getparams
    got_model.update(0, rank == 0 ? new_params : dummy_params); //ensure params are passed from rank 0
   
    Vector<double> pgot_params(got_model.nparams());
    poff = got_model.getParams(pgot_params, 0);
    assert(poff == eoff);
    if(rank == 0) assert(equal(pgot_params, new_params, true));
    
    //check FLOPS
    size_t ef = expect_model.FLOPS(0);
    size_t gf = got_model.FLOPS(0);
    if(rank == 0) assert(ef == gf);
    
    ef = expect_model.FLOPS(1);
    gf = got_model.FLOPS(1);
    if(rank == 0) assert(ef == gf);

    
  }

  //Demonstrate you can chain pipeline layers in the normal way
  {
    std::cout << "Checking pipeline layer chaining" << std::endl;
    int ubatch_size = 2;
    int batch_size = nrank * ubatch_size * 3;

    int fan_in = 3;
    int fan_out = 3;

    //have a non-trivial model below
    Matrix<double> bweight(fan_out,fan_in);
    Vector<double> bbias(fan_out);
    uniformRandom(bweight,rng);
    uniformRandom(bbias,rng);
    
    auto pbelow = dnn_layer(bweight, bbias, input_layer<pconfDouble>());
    auto player1 = pipeline_block_layer<Matrix<double> >(ubatch_size, pbelow);
    auto player2 = pipeline_block_layer<Matrix<double> >(ubatch_size, player1);
    //PipelineBlockLayer< Matrix<double>, LeafRef<decltype(pbelow)> > player1(pbelow, ubatch_size);
    //PipelineBlockLayer< Matrix<double>, LeafRef<decltype(player1)> > player2(player1, ubatch_size);

    //For covenience use a uniform fan_in, fan_out
    std::vector<Matrix<double> > weights1(nrank, Matrix<double>(fan_out,fan_in)), weights2(nrank, Matrix<double>(fan_out,fan_in));
    std::vector<Vector<double> > biases1(nrank, Vector<double>(fan_out)), biases2(nrank, Vector<double>(fan_out));
    for(int r=0;r<nrank;r++){ 
      uniformRandom(weights1[r],rng);
      uniformRandom(biases1[r],rng);

      uniformRandom(weights2[r],rng);
      uniformRandom(biases2[r],rng);
    }
      
    player1.setRankBlock(dnn_layer(weights1[rank],biases1[rank], input_layer<pconfDouble>()));
    player2.setRankBlock(dnn_layer(weights2[rank],biases2[rank], input_layer<pconfDouble>()));
    
    //have a non-trivial model above too
    Matrix<double> aweight(fan_out,fan_in);
    Vector<double> abias(fan_out);
    uniformRandom(aweight,rng);
    uniformRandom(abias,rng);
    auto got_model = dnn_layer(aweight,abias, player2);
    
    //generate the equivalent model on each rank separately
    auto expect_model = enwrap(dnn_layer(bweight,bbias,input_layer<confDouble>()));
    for(int r=0;r<nrank;r++)
      expect_model = enwrap(dnn_layer(weights1[r],biases1[r], std::move(expect_model)));
    for(int r=0;r<nrank;r++)
      expect_model = enwrap(dnn_layer(weights2[r],biases2[r], std::move(expect_model)));
    expect_model = enwrap(dnn_layer(aweight,abias,std::move(expect_model)));
    
    assert(got_model.nparams() == expect_model.nparams());
    
    for(int i=0;i<2;i++){ //run twice to ensure consistency as we store initializers on the first call
      Matrix<double> input(fan_in, batch_size);
      uniformRandom(input, rng);
      
      //value
      Matrix<double> expect = expect_model.value(input,DerivYes);
      Matrix<double> got = got_model.value(input,DerivYes);
      if(rank == 0) assert(abs_near(expect,got,1e-6,true));

      //deriv
      Matrix<double> above_deriv(fan_out, batch_size);
      uniformRandom(above_deriv,rng);
      
      Vector<double> got_der(got_model.nparams(),0.);
      Matrix<double> got_in_der(fan_in, batch_size);

      Vector<double> expect_der(expect_model.nparams(),0.);
      Matrix<double> expect_in_der(fan_in, batch_size);

      int dout_got = got_model.deriv(got_der, 0, Matrix<double>(above_deriv), &got_in_der);
      int dout_expect = expect_model.deriv(expect_der, 0, Matrix<double>(above_deriv), &expect_in_der);
      std::cout << "Offset got " << dout_got << " expect " << dout_expect << std::endl;
      assert(dout_got == dout_expect);

      if(rank == 0){
	std::cout << "Got der:\n" << got_der << "\nExpect der:\n" << expect_der << std::endl;
	std::cout << "Got input der:\n" << got_in_der << "\nExpect in der:\n" << expect_in_der << std::endl;

	assert(near(got_der,expect_der,1e-6,true));
	assert(near(got_in_der,expect_in_der,1e-6,true));
      }
    }

    //check update
    Vector<double> new_params(got_model.nparams());
    uniformRandom(new_params,rng);
    Vector<double> dummy_params(got_model.nparams(), 0.);
    
    int poff = got_model.update(0, rank == 0 ? new_params : dummy_params); //ensure params are passed from rank 0
    int eoff = expect_model.update(0, new_params);
    assert(poff == eoff);

    Matrix<double> input(fan_in, batch_size);
    uniformRandom(input, rng);
     
    Matrix<double> expect = expect_model.value(input);
    Matrix<double> got = got_model.value(input);
    if(rank == 0) assert(abs_near(expect,got,1e-6,true));

    //check step
    poff = got_model.step(0, rank == 0 ? new_params : dummy_params, 0.567); //ensure params are passed from rank 0
    eoff = expect_model.step(0, new_params, 0.567);
    assert(poff == eoff);
    
    expect = expect_model.value(input);
    got = got_model.value(input);
    if(rank == 0) assert(abs_near(expect,got,1e-6,true));

    //check getparams
    got_model.update(0, rank == 0 ? new_params : dummy_params); //ensure params are passed from rank 0
   
    Vector<double> pgot_params(got_model.nparams());
    poff = got_model.getParams(pgot_params, 0);
    assert(poff == eoff);
    if(rank == 0) assert(equal(pgot_params, new_params, true));
    
    //check FLOPS
    size_t ef = expect_model.FLOPS(0);
    size_t gf = got_model.FLOPS(0);
    if(rank == 0) assert(ef == gf);
    
    ef = expect_model.FLOPS(1);
    gf = got_model.FLOPS(1);
    if(rank == 0) assert(ef == gf);

    
  }

  

}






int main(int argc, char** argv){
  initialize(argc, argv);

  //testPipeline();
  testPipelineLayer();
  return 0;
}
