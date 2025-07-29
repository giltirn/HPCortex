#pragma once
#include <Comms.hpp>

/**
 * @brief Base class for containers for layer IO type initializers
 */
struct LayerIOcontainerInitializer{
  virtual ~LayerIOcontainerInitializer(){}
};

/**
 * @brief Layer IO initializer for Tensor objects
 */
template<int Dim>
struct LayerIOcontainerTensorInitializer: public LayerIOcontainerInitializer{
  int dims[Dim];
  LayerIOcontainerTensorInitializer(int const* in_dim){ memcpy(dims, in_dim, Dim*sizeof(int)); }    
};

/**
 * @brief Base class for containers for layer IO types
 */
class LayerIOcontainerBase{
public:
  /**
   * @brief Send the container's data to rank 'to'
   */
  virtual CommsRequest send(int to, MPI_Comm &comm) = 0;

  /**
   * @brief Receive the container's data from rank 'from'
   */
  virtual CommsRequest recv(int from, MPI_Comm &comm) = 0;

  /**
   * @brief Send the container's initialization info to rank 'to'
   */
  virtual CommsRequest sendInitializer(int to, MPI_Comm &comm) = 0;

  /**
   * @brief Receive the container's initialization info from rank 'from' and initialize the associated object
   */
  virtual CommsRequest recvInitializer(int from, MPI_Comm &comm) = 0;

  /**
   * @brief Return an object storing the container's initalization info
   */
  virtual std::unique_ptr<LayerIOcontainerInitializer> getInitializer() const = 0;

  /**
   * @brief Initialize the container from a stored instance of initalization info
   */
  virtual void initialize(const std::unique_ptr<LayerIOcontainerInitializer> &init) = 0;

  /**
   * @brief Return a deep copy of the container
   */
  virtual LayerIOcontainerBase* copy() const = 0;

  /**
   * @brief Return a new container comprising a split-off micro-batch of a specific index and size
   */  
  virtual LayerIOcontainerBase * getMicroBatch(int ubatch_idx, int ubatch_size) const = 0;

  /**
   * @brief Insert a microbatch into the container
   */
  virtual void insertMicroBatch(int ubatch_idx, int ubatch_size, LayerIOcontainerBase *from) = 0;

  /**
   * @brief Initialize the contained object and insert the first microbatch
   */
  virtual void insertFirstMicroBatch(int ubatch_size, LayerIOcontainerBase *from, int nubatch) = 0;

  /**
   * @brief Return the size of the batch dimension
   */
  virtual int batchDimSize() const = 0;
  
  virtual ~LayerIOcontainerBase(){}
};

/**
 * @brief Signature for containers for layer IO types based on type, implementing the above
 */
template<typename T>
class LayerIOcontainerImpl: public LayerIOcontainerBase{};

/**
 * @brief Implementation of the layer IO container for tensor objects
 */
template<typename FloatType, int Dim>
class LayerIOcontainerImpl<Tensor<FloatType,Dim> >: public LayerIOcontainerBase{
  std::unique_ptr<Tensor<FloatType,Dim> > tens;
public:
  LayerIOcontainerImpl(){}
  LayerIOcontainerImpl(const Tensor<FloatType,Dim> &t): tens(new Tensor<FloatType,Dim>(t)){}
  LayerIOcontainerImpl(Tensor<FloatType,Dim> &&t): tens(new Tensor<FloatType,Dim>(std::move(t))){}

  std::unique_ptr<LayerIOcontainerInitializer> getInitializer() const override;
  
  void initialize(const std::unique_ptr<LayerIOcontainerInitializer> &init) override;
  
  CommsRequest send(int to, MPI_Comm &comm) override;
  CommsRequest recv(int from, MPI_Comm &comm) override;
  CommsRequest sendInitializer(int to, MPI_Comm &comm) override;  
  CommsRequest recvInitializer(int from, MPI_Comm &comm) override;

  LayerIOcontainerBase* copy() const override;

  /**
   * @brief Get the underlying tensor
   */
  Tensor<FloatType,Dim>& get();
  
  /**
   * @brief Get the underlying tensor (const)
   */
  const Tensor<FloatType,Dim>& get() const;
  
  /**
   * @brief Release the contained tensor
   */
  Tensor<FloatType,Dim> remove();

  LayerIOcontainerBase * getMicroBatch(int ubatch_idx, int ubatch_size) const override;
  void insertMicroBatch(int ubatch_idx, int ubatch_size, LayerIOcontainerBase *from) override;
  void insertFirstMicroBatch(int ubatch_size, LayerIOcontainerBase *from, int nubatch) override;
  int batchDimSize() const override;
};

/**
 * @brief The public interface for the layer IO container
 */
struct LayerIOcontainer{
  std::unique_ptr<LayerIOcontainerBase> p;

  LayerIOcontainer(){}
  LayerIOcontainer(LayerIOcontainerBase *pin): p(pin){} //takes ownership
  
  template<typename T, typename std::enable_if<!std::is_same<typename std::decay<T>::type,LayerIOcontainer>::value, int>::type = 0>
  LayerIOcontainer(T &&v): p(new LayerIOcontainerImpl<typename std::decay<T>::type>(std::forward<T>(v))){}

  LayerIOcontainer(const LayerIOcontainer &to_copy): p(to_copy.p->copy()){}
  LayerIOcontainer(LayerIOcontainer &&to_move): p(std::move(to_move.p)){}  

  LayerIOcontainer & operator=(LayerIOcontainer &&to_move){
    p.reset(to_move.p.release());
    return *this;
  }
  template<typename T, typename std::enable_if<!std::is_same<typename std::decay<T>::type,LayerIOcontainer>::value, int>::type = 0>
  LayerIOcontainer & operator=(T &&v){
    p.reset(new LayerIOcontainerImpl<typename std::decay<T>::type>(std::forward<T>(v)));
    return *this;
  }  

  /**
   * @brief Set the stored data type but do not populate its underlying instance
   */
  template<typename T>
  void setType(){ p.reset(new LayerIOcontainerImpl<T>()); }

  /**
   * @brief Insert (copy/move) an instance of the data type into the container
   */  
  template<typename T>
  void insert(T &&v){
    p.reset(new LayerIOcontainerImpl<typename std::decay<T>::type>(std::forward<T>(v)));
  }

  /**
   * @brief Access to the underlying data type
   */ 
  template<typename T>
  T & as(){  assert(p); return dynamic_cast<LayerIOcontainerImpl<T> *>(p.get())->get(); }

  /**
   * @brief Access to the underlying data type (const)
   */ 
  template<typename T>
  const T & as() const{ assert(p); return dynamic_cast<LayerIOcontainerImpl<T> const*>(p.get())->get(); }

  std::unique_ptr<LayerIOcontainerInitializer> getInitializer() const{ assert(p); return p->getInitializer(); }

  void initialize(const std::unique_ptr<LayerIOcontainerInitializer> &init){
    assert(p); p->initialize(init);
  }
  
  template<typename T>
  T remove(){
    assert(p);
    LayerIOcontainerBase *pp = p.release();
    auto out = dynamic_cast<LayerIOcontainerImpl<T> *>(pp)->remove();
    delete pp;
    return out;
  }
    
  CommsRequest send(int to, MPI_Comm &comm){ return p->send(to,comm); }
  CommsRequest recv(int from, MPI_Comm &comm){ return p->recv(from,comm); }
  CommsRequest sendInitializer(int to, MPI_Comm &comm){ return p->sendInitializer(to,comm); }
  CommsRequest recvInitializer(int from, MPI_Comm &comm){ return p->recvInitializer(from,comm); }

  LayerIOcontainer getMicroBatch(int ubatch_idx, int ubatch_size) const{
    assert(p);
    return LayerIOcontainer(p->getMicroBatch(ubatch_idx,ubatch_size));
  }
  void insertMicroBatch(int ubatch_idx, int ubatch_size, LayerIOcontainer &from){
    assert(p);
    p->insertMicroBatch(ubatch_idx,ubatch_size,from.p.get());
  }
  void insertFirstMicroBatch(int ubatch_size, LayerIOcontainer &from, int nubatch){
    assert(p);
    p->insertFirstMicroBatch(ubatch_size,from.p.get(),nubatch);
  }
  int batchDimSize() const{
    assert(p);
    return p->batchDimSize();
  }
  
};

//////////////////////////////////////// For pipelining ////////////////////////////////////////////////////

/**
 * @brief Send a (pre-initialized) container from one rank to another
 */
void pipelineSendRecv(std::vector<CommsRequest> &reqs,
	      LayerIOcontainer &to, LayerIOcontainer &from,
	      int rank_to, int rank_from);

/**
 * @brief Send (pre-initialized) containers rightwards from from one rank to the next
 */
void pipelinePassRight(std::vector<CommsRequest> &reqs,
	       LayerIOcontainer &to, LayerIOcontainer &from);

/**
 * @brief Send (pre-initialized) containers rightwards from from one rank to the next according to a condition based on the send rank index, e.g. [=](int send_rank){ return send_rank == my_send_rank; }
 */
template<typename Cond>
void pipelinePassRightConditional(std::vector<CommsRequest> &reqs,
			  LayerIOcontainer &to, LayerIOcontainer &from, const Cond &send_rank_cond);

/**
 * @brief Send (pre-initialized) containers rightwards from from one rank to the next
 */
void pipelinePassLeft(std::vector<CommsRequest> &reqs,
	      LayerIOcontainer &to, LayerIOcontainer &from);

template<typename Cond>
void pipelinePassLeftConditional(std::vector<CommsRequest> &reqs,
			 LayerIOcontainer &to, LayerIOcontainer &from, const Cond &send_rank_cond);


void pipelineSendRecvInitializer(std::vector<CommsRequest> &reqs,
			 LayerIOcontainer &to, LayerIOcontainer &from,
			 int rank_to, int rank_from);


#include "implementation/LayerIOcontainer.tcc"
