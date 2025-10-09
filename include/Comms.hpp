#pragma once
#include <mpi.h>
#include <cassert>
#include <HPCortexConfig.h>
#include <Tensors.hpp>

//Communicators for batch and pipeline parallelism
class Communicators{
  MPI_Comm pipeline_comm;
  MPI_Comm ddp_comm;
  int world_rank;
  int world_nrank;
  int node_rank;
  int node_nrank;
  
  int pipeline_rank;
  int pipeline_nrank;
  bool is_pipeline_leader; //indicate this rank is the head of the pipeline and thus part of the ddp communicator
  
  int ddp_rank;
  int ddp_nrank;
private:
  //Once a pipeline communicator has been built, figure out leader ranks in the pipeline groups
  //and build the DDP communicator
  void setupDDPcommunicator();

  void freeCommunicators();

  static void createCommJustThisRank(int world_rank, MPI_Comm &comm);
  
  void enableDDPnoPipeliningInternal();
  
public:
  Communicators();
  
  ~Communicators();

  //Ranks within MPI_COMM_WORLD
  inline int worldRank() const{
    return world_rank;
  }
  inline int worldNrank() const{
    return world_nrank;
  }

  //Ranks on this node
  inline int nodeRank() const{
    return node_rank;
  }
  inline int nodeNrank() const{
    return node_nrank;
  }

  //Ranks within DDP subgroup  
  inline int ddpRank() const{
    return ddp_rank;
  }
  inline int ddpNrank() const{
    return ddp_nrank;
  }

  //Ranks within pipeline subgroup
  inline int pipelineRank() const{
    return pipeline_rank;
  }
  inline int pipelineNrank() const{
    return pipeline_nrank;
  }
  //This rank is the leader of the pipeline subcommunicator
  //MPI calls between different pipeline blocks should only occur on the leader ranks
  inline bool isPipelineLeader() const{
    return is_pipeline_leader;
  }
  
  inline MPI_Comm & pipelineCommunicator(){ return pipeline_comm; }
  inline MPI_Comm & ddpCommunicator(){ return ddp_comm; }
  
  //Enable pipeline parallelism spanning the ranks on this node
  void enableNodePipelining();

  //Define pipeline groups based on a "color" shared by ranks within the pipeline group
  //DDP is used between groups of different color
  void enableColorPipelining(int rank_color);
    
  //Disable DDP and use all ranks for pipeline parallelism
  void enableGlobalPipelining();

  //Disable both DDP and pipeline parallelism. All communicators span just the current rank.
  void disableParallelism();
  
  //(default) Enable DDP but no pipelining, with an MPI communicator spanning all nodes and the pipeline communicators local to each node.  
  void enableDDPnoPipelining();
    
  //Output information about the communicators to cout
  void reportSetup();
};

//A global Communicators instance should be a singleton
Communicators & communicators();

//A unique index for this rank
inline int UniqueID(){ return communicators().worldRank(); }

template<typename FloatType>
inline MPI_Datatype getMPIdataType();

/**
 * @brief Return true if MPI has been initialized and not yet finalized
 */
bool MPIisActive();

//MPI reduction
template<typename FloatType>
inline void commsReduce(FloatType *data, size_t data_len, const MPI_Comm &comm);

template<typename FloatType>
inline void commsReduce(Vector<FloatType> &v, const MPI_Comm &comm);

//MPI broadcast
template<typename FloatType>
inline void commsBroadcast(FloatType* data, size_t data_len, int from_rank, const MPI_Comm &comm);

template<typename FloatType>
inline void commsBroadcast(Vector<FloatType> &v, int from_rank, const MPI_Comm &comm);

template<typename FloatType>
inline void commsBroadcast(Matrix<FloatType> &v, int from_rank, const MPI_Comm &comm);

/**
 * @brief A generic callback function applied after comms have completed
 */
struct PostCommActionCallback{
  virtual void performAction() = 0;
  virtual ~PostCommActionCallback(){}
};

/**
 * @brief A post-comms callback to unlock a managed object
 */
template<typename T>
struct PostCommActionCallbackUnlock: public PostCommActionCallback{
  T const* v;  
  PostCommActionCallbackUnlock(T const* v): v(v){}  
  void performAction() override{ v->unlock(); }
};

/**
 * @brief A post-comms callback to initialize a tensor. The associated comms should populate the "size" field
 */
template<typename FloatType, int Dim>
struct PostCommActionCallbackTensorInitialize: public PostCommActionCallback{
  std::unique_ptr<Tensor<FloatType,Dim> > &tens;
  int size[Dim];
  PostCommActionCallbackTensorInitialize(std::unique_ptr<Tensor<FloatType,Dim> > &tens): tens(tens){}  
  void performAction() override{
    tens.reset(new Tensor<FloatType,Dim>(size));
  }
};

/**
 * @brief A comms request with a callback hook
 */
struct CommsRequest{
  MPI_Request req;
  std::unique_ptr<PostCommActionCallback> post;
};

/**
 * @brief Wait for all comms activity on the associated requests to complete
 */
void waitAll(std::vector<CommsRequest> &reqs);

#include "implementation/Comms.tcc"
