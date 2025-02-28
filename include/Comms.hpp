#pragma once
#include <mpi.h>

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
  Communicators(int argc, char** argv);
  
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

//Initialize the library communications
void initializeComms(int argc, char** argv);
