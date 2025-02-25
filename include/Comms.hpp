#pragma once
#include <mpi.h>
#include <memory>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <cassert>

//Communicators for batch and pipeline parallelism
class Communicators{
  MPI_Comm pipeline_comm;
  MPI_Comm batch_comm;
  int world_rank;
  int world_nrank;
  int pipeline_rank;
  int pipeline_nrank;
  bool is_pipeline_leader; //indicate you are part of the batch group
  
  int batch_rank;
  int batch_nrank;
private:
  //Once a pipeline communicator has been built, figure out leader ranks in the pipeline groups
  //and build the batch communicator
  void setupBatchCommunicator(){   
    //Share the mapping between pipeline rank and world rank  
    std::vector<int> pipeline_ranks(world_nrank,0);
    pipeline_ranks[world_rank] = pipeline_rank;

    assert(MPI_Allreduce(MPI_IN_PLACE, pipeline_ranks.data(), world_nrank, MPI_INT, MPI_SUM, MPI_COMM_WORLD) == MPI_SUCCESS );

    //figure out who the pipeline leaders are
    std::vector<int> batch_comm_leaders;
    for(int w=0;w<world_nrank;w++)
      if(pipeline_ranks[w] == 0)
	batch_comm_leaders.push_back(w);

    //make a new group containing the pipeline leaders for the batch communicator
    MPI_Group world_group;
    assert( MPI_Comm_group(MPI_COMM_WORLD, &world_group) == MPI_SUCCESS );
    
    MPI_Group batch_group;
    assert( MPI_Group_incl(world_group, batch_comm_leaders.size(), batch_comm_leaders.data(), &batch_group) == MPI_SUCCESS );

    assert( MPI_Comm_create(MPI_COMM_WORLD, batch_group, &batch_comm) == MPI_SUCCESS );

    assert( MPI_Group_free(&world_group) == MPI_SUCCESS );

    if(is_pipeline_leader){
      assert( MPI_Comm_rank(batch_comm, &batch_rank) == MPI_SUCCESS );
      assert( MPI_Comm_size(batch_comm, &batch_nrank) == MPI_SUCCESS );
    }else{
      batch_rank = -1;
      batch_nrank = -1;
    }
      
  }

  void freeCommunicators(){
    if(is_pipeline_leader) assert( MPI_Comm_free(&batch_comm) == MPI_SUCCESS );
    MPI_Comm_free(&pipeline_comm);
  }
public:
  Communicators(int argc, char** argv){
    MPI_Init(&argc, &argv);
    MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
      
    //By default the setup is just for batch parallelism, with an MPI communicator spanning all nodes and
    //the pipeline communicators local to each node.

    assert( MPI_Comm_rank(MPI_COMM_WORLD, &world_rank) == MPI_SUCCESS );
    assert( MPI_Comm_size(MPI_COMM_WORLD, &world_nrank) == MPI_SUCCESS );
    
    //duplicate comm world to the batch communicator
    assert( MPI_Comm_dup(MPI_COMM_WORLD, &batch_comm) == MPI_SUCCESS );

    //setup communicator spanning just this rank
    MPI_Group world_group;
    assert( MPI_Comm_group(MPI_COMM_WORLD, &world_group) == MPI_SUCCESS );
    
    MPI_Group this_rank_group;
    assert( MPI_Group_incl(world_group, 1, &world_rank, &this_rank_group) == MPI_SUCCESS );

    assert( MPI_Comm_create(MPI_COMM_WORLD, this_rank_group, &pipeline_comm) == MPI_SUCCESS );

    pipeline_rank = 0;
    batch_rank = world_rank;

    pipeline_nrank = 1;
    is_pipeline_leader = true;
    batch_nrank = world_nrank;
    
    assert( MPI_Group_free(&world_group) == MPI_SUCCESS );
    assert( MPI_Group_free(&this_rank_group) == MPI_SUCCESS );
  }
  ~Communicators(){
    freeCommunicators();
    MPI_Finalize();
  }
    
  inline int worldRank() const{
    return world_rank;
  }
  inline int worldNrank() const{
    return world_nrank;
  }
  inline int batchRank() const{
    return batch_rank;
  }
  inline int batchNrank() const{
    return batch_nrank;
  }
  inline int pipelineRank() const{
    return pipeline_rank;
  }
  inline int pipelineNrank() const{
    return pipeline_nrank;
  }

  inline MPI_Comm & pipelineCommunicator(){ return pipeline_comm; }
  inline MPI_Comm & batchCommunicator(){ return batch_comm; }
  
  //Enable pipeline parallelism spanning the ranks on this node
  void enableNodePipelining(){
    freeCommunicators();
    
    //Creat the subcommunicator spanning this node
    assert( MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
			MPI_INFO_NULL, &pipeline_comm) == MPI_SUCCESS );
    assert( MPI_Comm_size(pipeline_comm, &pipeline_nrank) == MPI_SUCCESS );
    assert( MPI_Comm_rank(pipeline_comm, &pipeline_rank) == MPI_SUCCESS );
    is_pipeline_leader = pipeline_rank == 0;
    
    setupBatchCommunicator();
  }

  //Define pipeline groups based on a "color" shared by ranks within the pipeline group
  void enableColorPipelining(int rank_color){
    freeCommunicators();
    
    //Share colors
    std::vector<int> colors(world_nrank, 0);
    colors[world_rank] = rank_color;
    assert( MPI_Allreduce(MPI_IN_PLACE, colors.data(), world_nrank, MPI_INT, MPI_SUM, MPI_COMM_WORLD) == MPI_SUCCESS );

    //get ranks in my pipeline
    std::vector<int> my_pipeline_ranks;
    for(int w=0;w<world_nrank;w++)
      if(colors[w] == rank_color)
	my_pipeline_ranks.push_back(w);

    //create the communicator
    MPI_Group world_group;
    assert( MPI_Comm_group(MPI_COMM_WORLD, &world_group) == MPI_SUCCESS );
   
    MPI_Group my_pipeline_group;
    assert( MPI_Group_incl(world_group, my_pipeline_ranks.size(), my_pipeline_ranks.data(), &my_pipeline_group) == MPI_SUCCESS );

    assert( MPI_Comm_create(MPI_COMM_WORLD, my_pipeline_group, &pipeline_comm) == MPI_SUCCESS ); 

    assert( MPI_Comm_size(pipeline_comm, &pipeline_nrank) == MPI_SUCCESS );
    assert( MPI_Comm_rank(pipeline_comm, &pipeline_rank) == MPI_SUCCESS );
    is_pipeline_leader = pipeline_rank == 0;

    assert( MPI_Group_free(&world_group) == MPI_SUCCESS );
    assert( MPI_Group_free(&my_pipeline_group) == MPI_SUCCESS );

    setupBatchCommunicator();
  }
  //No batch parallelism
  void enableGlobalPipelining(){
    freeCommunicators();
    assert( MPI_Comm_dup(MPI_COMM_WORLD, &pipeline_comm) == MPI_SUCCESS );
    assert( MPI_Comm_size(pipeline_comm, &pipeline_nrank) == MPI_SUCCESS );
    assert( MPI_Comm_rank(pipeline_comm, &pipeline_rank) == MPI_SUCCESS );
    is_pipeline_leader = pipeline_rank == 0;
    setupBatchCommunicator();
  }
    
  void reportSetup(){
    for(int w=0;w<world_nrank;w++){
      assert( MPI_Barrier(MPI_COMM_WORLD) == MPI_SUCCESS );
      if(w == world_rank)
	std::cout << "world:" << world_rank << "/" << world_nrank
		  << " batch:" << batch_rank << "/" << batch_nrank
	  	  << " pipeline:" << pipeline_rank << "/" << pipeline_nrank
		  << std::endl << std::flush;
    }
  }
};

//A global Communicators instance should be a singleton

inline std::unique_ptr<Communicators> &_communicators_internal(){
  static std::unique_ptr<Communicators> c;
  return c;
}

inline Communicators & communicators(){
  auto &c = _communicators_internal();
  if(!c) throw std::runtime_error("Global Communicators instance has not been initialized");
  return *c;
}

void initialize(int argc, char** argv){
  auto &c = _communicators_internal();
  if(c) return; 
  c.reset(new Communicators(argc, argv));
}
