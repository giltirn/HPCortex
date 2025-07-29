#include <memory>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <cassert>

#include <Comms.hpp>

void Communicators::setupDDPcommunicator(){   
  //Share the mapping between pipeline rank and world rank  
  std::vector<int> pipeline_ranks(world_nrank,0);
  pipeline_ranks[world_rank] = pipeline_rank;

  assert(MPI_Allreduce(MPI_IN_PLACE, pipeline_ranks.data(), world_nrank, MPI_INT, MPI_SUM, MPI_COMM_WORLD) == MPI_SUCCESS );

  //figure out who the pipeline leaders are
  std::vector<int> ddp_comm_leaders;
  for(int w=0;w<world_nrank;w++)
    if(pipeline_ranks[w] == 0)
      ddp_comm_leaders.push_back(w);

  //make a new group containing the pipeline leaders for the batch communicator
  MPI_Group world_group;
  assert( MPI_Comm_group(MPI_COMM_WORLD, &world_group) == MPI_SUCCESS );
    
  MPI_Group ddp_group;
  assert( MPI_Group_incl(world_group, ddp_comm_leaders.size(), ddp_comm_leaders.data(), &ddp_group) == MPI_SUCCESS );

  assert( MPI_Comm_create(MPI_COMM_WORLD, ddp_group, &ddp_comm) == MPI_SUCCESS );

  assert( MPI_Group_free(&world_group) == MPI_SUCCESS );

  if(is_pipeline_leader){
    assert( MPI_Comm_rank(ddp_comm, &ddp_rank) == MPI_SUCCESS );
    assert( MPI_Comm_size(ddp_comm, &ddp_nrank) == MPI_SUCCESS );
  }

  //Ensure that all ranks in the pipeline know how what ddp rank and how many ddp ranks there are, even if they are not part of the group
  if(pipeline_nrank>1){
    assert( MPI_Bcast(&ddp_nrank, 1, MPI_INT, 0, pipeline_comm) == MPI_SUCCESS );
    assert( MPI_Bcast(&ddp_rank, 1, MPI_INT, 0, pipeline_comm) == MPI_SUCCESS );
  }
      
}

void Communicators::freeCommunicators(){
  if(is_pipeline_leader) assert( MPI_Comm_free(&ddp_comm) == MPI_SUCCESS );
  MPI_Comm_free(&pipeline_comm);
}

void Communicators::createCommJustThisRank(int world_rank, MPI_Comm &comm){
  MPI_Group world_group;
  assert( MPI_Comm_group(MPI_COMM_WORLD, &world_group) == MPI_SUCCESS );
    
  MPI_Group this_rank_group;
  assert( MPI_Group_incl(world_group, 1, &world_rank, &this_rank_group) == MPI_SUCCESS );

  assert( MPI_Comm_create(MPI_COMM_WORLD, this_rank_group, &comm) == MPI_SUCCESS );
    
  assert( MPI_Group_free(&world_group) == MPI_SUCCESS );
  assert( MPI_Group_free(&this_rank_group) == MPI_SUCCESS );
}

void Communicators::enableDDPnoPipeliningInternal(){
  //duplicate comm world to the ddp communicator
  assert( MPI_Comm_dup(MPI_COMM_WORLD, &ddp_comm) == MPI_SUCCESS );

  //setup communicator spanning just this rank
  createCommJustThisRank(world_rank, pipeline_comm);
    
  pipeline_rank = 0;
  ddp_rank = world_rank;

  pipeline_nrank = 1;
  is_pipeline_leader = true;
  ddp_nrank = world_nrank;
}

Communicators::Communicators(int argc, char** argv){
  MPI_Init(&argc, &argv);
  MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
      
  assert( MPI_Comm_rank(MPI_COMM_WORLD, &world_rank) == MPI_SUCCESS );
  assert( MPI_Comm_size(MPI_COMM_WORLD, &world_nrank) == MPI_SUCCESS );

  //Figure out node-local rank
  MPI_Comm comm_local;
  assert( MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
				MPI_INFO_NULL, &comm_local) == MPI_SUCCESS );
  assert( MPI_Comm_size(comm_local, &node_nrank) == MPI_SUCCESS );
  assert( MPI_Comm_rank(comm_local, &node_rank) == MPI_SUCCESS );
  MPI_Comm_free(&comm_local);
  
  //By default the setup is just for DDP, with an MPI communicator spanning all nodes and
  //the pipeline communicators local to each node.
  enableDDPnoPipeliningInternal();
}
   
Communicators::~Communicators(){
  freeCommunicators();
  MPI_Finalize();
}


void Communicators::enableNodePipelining(){
  freeCommunicators();
    
  //Creat the subcommunicator spanning this node
  assert( MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
			      MPI_INFO_NULL, &pipeline_comm) == MPI_SUCCESS );
  assert( MPI_Comm_size(pipeline_comm, &pipeline_nrank) == MPI_SUCCESS );
  assert( MPI_Comm_rank(pipeline_comm, &pipeline_rank) == MPI_SUCCESS );
  is_pipeline_leader = pipeline_rank == 0;
    
  setupDDPcommunicator();
}

//Define pipeline groups based on a "color" shared by ranks within the pipeline group
void Communicators::enableColorPipelining(int rank_color){
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

  setupDDPcommunicator();
}

void Communicators::enableGlobalPipelining(){
  freeCommunicators();
  assert( MPI_Comm_dup(MPI_COMM_WORLD, &pipeline_comm) == MPI_SUCCESS );
  assert( MPI_Comm_size(pipeline_comm, &pipeline_nrank) == MPI_SUCCESS );
  assert( MPI_Comm_rank(pipeline_comm, &pipeline_rank) == MPI_SUCCESS );
  is_pipeline_leader = pipeline_rank == 0;
  setupDDPcommunicator();
}


void Communicators::disableParallelism(){
  freeCommunicators();
  createCommJustThisRank(world_rank, pipeline_comm);
  createCommJustThisRank(world_rank, ddp_comm);
        
  pipeline_rank = 0;
  ddp_rank = 0;

  pipeline_nrank = 1;
  is_pipeline_leader = true;
  ddp_nrank = 1;
}

void Communicators::enableDDPnoPipelining(){
  freeCommunicators();
  enableDDPnoPipeliningInternal();
}
    
  
void Communicators::reportSetup(){
  for(int w=0;w<world_nrank;w++){
    assert( MPI_Barrier(MPI_COMM_WORLD) == MPI_SUCCESS );
    if(w == world_rank)
      std::cout << "world:" << world_rank << "/" << world_nrank
		<< " node:" << node_rank << "/" << node_nrank
		<< " ddp:" << ddp_rank << "/" << ddp_nrank
		<< " pipeline:" << pipeline_rank << "/" << pipeline_nrank
		<< std::endl << std::flush;
  }
}


static std::unique_ptr<Communicators> &_communicators_internal(){
  static std::unique_ptr<Communicators> c;
  return c;
}

Communicators & communicators(){
  auto &c = _communicators_internal();
  if(!c) throw std::runtime_error("Global Communicators instance has not been initialized");
  return *c;
}

void initializeComms(int argc, char** argv){
  auto &c = _communicators_internal();
  if(c) return; 
  c.reset(new Communicators(argc, argv));
}

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
