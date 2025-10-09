#include<Comms.hpp>
#include<Accelerator.hpp>

struct HPCortexFinalizer{
  HPCortexFinalizer(){}
  ~HPCortexFinalizer(){
    acceleratorFinalize();
    MPI_Finalize();
    assert(!MPIisActive() && "MPI did not finalize properly");
  }
};    

void initialize(int argc, char** argv){
  static HPCortexFinalizer fin; //instantiate first so it is destoyed *last*
  MPI_Init(&argc, &argv);
  communicators().reportSetup(); //this will instantiate the global communicator which is needed for acceleratorInit
  acceleratorInit();
  acceleratorReport(); 
}
