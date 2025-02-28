#include<Comms.hpp>
#include<Accelerator.hpp>

void initialize(int argc, char** argv){
  initializeComms(argc,argv);
  communicators().reportSetup();
  acceleratorInit();
  acceleratorReport();
}
