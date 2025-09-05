#include <HPCortex.hpp>
#include <Testing.hpp>

#include "test_gnn/Graph.hpp"
#include "test_gnn/EdgeAggregate.hpp"
#include "test_gnn/EdgeUpdate.hpp"
#include "test_gnn/NodeAggregate.hpp"
#include "test_gnn/NodeUpdate.hpp"
#include "test_gnn/GlobalUpdate.hpp"
#include "test_gnn/GCNblock.hpp"

int main(int argc, char** argv){
  initialize(argc,argv);
  testGraph();
  testExtractEdgeUpdateInputComponent();
  testInsertEdgeUpdateOutput();
  testEdgeAggregateSum();
  testExtractNodeUpdateInputComponent();
  testInsertNodeUpdateOutput();
  testEdgeAggregateGlobalSum();
  testNodeAggregateGlobalSum();
  testEdgeUpdateBlock();
  testNodeUpdateBlock();
  testExtractGlobalUpdateInputComponent();
  testInsertGlobalUpdateOutput();
  testGlobalUpdateBlock();
  testGCNblock();
  return 0;
}
