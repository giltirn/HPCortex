#include<Graph.hpp>


bool GraphInitialize::operator==(const GraphInitialize &r) const{
  if(nnode != r.nnode || global_attr_sizes.size() != r.global_attr_sizes.size() || batch_size != r.batch_size || node_attr_sizes.size() != r.node_attr_sizes.size() || edge_attr_sizes.size() != r.edge_attr_sizes.size() || edge_map.size() != r.edge_map.size()) return false;
  for(int n=0;n<node_attr_sizes.size();n++) if(node_attr_sizes[n] != r.node_attr_sizes[n]) return false;
  for(int n=0;n<edge_attr_sizes.size();n++) if(edge_attr_sizes[n] != r.edge_attr_sizes[n]) return false;
  for(int n=0;n<edge_map.size();n++) if(edge_map[n] != r.edge_map[n]) return false;
  for(int n=0;n<global_attr_sizes.size();n++) if(global_attr_sizes[n] != r.global_attr_sizes[n]) return false;
  return true;
}
