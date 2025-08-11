template<typename Config>
Graph<typename Config::FloatType> NodeAggregateGlobalSumComponent<Config>::value(const Graph<FloatType> &graph){
  if(!setup){
    ginit = graph.getInitializer();
    ginit_out = ginit;
    ginit_out.nnode = 1;
    nnode_attr = ginit.node_attr_sizes.size();
    setup = true;
  }
    
  Graph<FloatType> out(ginit_out);
  out.edges = graph.edges;
  out.global = graph.global;

  Node<FloatType> &node_out = out.nodes[0];
  for(int n=0;n<graph.nodes.size();n++){
    const Node<FloatType> &node_in = graph.nodes[n];
    for(int a=0;a<nnode_attr;a++){
      node_out.attributes[a] += node_in.attributes[a];
      if(!value_flops.locked()) value_flops.add( node_in.attributes[a].size(0) * node_in.attributes[a].size(1) );
    }
  }    
  value_flops.lock();
  return out;
}

template<typename Config>
void NodeAggregateGlobalSumComponent<Config>::deriv(Graph<FloatType> &&_dCost_by_dOut, Graph<FloatType> &dCost_by_dIn) const{
  assert(setup);
  Graph<FloatType> dCost_by_dOut(std::move(_dCost_by_dOut));
  dCost_by_dIn = Graph<FloatType>(ginit);
  dCost_by_dIn.edges = dCost_by_dOut.edges;
  dCost_by_dIn.global = dCost_by_dOut.global;

  const Node<FloatType> &node_in = dCost_by_dOut.nodes[0];
  for(int e=0;e<dCost_by_dIn.nodes.size();e++){
    Node<FloatType> &node_out = dCost_by_dIn.nodes[e];
    for(int a=0;a<nnode_attr;a++)
      node_out.attributes[a] = node_in.attributes[a];      
  }
}
