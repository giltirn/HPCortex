template<typename Config>
Graph<typename Config::FloatType> EdgeAggregateSumComponent<Config>::value(const Graph<FloatType> &graph){
  if(!setup){
    nnode = graph.nodes.size();
    nedge_attr = graph.edges[0].attributes.size();

    receive_map.resize(nnode);
    for(int e=0;e<graph.edges.size();e++)
      receive_map[ graph.edges[e].recv_node ].push_back(e);
    ginit = graph.getInitializer();
      
    ginit_out = ginit;
    ginit_out.edge_map.resize(nnode);
    for(int n=0;n<nnode;n++){
      ginit_out.edge_map[n].first = -1;
      ginit_out.edge_map[n].second = n;
    }
      
    setup = true;
  }

  Graph<FloatType> out(ginit_out);
  out.global = graph.global;
  for(int n=0;n<nnode;n++){
    out.nodes[n] = graph.nodes[n];

    Edge<FloatType> &aggr_edge = out.edges[n];
    for(int ee=0;ee<receive_map[n].size();ee++){
      const Edge<FloatType> &edge_in = graph.edges[ receive_map[n][ee] ];
      for(int a=0;a<nedge_attr;a++){
	aggr_edge.attributes[a] += edge_in.attributes[a];
	if(!value_flops.locked()) value_flops.add( edge_in.attributes[a].size(0) * edge_in.attributes[a].size(1) );
      }
    }
  }
  value_flops.lock();
  return out;
}

template<typename Config>
void EdgeAggregateSumComponent<Config>::deriv(Graph<FloatType> &&_dCost_by_dOut, Graph<FloatType> &dCost_by_dIn) const{
  assert(setup);
  Graph<FloatType> dCost_by_dOut(std::move(_dCost_by_dOut));
  dCost_by_dIn = Graph<FloatType>(ginit);
  dCost_by_dIn.global = dCost_by_dOut.global;
    
  for(int n=0;n<nnode;n++){
    dCost_by_dIn.nodes[n] = dCost_by_dOut.nodes[n];
      
    for(int ee=0;ee<receive_map[n].size();ee++){
      Edge<FloatType> &edge_out = dCost_by_dIn.edges[ receive_map[n][ee] ];
      for(int a=0;a<nedge_attr;a++)
	edge_out.attributes[a] = dCost_by_dOut.edges[n].attributes[a]; //edge can only contribute to one node (receiver)
    }
  }
}

template<typename Config>
Graph<typename Config::FloatType> EdgeAggregateGlobalSumComponent<Config>::value(const Graph<FloatType> &graph){
  if(!setup){
    nedge_attr = graph.edges[0].attributes.size();
    ginit = graph.getInitializer();
    ginit_out = ginit;
    ginit_out.edge_map.resize(1);
    ginit_out.edge_map[0].first = -1;
    ginit_out.edge_map[0].second = -1;      
    setup = true;
  }
  Graph<FloatType> out(ginit_out);
  out.nodes = graph.nodes;
  out.global = graph.global;
    
  for(int e=0;e<graph.edges.size();e++){
    const Edge<FloatType> &edge_in = graph.edges[e];
    for(int a=0;a<nedge_attr;a++){
      out.edges[0].attributes[a] += edge_in.attributes[a];
      if(!value_flops.locked()) value_flops.add( edge_in.attributes[a].size(0) * edge_in.attributes[a].size(1) );
    }
  }    
  value_flops.lock();
  return out;
}

template<typename Config>
void EdgeAggregateGlobalSumComponent<Config>::deriv(Graph<FloatType> &&_dCost_by_dOut, Graph<FloatType> &dCost_by_dIn) const{
  assert(setup);
  Graph<FloatType> dCost_by_dOut(std::move(_dCost_by_dOut));
  dCost_by_dIn = Graph<FloatType>(ginit);
  dCost_by_dIn.nodes = dCost_by_dOut.nodes;
  dCost_by_dIn.global = dCost_by_dOut.global;

  const Edge<FloatType> &edge_in = dCost_by_dOut.edges[0];
  for(int e=0;e<dCost_by_dIn.edges.size();e++){
    Edge<FloatType> &edge_out = dCost_by_dIn.edges[e];
    for(int a=0;a<nedge_attr;a++)
      edge_out.attributes[a] = edge_in.attributes[a];      
  }
}
