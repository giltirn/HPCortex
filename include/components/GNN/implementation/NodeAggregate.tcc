template<typename FloatType>
inline void _node_aggregate_global_sum_cpt_value(typename AttributedGraphElements<FloatType>::AttributesType &out_nodes,
						 const typename AttributedGraphElements<FloatType>::AttributesType &in_nodes,
						 int batch_size, int nnode_attr){
  autoView(in_node_attr_v, in_nodes, DeviceRead);
  autoView(out_node_attr_v, out_nodes, DeviceWrite);
  
  accelerator_for_2d_gen(1,1,normal(),b, batch_size, a, nnode_attr, {
      auto in_attr_v = in_node_attr_v[a];
      auto out_attr_v = out_node_attr_v[a];
					   
      for(int i=0;i<in_attr_v.size(1);i++){
	FloatType val = 0.;
	for(int n=0;n<in_attr_v.size(0);n++)
	  val += in_attr_v(n,i,b);
	out_attr_v(0,i,b) = val;
      }
    });
}


template<typename Config>
template<typename InGraphType, enable_if_fwd_ref<InGraphType,Graph<typename Config::FloatType> > >
Graph<typename Config::FloatType> NodeAggregateGlobalSumComponent<Config>::value(InGraphType &&in){
  if(!setup){
    ginit = in.getInitializer();
    ginit_out = ginit;
    ginit_out.nnode = 1;
    nnode_attr = ginit.node_attr_sizes.size();
    setup = true;
  }
    
  Graph<FloatType> out(ginit_out);
  copyOrMoveGraphElement(out, std::forward<InGraphType>(in), GraphElementType::Global);
  copyOrMoveGraphElement(out, std::forward<InGraphType>(in), GraphElementType::Edges);

  _node_aggregate_global_sum_cpt_value<FloatType>(out.nodes.attributes, in.nodes.attributes, in.nodes.batchSize(), nnode_attr);
  
  if(!value_flops.locked()){    
    for(int n=0;n<in.nodes.nElem();n++){
      for(int a=0;a<nnode_attr;a++)	
	value_flops.add( in.nodes.attribSize(a) * in.nodes.batchSize() );
    }
    value_flops.lock();
  }    
  return out;
}

template<typename Config>
void NodeAggregateGlobalSumComponent<Config>::deriv(Graph<FloatType> &&_dCost_by_dOut, Graph<FloatType> &dCost_by_dIn) const{
  assert(setup);
  Graph<FloatType> dCost_by_dOut(std::move(_dCost_by_dOut));
  dCost_by_dIn = Graph<FloatType>(ginit);
  dCost_by_dIn.edges = std::move(dCost_by_dOut.edges);
  dCost_by_dIn.global = std::move(dCost_by_dOut.global);

  autoView(out_node_attr_v, dCost_by_dIn.nodes.attributes, DeviceWrite);
  autoView(in_node_attr_v, dCost_by_dOut.nodes.attributes, DeviceRead);

  accelerator_for_2d_gen(1,1,normal(),b, dCost_by_dOut.nodes.batchSize(), a, nnode_attr, {
      auto in_attr_v = in_node_attr_v[a];
      auto out_attr_v = out_node_attr_v[a];
					   
      for(int i=0;i<out_attr_v.size(1);i++){
	FloatType val = in_attr_v(0,i,b);
	for(int n=0;n<out_attr_v.size(0);n++)
	  out_attr_v(n,i,b) = val;
      }
    });
}
