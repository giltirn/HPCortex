template<typename FloatType>
inline void _edge_aggregate_sum_cpt_value(typename AttributedGraphElements<FloatType>::AttributesType &out_edges,
					  const typename AttributedGraphElements<FloatType>::AttributesType &in_edges,
					  const ManagedTypeArray<Vector<int> > &receive_map,
					  int batch_size, int nnode, int nedge_attr){

  autoView(oedges_v, out_edges, DeviceWrite);
  autoView(iedges_v, in_edges, DeviceRead);
  autoView(rcv_map_v, receive_map, DeviceRead);
  
  accelerator_for_3d_gen(1,2,normal(), b, batch_size, n, nnode, a, nedge_attr, {
      auto oedge_attr_v = oedges_v[a];
      auto iedge_attr_v = iedges_v[a];
      int attr_sz = iedge_attr_v.size(1);
      auto edge_n_rcv_map_v = rcv_map_v[n];

      for(int i=0;i<attr_sz;i++){
	FloatType sum = 0.;      
	for(int ee=0;ee<edge_n_rcv_map_v.size(0);ee++){
	  int efrom = edge_n_rcv_map_v(ee);
	  sum += iedge_attr_v(efrom, i, b);
	}
	oedge_attr_v(n,i,b) = sum;
      }
    });
}

template<typename Config>
template<typename InGraphType, enable_if_fwd_ref<InGraphType,Graph<typename Config::FloatType> > >
Graph<typename Config::FloatType> EdgeAggregateSumComponent<Config>::value(InGraphType &&in){
  
  if(!setup){
    nnode = in.nodes.nElem();
    nedge_attr = in.edges.nAttrib();

    std::vector<std::vector<int> > rcv_map(nnode);
    for(int e=0;e<in.edges.nElem();e++)
      rcv_map[ in.edges.recvNode(e) ].push_back(e);
    receive_map.resize(nnode);
    for(int n=0;n<nnode;n++){
      receive_map[n] = Vector<int>(rcv_map[n].size());
      autoView(rv, receive_map[n], HostWrite);
      memcpy(rv.data(), rcv_map[n].data(), rcv_map[n].size()*sizeof(int));
    }
        
    ginit = in.getInitializer();
      
    ginit_out = ginit;
    ginit_out.edge_map.resize(nnode);
    for(int n=0;n<nnode;n++){
      ginit_out.edge_map[n].first = -1;
      ginit_out.edge_map[n].second = n;
    }
      
    setup = true;
  }

  Graph<FloatType> out(ginit_out);
  copyOrMoveGraphElement(out, std::forward<InGraphType>(in), GraphElementType::Global);
  copyOrMoveGraphElement(out, std::forward<InGraphType>(in), GraphElementType::Nodes);
  
  int batch_size = out.edges.batchSize();
  _edge_aggregate_sum_cpt_value<FloatType>(out.edges.attributes, in.edges.attributes, receive_map, batch_size, nnode, nedge_attr);
      
  if(!value_flops.locked()){
    autoView(rcv_map_v, receive_map, HostRead);
    for(int n=0;n<nnode;n++){
      for(int ee=0;ee<rcv_map_v[n].size(0);ee++){
	for(int a=0;a<nedge_attr;a++){
	  value_flops.add( in.edges.attribSize(a) * batch_size );
	}
      }
    }
    value_flops.lock();
  }

  return out;
}

template<typename Config>
void EdgeAggregateSumComponent<Config>::deriv(Graph<FloatType> &&_dCost_by_dOut, Graph<FloatType> &dCost_by_dIn) const{
  assert(setup);
  Graph<FloatType> dCost_by_dOut(std::move(_dCost_by_dOut));
  dCost_by_dIn = Graph<FloatType>(ginit);
  dCost_by_dIn.global = std::move(dCost_by_dOut.global);
  dCost_by_dIn.nodes = std::move(dCost_by_dOut.nodes);

  int batch_size = dCost_by_dIn.nodes.batchSize();
  autoView(rcv_map_v, receive_map, DeviceRead);
  autoView(oedges_v, dCost_by_dIn.edges.attributes, DeviceWrite);
  autoView(iedges_v, dCost_by_dOut.edges.attributes, DeviceRead);

  accelerator_for_3d_gen(1,2,normal(),b,batch_size, n, nnode, a, nedge_attr, {
      auto oedge_attr_v = oedges_v[a];
      auto iedge_attr_v = iedges_v[a];
      int attr_sz = iedge_attr_v.size(1);
      auto edge_n_rcv_map_v = rcv_map_v[n];

      for(int ee=0;ee<edge_n_rcv_map_v.size(0);ee++){
	int eto = edge_n_rcv_map_v(ee);
	for(int i=0;i<attr_sz;i++)
	  oedge_attr_v(eto, i, b) = iedge_attr_v(n,i,b); //edge can only contribute to one node (receiver) so no need to sum
      }
    });
}

template<typename FloatType>
inline void _edge_aggregate_global_sum_cpt_value(typename AttributedGraphElements<FloatType>::AttributesType &out_edges,
						 const typename AttributedGraphElements<FloatType>::AttributesType &in_edges,
						 int batch_size, int nedge_attr){
  autoView(in_edges_v,in_edges,DeviceRead);
  autoView(out_edges_v,out_edges,DeviceWrite);
  
  accelerator_for_2d_gen(1,1,normal(),b,batch_size,a,nedge_attr,{
      auto in_attr_v = in_edges_v[a];
      auto out_attr_v = out_edges_v[a];

      for(int i=0;i<in_attr_v.size(1);i++){
	FloatType sum=0.;
	for(int e=0;e<in_attr_v.size(0);e++)
	  sum += in_attr_v(e,i,b);
	out_attr_v(0,i,b) = sum;
      }
    });
}
  
template<typename Config>
template<typename InGraphType, enable_if_fwd_ref<InGraphType,Graph<typename Config::FloatType> > >
Graph<typename Config::FloatType> EdgeAggregateGlobalSumComponent<Config>::value(InGraphType &&in){
  if(!setup){
    nedge_attr = in.edges.nAttrib();
    ginit = in.getInitializer();
    ginit_out = ginit;
    ginit_out.edge_map.resize(1);
    ginit_out.edge_map[0].first = -1;
    ginit_out.edge_map[0].second = -1;      
    setup = true;
  }
  Graph<FloatType> out(ginit_out);
  copyOrMoveGraphElement(out, std::forward<InGraphType>(in), GraphElementType::Global);
  copyOrMoveGraphElement(out, std::forward<InGraphType>(in), GraphElementType::Nodes);

  _edge_aggregate_global_sum_cpt_value<FloatType>(out.edges.attributes, in.edges.attributes, in.edges.batchSize(),nedge_attr);
  
  if(!value_flops.locked()){  
    for(int e=0;e<in.edges.nElem();e++)
      for(int a=0;a<nedge_attr;a++)
	value_flops.add( in.edges.attribSize(a) * in.edges.batchSize() );
    value_flops.lock();
  }    

  return out;
}

template<typename Config>
void EdgeAggregateGlobalSumComponent<Config>::deriv(Graph<FloatType> &&_dCost_by_dOut, Graph<FloatType> &dCost_by_dIn) const{
  assert(setup);
  Graph<FloatType> dCost_by_dOut(std::move(_dCost_by_dOut));
  dCost_by_dIn = Graph<FloatType>(ginit);
  dCost_by_dIn.nodes = std::move(dCost_by_dOut.nodes);
  dCost_by_dIn.global = std::move(dCost_by_dOut.global);

  autoView(in_edges_v,dCost_by_dOut.edges.attributes,DeviceRead);
  autoView(out_edges_v,dCost_by_dIn.edges.attributes,DeviceWrite);
  int batch_size = dCost_by_dIn.edges.batchSize();  
  
  accelerator_for_2d_gen(1,1,normal(),b,batch_size,a,nedge_attr,{
      auto in_attr_v = in_edges_v[a];
      auto out_attr_v = out_edges_v[a];
      
      for(int i=0;i<in_attr_v.size(1);i++){
	FloatType val = in_attr_v(0,i,b);
	for(int e=0;e<out_attr_v.size(0);e++)
	  out_attr_v(e,i,b) = val;
      }
    });  
}
