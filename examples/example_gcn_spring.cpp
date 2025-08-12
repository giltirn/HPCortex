//An example of using a graph convolutional network to describe a system of balls attached to springs

#include <HPCortex.hpp>
#include <Testing.hpp>

inline double force(double x, double k1, double k2, double L){
  return -k1 * x + k2 * (L-x);
}
inline double kineticEnergy(double v, double mass){
  return 0.5 * mass * pow(v,2);
}
inline double potentialEnergy(double x, double k1, double k2, double L){
  return 0.5 * k1 * pow(x,2) + 0.5 * k2 * pow(L-x,2);
}

inline double energy(double x, double v, double k1, double k2, double L, double mass){
  return kineticEnergy(v,mass) + potentialEnergy(x, k1, k2, L);
}

Graph<double> graphify(double x, double v, double EK, double Epot, double L, const GraphInitialize &ginit_unbatched){
  Graph<double> out(ginit_unbatched);
  //nodes
  {
    autoView(n, out.nodes[0].attributes[0], HostWrite);
    n(0,0) = 0.; //position
  }
  {
    autoView(n, out.nodes[0].attributes[1], HostWrite);
    n(0,0) = 0.; //velocity
  }

  {
    autoView(n, out.nodes[1].attributes[0], HostWrite);
    n(0,0) = x; //position
  }
  {
    autoView(n, out.nodes[1].attributes[1], HostWrite);
    n(0,0) = v; //velocity
  }
  
  {
    autoView(n, out.nodes[2].attributes[0], HostWrite);
    n(0,0) = L; //position
  }
  {
    autoView(n, out.nodes[2].attributes[1], HostWrite);
    n(0,0) = 0.; //velocity
  }

  //edges
  {
    autoView(n, out.edges[0].attributes[0], HostWrite);
    n(0,0) = x; //length
  }
  {
    autoView(n, out.edges[1].attributes[0], HostWrite);
    n(0,0) = L-x; //length
  }
  
  //global
  {
    autoView(n, out.global.attributes[0], HostWrite);
    n(0,0) = EK; //kinetic energy
  }
  {
    autoView(n, out.global.attributes[1], HostWrite);
    n(0,0) = Epot; //potential energy
  }
  
  return out;
}

template<typename FloatType>
class MSEcostFunc<Graph<FloatType> >{
  mutable GraphInitialize ginit;
  mutable bool setup;

public:
  typedef Graph<FloatType> DataType;
  typedef DataType ComparisonType;
  typedef DataType PredictionType;
  
  MSEcostFunc(): setup(false){}
  
  FloatType loss(const ComparisonType &y, const PredictionType &ypred) const{
    if(!setup){
      ginit = y.getInitializer();
      setup = true;
    }
    
    FloatType out = 0.;
    assert(y.nodes.size() == ypred.nodes.size());
    for(int n=0;n<y.nodes.size();n++){
      assert(y.nodes[n].attributes.size() == ypred.nodes[n].attributes.size());
      for(int a=0;a<y.nodes[n].attributes.size();a++)
	out += MSEcostFunc< Tensor<FloatType,2> >::loss(y.nodes[n].attributes[a], ypred.nodes[n].attributes[a]);
    }
    assert(y.edges.size() == ypred.edges.size());
    for(int e=0;e<y.edges.size();e++){
      assert(y.edges[e].attributes.size() == ypred.edges[e].attributes.size() && y.edges[e].send_node == ypred.edges[e].send_node && y.edges[e].recv_node == ypred.edges[e].recv_node);
      
      for(int a=0;a<y.edges[e].attributes.size();a++)
	out += MSEcostFunc< Tensor<FloatType,2> >::loss(y.edges[e].attributes[a], ypred.edges[e].attributes[a]);
    }

    for(int a=0;a<y.global.attributes.size();a++)
      out += MSEcostFunc< Tensor<FloatType,2> >::loss(y.global.attributes[a], ypred.global.attributes[a]);
    return out;
  }

  //dloss / dypred_i
  PredictionType layer_deriv(const ComparisonType &y, const PredictionType &ypred) const{
    assert(setup);
    PredictionType out(ginit);
    
    for(int n=0;n<y.nodes.size();n++){
      for(int a=0;a<y.nodes[n].attributes.size();a++)
	out.nodes[n].attributes[a] = MSEcostFunc< Tensor<FloatType,2> >::layer_deriv(y.nodes[n].attributes[a], ypred.nodes[n].attributes[a]);
    }
    for(int e=0;e<y.edges.size();e++){
      for(int a=0;a<y.edges[e].attributes.size();a++)
	out.edges[e].attributes[a] = MSEcostFunc< Tensor<FloatType,2> >::layer_deriv(y.edges[e].attributes[a], ypred.edges[e].attributes[a]);
    }
    for(int a=0;a<y.global.attributes.size();a++)
      out.global.attributes[a] = MSEcostFunc< Tensor<FloatType,2> >::layer_deriv(y.global.attributes[a], ypred.global.attributes[a]);
    return out;
  }
};

//normalize tonrm, return mean and std
std::pair<double,double> normalize(std::vector<double> &tonrm){
  double mu = 0.;
  double std = 0.;
  for(double v: tonrm){
    mu += v;
    std += v*v;
  }
  mu = mu / tonrm.size();
  std = sqrt(  std/tonrm.size() - mu*mu );
  for(double &v : tonrm)
    v = (v - mu)/std;
  return std::pair<double,double>(mu,std);
}

inline double unnormalize(double v, const std::pair<double,double> &mu_sigma){
  return v * mu_sigma.second + mu_sigma.first;
}


//normalize two vectors together based on the mean, std of their sym; must be the same length!
std::pair<double,double> normalizeBySum(std::vector<double> &tonrm1, std::vector<double> &tonrm2){
  assert(tonrm1.size() == tonrm2.size());
  int N = tonrm1.size();
  double mu = 0.;
  double std = 0.;
  for(int i=0;i<N;i++){
    double v = tonrm1[i] + tonrm2[i];
    mu += v;
    std += v*v;
  }
  mu = mu / N;
  std = sqrt(  std/N - mu*mu );
  for(double &v : tonrm1)
    v = (v - mu/2)/std;
  for(double &v : tonrm2)
    v = (v - mu/2)/std;
  
  return std::pair<double,double>(mu,std);
}  


struct GraphDataLoader{
  GraphInitialize ginit_unbatched;
  const std::vector< std::pair<Graph<double>, Graph<double> > > &in_out_data_train;
  GraphDataLoader(const GraphInitialize &ginit_unbatched, const std::vector< std::pair<Graph<double>, Graph<double> > > &in_out_data_train): ginit_unbatched(ginit_unbatched), in_out_data_train(in_out_data_train){}
  
  struct Elem{
    Graph<double> x;
    Graph<double> y;
  };
  
  Elem batch(int const* indices, int batch_size) const{
    GraphInitialize ginit_batched(ginit_unbatched);
    ginit_batched.batch_size = batch_size;

    Elem out;
    out.x = Graph<double>(ginit_batched);
    out.y = Graph<double>(ginit_batched);
    for(int b=0;b<batch_size;b++){
      int idx = indices[b];
      out.x.insertBatch(in_out_data_train[idx].first,b);
      out.y.insertBatch(in_out_data_train[idx].second,b);
    }
    return out;
  }
  size_t size() const{ return in_out_data_train.size(); }
};

std::vector<double> getX(const Graph<double> &graph, int batch_idx = 0){
  std::vector<double> out(graph.nodes.size());
  for(int n=0;n<graph.nodes.size();n++){  
    autoView(a_v, graph.nodes[n].attributes[0], HostRead);
    out[n] = a_v(0,batch_idx);
  }
  return out;
}
std::pair<double,double> getEnergies(const Graph<double> &graph, int batch_idx = 0){
  std::pair<double,double> out;
  {
    autoView(a_v, graph.global.attributes[0],HostRead);
    out.first = a_v(0,batch_idx);
  }
  {
    autoView(a_v, graph.global.attributes[1],HostRead);
    out.second = a_v(0,batch_idx);
  }
  return out;
}

void springSystem(){
  std::mt19937 rng(1234);
  typedef confDouble Config;
  typedef double FloatType;

  //we have one node fixed at 0 and another at L
  //a ball in between is attached to 2 springs that are themselves attached to the nodes above
  double L =3;
  
  double v=0.1; //initial velocity
  double x=0.5 * L; //initial position
  
  double mass = 1.0;
  double k1 = 2.0;
  double k2 = 3.0;
  
  int steps = 10000;
  double dt = 0.05;
  int print_freq=20;

  std::vector<double> xvals(steps), vvals(steps), tvals(steps), EKvals(steps), EpotVals(steps);

  v += force(x,k1,k2,L) * dt/2./mass; //initialize v_t+1/2
  
  double t = 0;
  for(int step=0;step<steps;step++){
    double Epot = potentialEnergy(x,k1,k2,L);
    double EK = kineticEnergy(v,mass);
        
    if(step % print_freq == 0){
      std::cout << step << " " << x << " " << EK << " " << Epot << " " << EK+Epot << std::endl;
    }
    xvals[step] = x;
    vvals[step] = v;
    tvals[step] = t;
    EKvals[step] = EK;
    EpotVals[step] = Epot;
    
    x += v * dt;   
    v += force(x,k1,k2,L) * dt/mass;
    t += dt;
  }

  //normalize
  //auto mu_std_x = normalize(xvals);

  //we will manually normalize the positions to lie between 0 and 1 such that the spring lengths are sensible. Currently they lie between 0 and L so we can just divide by L
  std::pair<double,double> mu_std_x(0., L);
  for(auto &v : xvals) v /= L;
  double Lnorm = 1.0;
    
  auto mu_std_v = normalize(vvals);
  
  auto mu_std_EK = normalize(EKvals);
  auto mu_std_Epot = normalize(EpotVals);
  //as they are separately normalized, their sum won't be conserved, instead   EK * sigma_EK + mu_EK   +   Epot * sigma_Epot + mu_Epot    will be
  
  //setup a GCN model
  GraphInitialize ginit;
  ginit.nnode = 3; //node 1 is the actual ball
  ginit.node_attr_sizes = std::vector<int>({1,1}); //position, velocity
  ginit.edge_attr_sizes = std::vector<int>({1}); //length
  ginit.edge_map = std::vector<std::pair<int,int> >({ {0,1}, {2,1} }); //0, 1 are send nodes, 1 is a recv node. Only receive nodes pull in edge information
  ginit.global_attr_sizes = std::vector<int>({1,1}); //kinetic, potential energy
  ginit.batch_size = 16;
  
  //We want to learn a mapping between one step and the next on the trajectory
  std::cout << "Inserting data into graphs" << std::endl;
  GraphInitialize ginit_unbatched = ginit;
  ginit_unbatched.batch_size = 1;
  std::vector< std::pair<Graph<double>, Graph<double> > > in_out_data;
  for(int i=0; i < steps; i+=2){
    std::pair<Graph<double>, Graph<double> > io;
    io.first = graphify(xvals[i], vvals[i], EKvals[i], EpotVals[i], Lnorm,  ginit_unbatched);
    io.second = graphify(xvals[i+1], vvals[i+1], EKvals[i+1], EpotVals[i+1], Lnorm,  ginit_unbatched);
    in_out_data.push_back( std::move(io) );
  }
  //Shuffle data and segment into training and validation
  std::cout << "Splitting training and validation data" << std::endl;
  std::uniform_int_distribution<int> dist(0,in_out_data.size()-1);
  std::random_shuffle ( in_out_data.begin(), in_out_data.end(), [&](const int l){ return dist(rng); }  );

  int ntrain = int(0.8 * in_out_data.size());
  std::vector< std::pair<Graph<double>, Graph<double> > > in_out_data_train( in_out_data.begin(), in_out_data.begin() + ntrain );
  std::vector< std::pair<Graph<double>, Graph<double> > > in_out_data_valid( in_out_data.begin() + ntrain, in_out_data.end() );
   
  GraphDataLoader loader(ginit_unbatched, in_out_data_train);

  std::cout << "Constructing model" << std::endl;
  auto model = GCNblock(ginit,
			[&](int fan_out, int fan_in, auto &&in){ //edge update
			  int hidden_size = 8;
			  return batch_tensor_dnn_layer<3>(1, fan_out, hidden_size, noActivation<FloatType>(),
							   batch_tensor_dnn_layer<3>(1, hidden_size, fan_in, ReLU<FloatType>(),
										     std::forward<decltype(in)>(in) ) );
			},
			[&](int fan_out, int fan_in, auto &&in){ //node update
			  int hidden_size = 8;
			  return batch_tensor_dnn_layer<3>(1, fan_out, hidden_size, noActivation<FloatType>(),
							   batch_tensor_dnn_layer<3>(1, hidden_size, fan_in, ReLU<FloatType>(),
										     std::forward<decltype(in)>(in) ) );
			},
			[&](int fan_out, int fan_in, auto &&in){ //global update
			  int hidden_size = 8;
			  return batch_tensor_dnn_layer<2>(0, fan_out, hidden_size, noActivation<FloatType>(),
							   batch_tensor_dnn_layer<2>(0, hidden_size, fan_in, ReLU<FloatType>(),
										     std::forward<decltype(in)>(in) ) );
			},
			input_layer<Config, Graph<FloatType> >()
			);
  
  auto loss = mse_cost(model);

  std::cout << "Training" << std::endl;
  int nepoch = 50;
  DecayScheduler<double> lr(0.01, 0.05);
  AdamOptimizer<double, DecayScheduler<double> > opt(lr);
  //ProfilerStart("train_profile.prof");
  train(loss, loader, opt, nepoch, ginit.batch_size);
  //ProfilerStop();
  
  std::cout << "Examining model performance" << std::endl;
  int ntest = std::min(20, int(in_out_data_valid.size()));
  for(int test=0;test<ntest;test++){
    Graph<FloatType> ing(ginit);
    ing.insertBatch(in_out_data_valid[test].first,0);
    
    Graph<FloatType> og = loss.predict(ing);

    std::cout << "Validation test idx " << test << std::endl;
    std::vector<double> init_x = getX(in_out_data_valid[test].first);
    std::vector<double> true_out_x = getX(in_out_data_valid[test].second);
    std::vector<double> pred_out_x = getX(og);
    for(int n=0;n<3;n++){
      double got = unnormalize(pred_out_x[n], mu_std_x);
      double expect = unnormalize(true_out_x[n], mu_std_x);
      
      std::cout << "Node " << n
		<< " input x " << unnormalize(init_x[n], mu_std_x)
		<< " got x " << got
		<< " expect x " << expect << " diff " << got - expect << std::endl;
    }
    std::pair<double,double> Egot = getEnergies(og);
    std::pair<double,double> Eexpect = getEnergies(in_out_data_valid[test].second);
    
    Egot.first = unnormalize(Egot.first, mu_std_EK);
    Eexpect.first = unnormalize(Eexpect.first, mu_std_EK); 

    Egot.second = unnormalize(Egot.second, mu_std_Epot);
    Eexpect.second = unnormalize(Eexpect.second, mu_std_Epot); 

    std::cout << "EK got " << Egot.first << " expect " << Eexpect.first << " diff " << Egot.first - Eexpect.first << std::endl;
    std::cout << "Epot got " << Egot.second << " expect " << Eexpect.second << " diff " << Egot.second - Eexpect.second << std::endl;
    std::cout << "Total got " << Egot.first + Egot.second << " expect " << Eexpect.first + Eexpect.second << " diff " << Egot.first + Egot.second -  Eexpect.first - Eexpect.second << std::endl;
	
    std::cout << std::endl;
  }
  
}

int main(int argc, char** argv){
  initialize(argc,argv);
  springSystem();
  return 0;
}
