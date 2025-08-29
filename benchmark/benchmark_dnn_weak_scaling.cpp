//A strong-scaling benchmark using a deep fully-connected network of variable size
//The total batch size is scaled to the number of ranks

#include<HPCortex.hpp>


int main(int argc, char** argv){
  initialize(argc,argv);
  communicators().reportSetup();
    
  int hidden_layers = 5;
  int input_features = 32;
  int output_features = 32;
  int hidden_neurons = 128;
  int rank_batch_size = 32;
  int nbatch_per_epoch = 100;
  int nepoch = 10;

  int arg=1;
  while(arg < argc){
    std::string sarg(argv[arg]);
    if(sarg == "--hidden_layers"){
      hidden_layers = std::stoi(argv[arg+1]);
      arg+=2;
    }else if(sarg == "--input_features"){
      input_features = std::stoi(argv[arg+1]);
      arg+=2;
    }else if(sarg == "--output_features"){
      output_features = std::stoi(argv[arg+1]);
      arg+=2;
    }else if(sarg == "--hidden_neurons"){
      hidden_neurons = std::stoi(argv[arg+1]);
      arg+=2;
    }else if(sarg == "--rank_batch_size"){
      rank_batch_size = std::stoi(argv[arg+1]);
      arg+=2;
    }else if(sarg == "--nbatch_per_epoch"){
      nbatch_per_epoch = std::stoi(argv[arg+1]);
      arg+=2;
    }else if(sarg == "--nepoch"){
      nepoch = std::stoi(argv[arg+1]);
      arg+=2;
    }else{
      std::cout << "Unknown cmdline argument: " << sarg << std::endl;
      assert(0);
    }
  }
  
  
  std::mt19937 rng(1234);

  auto model_body = enwrap( dnn_layer(hidden_neurons, input_features, ReLU<float>(),
				 input_layer< confSingle, Matrix<float> >()
				 )
		       );
  for(int l=0;l<hidden_layers-2;l++) //the first dnn_layer has one hidden layer, as does the last
    model_body = enwrap( dnn_layer(hidden_neurons,hidden_neurons, ReLU<float>(), std::move(model_body)) );

  auto model = dnn_layer(output_features, hidden_neurons, noActivation<float>(), model_body);

  std::cout << "Params: " << model.nparams() << std::endl;
  
  int nrank = communicators().ddpNrank();
  int ndata = rank_batch_size * nrank;
  std::vector<XYpair<float,1,1> > data(ndata);
  for(auto &d : data){
    d.x = Vector<float>(input_features);
    d.y = Vector<float>(output_features);
    uniformRandom(d.x,rng);
    uniformRandom(d.y,rng);
  }

  auto loss = mse_cost(model);
  
  AdamOptimizer<float> opt(0.01);
  XYpairDataLoader<float,1,1> loader(data);
  train(loss, loader, opt, nepoch, ndata);
}
