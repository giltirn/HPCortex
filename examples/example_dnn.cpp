#include <HPCortex.hpp>
#include <Testing.hpp>
#include <unordered_map>
#include <sstream>

int n_out = 10;
int n_in = 5;
int n_hidden = 25;

std::vector<XYpair<float,1,1> > generate_data(int ndata){
  //some kind of linear transformation
  Matrix<float> rot(n_out,n_in);
  uniformRandom(rot);

  std::vector<XYpair<float,1,1> > out(ndata);
  for(int d=0;d<ndata;d++){    
    Vector<float> x(n_in);
    uniformRandom(x); //all elements between -1 and 1
    Vector<float> y = rot * x;

    out[d].x = std::move(x);
    out[d].y = std::move(y);
  }
  return out;  
}

int main(int argc, char** argv){
  initialize(argc, argv);

  //Model specification 
  auto model = dnn_layer(n_out, n_hidden,
			 dnn_layer(n_hidden, n_in, ReLU<float>(),
				   input_layer<float>()				   
				   )
			 ); 

  auto loss = mse_cost( model );

  //Optimizer
  float learning_rate = 0.005;
  AdamOptimizer<float> optimizer(learning_rate);

  //Data loader
  constexpr int DataDim = 1;
  typedef std::vector<XYpair<float,DataDim,DataDim> > XYpairVector;
  int ntrain = 200;
  int nvalid = 50;  
  XYpairVector data = generate_data(ntrain + nvalid);  
  XYpairVector train_data(data.begin(),data.begin()+ntrain);
  XYpairVector valid_data(data.begin()+ntrain,data.end());
    
  XYpairDataLoader<float,DataDim,DataDim> loader(train_data);

  //Train the model
  int nepoch = 100;
  int batch_size = 4;
  std::vector<float> loss_history = train(loss, loader, optimizer, nepoch, batch_size);

  //Perform validation
  for(int i=0;i<valid_data.size();i++){
    auto const &xy = valid_data[i];
    Vector<float> prediction = loss.predict(xy.x, batch_size);
    Vector<float> diff = prediction - xy.y;
    double loss = norm2(diff) / n_out;    
    std::cout << xy.x << " -> " << loss << std::endl;
  }   
  
  return 0;
};
