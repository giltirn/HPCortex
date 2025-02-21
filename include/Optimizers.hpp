#pragma once
#include <random>
#include <algorithm>
#include <Tensors.hpp>

struct XYpair{
  Matrix x;
  Matrix y;
};


template<typename T, typename LRscheduler>
void optimizeGradientDescent(T &model, const std::vector<XYpair> &data, const LRscheduler &lr, int nepoch){
  std::default_random_engine gen(1234);
  std::uniform_int_distribution<int> dist(0,data.size());

  int ndata = data.size();
  
  std::vector<int> didx(ndata);
  for(int i=0;i<ndata;i++) didx[i] = i;
  
  for(int epoch=0;epoch<nepoch;epoch++){
    std::random_shuffle ( didx.begin(), didx.end(), [&](const int l){ return dist(gen); }  );
    double eps = lr(epoch);
    std::cout << "Epoch " << epoch << " learning rate " << eps << std::endl;
    
    for(int ii=0;ii<ndata;ii++){
      int i = didx[ii];
      double loss = model.loss(data[i].x, data[i].y);
      std::cout << epoch << "-" << ii << " : "<< loss << std::endl;
      model.step( model.deriv(), eps );
    }
  }

}



struct AdamParams{ //NB, alpha comes from the learning scheduler
  double beta1;
  double beta2;
  double eps;
  AdamParams( double beta1=0.99, double beta2=0.999, double eps=1e-8): beta1(beta1), beta2(beta2), eps(eps){}
};
  
template<typename T, typename LRscheduler>
void optimizeAdam(T &model, const std::vector<XYpair> &data, const LRscheduler &lr, const AdamParams &ap, int nepoch){
  std::default_random_engine gen(1234);
  std::uniform_int_distribution<int> dist(0,data.size());

  int nparam = model.nparams();
  Vector m(nparam,0.0);
  Vector v(nparam,0.0);
  int t=0;
  
  int ndata = data.size();
  
  std::vector<int> didx(ndata);
  for(int i=0;i<ndata;i++) didx[i] = i;
  
  for(int epoch=0;epoch<nepoch;epoch++){
    std::random_shuffle ( didx.begin(), didx.end(), [&](const int l){ return dist(gen); }  );
    double alpha = lr(epoch);
    std::cout << "Epoch " << epoch << " learning rate " << alpha << std::endl;
    
    for(int ii=0;ii<ndata;ii++){
      int i = didx[ii];
      double loss = model.loss(data[i].x, data[i].y);
      auto g = model.deriv();

      double delta = t>0 ? alpha * sqrt(1. - pow(ap.beta2,t))  / (1. - pow(ap.beta1,t) ) : alpha;
      for(int p=0;p<nparam;p++){
	double gp_init = g(p);
	m(p) = ap.beta1 * m(p) + (1.-ap.beta1)*g(p);
	v(p) = ap.beta2 * v(p) + (1.-ap.beta2)*pow(g(p),2);

	g(p) = m(p)/(sqrt(v(p)) + ap.eps);
	//std::cout << "p="<< p << " m=" << m(p) << " v=" << v(p) << " g:" << gp_init << "->" <<  g(p) << std::endl;
      }
      ++t;      
      std::cout << epoch << "-" << ii << " : "<< loss << " update model with step size " << delta << std::endl;
      model.step( g , delta );
    }
  }

}


class DecayScheduler{
  double eps;
  double decay_rate;
public:
  DecayScheduler(double eps, double decay_rate): eps(eps), decay_rate(decay_rate){}
  double operator()(const int epoch) const{ return eps * 1./(1. + decay_rate * epoch); }
};
