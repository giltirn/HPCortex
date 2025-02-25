#pragma once
#include <random>
#include <algorithm>
#include <Tensors.hpp>

template<typename LRscheduler>
class GradientDescentOptimizer{
  const LRscheduler &sched;
  double eps;  
public:
  GradientDescentOptimizer(const LRscheduler &sched): sched(sched), eps(0.){}
  
  void epochStart(int epoch){
    eps = sched(epoch);
    std::cout << "GradientDescentOptimizer: Epoch " << epoch << " learning rate " << eps << std::endl;
  }
  inline Vector descentProfile(double &step_size, const Vector &deriv) const{
    step_size = eps;
    return deriv;
  }

};


struct AdamParams{ //NB, alpha comes from the learning scheduler
  double beta1;
  double beta2;
  double eps;
  AdamParams( double beta1=0.99, double beta2=0.999, double eps=1e-8): beta1(beta1), beta2(beta2), eps(eps){}
};

template<typename LRscheduler>
class AdamOptimizer{
  const LRscheduler &sched;
  AdamParams ap;
  double alpha;
  size_t t;

  Vector m;
  Vector v;

  void reset(){
    t=0;
    m=Vector();
    v=Vector();
  }
public:
  AdamOptimizer(const AdamParams &ap, const LRscheduler &sched): sched(sched), ap(ap), alpha(0.), t(0){}
  
  void epochStart(int epoch){
    alpha = sched(epoch);
    if(epoch == 0) reset();
    std::cout << "AdamOptimizer: Epoch " << epoch << " learning rate (alpha) " << alpha << std::endl;
  }
  inline Vector descentProfile(double &step_size, const Vector &g){
    int nparam = g.size(0);
    if(t==0){
      m = Vector(nparam,0);
      v = Vector(nparam,0);
    }
    Vector out(nparam);
    
    for(int p=0;p<nparam;p++){
      double gp_init = g(p);
      m(p) = ap.beta1 * m(p) + (1.-ap.beta1)*g(p);
      v(p) = ap.beta2 * v(p) + (1.-ap.beta2)*pow(g(p),2);

      out(p) = m(p)/(sqrt(v(p)) + ap.eps);
    }

    step_size =  t>0 ? alpha * sqrt(1. - pow(ap.beta2,t))  / (1. - pow(ap.beta1,t) ) : alpha;
    ++t;
    return out;
  }
};


class DecayScheduler{
  double eps;
  double decay_rate;
public:
  DecayScheduler(double eps, double decay_rate): eps(eps), decay_rate(decay_rate){}
  double operator()(const int epoch) const{ return eps * 1./(1. + decay_rate * epoch); }
};

struct XYpair{
  Matrix x;
  Matrix y;
};
template<typename ModelType, typename Optimizer>
std::vector<double> train(ModelType &model, const std::vector<XYpair> &data, Optimizer &optimizer, int nepoch){
  std::default_random_engine gen(1234); //important that every rank shuffles in the same way
  std::uniform_int_distribution<int> dist(0,data.size()-1);

  int ndata = data.size();
  
  std::vector<int> didx(ndata);
  for(int i=0;i<ndata;i++) didx[i] = i;

  std::vector<double> losses(ndata*nepoch);
  
  for(int epoch=0;epoch<nepoch;epoch++){
    optimizer.epochStart(epoch);
    std::random_shuffle ( didx.begin(), didx.end(), [&](const int l){ return dist(gen); }  );

    for(int ii=0;ii<ndata;ii++){
      int i = didx[ii];
      double loss = model.loss(data[i].x, data[i].y);
      std::cout << epoch << "-" << ii << " : "<< loss << std::endl;

      double eps;
      Vector direction = optimizer.descentProfile(eps, model.deriv());
      
      model.step( direction, eps );

      losses[ii+ndata*epoch] = loss;
    }
  }
  return losses;
}


