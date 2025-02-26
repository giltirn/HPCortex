#pragma once
#include <random>
#include <algorithm>
#include <Tensors.hpp>
#include <Comms.hpp>
#include <Batching.hpp>

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

  int nparam = model.nparams();

  int blocksize = communicators().batchNrank();
  int nblocks = (ndata + blocksize - 1) / blocksize; //round up
  int me = communicators().batchRank(); //all ranks in a pipeline will have the same value for the batch rank, but only the pipeline leader should communicate

  bool do_print = me == 0 && communicators().isPipelineLeader();

  if(do_print) std::cout << "Training with DDP over " << blocksize << " ranks with " << nblocks << " per epoch" << std::endl;
  
  std::vector<double> losses(nblocks*nepoch);
    
  for(int epoch=0;epoch<nepoch;epoch++){
    optimizer.epochStart(epoch);
    std::random_shuffle ( didx.begin(), didx.end(), [&](const int l){ return dist(gen); }  );

    for(int block=0;block<nblocks;block++){
      int blocksize_actual = std::min(ndata - block*blocksize, blocksize);

      double loss = 0;
      Vector deriv(nparam, 0.);
      
      if(me < blocksize_actual){ //if not enough data to have all ranks do work in this block
	int ii = block*blocksize + me;
	int i = didx[ii];
	loss = model.loss(data[i].x, data[i].y);
	deriv = model.deriv();
      }
      batchAverage(&loss,1,false); //no need to bcast the loss to the pipeline ranks
      batchAverage(deriv.data(),deriv.data_len(),true); //share the deriv over all pipeline ranks
           
      if(do_print) std::cout << epoch << "-" << block << " : "<< loss << std::endl;
      
      double eps;
      Vector direction = optimizer.descentProfile(eps,deriv);
      
      model.step( direction, eps );

      losses[block+nblocks*epoch] = loss;
    }
  }
  return losses;
}


