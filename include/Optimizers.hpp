#pragma once
#include <random>
#include <algorithm>
#include <Tensors.hpp>
#include <Comms.hpp>
#include <DDP.hpp>

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
  Vector x;
  Vector y;
};
struct batchedXYpair{
  Matrix x;
  Matrix y;
};


batchedXYpair batchData(int* indices, int batch_size, const std::vector<XYpair> &data){
  assert(data.size()>0);
  int x_features = data[0].x.size(0);
  int y_features = data[0].y.size(0);
  
  batchedXYpair out;
  out.x = Matrix(x_features, batch_size);
  out.y = Matrix(y_features, batch_size);

  for(int b=0;b<batch_size;b++){
    int i = indices[b];
    out.x.pokeColumn(b, data[i].x);
    out.y.pokeColumn(b, data[i].y);
  }
  return out;
}


template<typename ModelType, typename Optimizer>
std::vector<double> train(ModelType &model, const std::vector<XYpair> &data, Optimizer &optimizer, int nepoch, int batch_size){
  std::default_random_engine gen(1234); //important that every rank shuffles in the same way

  //We want to divide the data evenly over batches. This means we may need to discard some data
  int nbatch = data.size() / batch_size;
  int ndata = nbatch * batch_size;
  std::uniform_int_distribution<int> dist(0,ndata-1);
  
  std::vector<int> didx(ndata);
  for(int i=0;i<ndata;i++) didx[i] = i;

  int nparam = model.nparams();

  //For DDP we solve blocks of batches in parallel
  int blocksize = communicators().ddpNrank();
  int nblocks = (nbatch + blocksize - 1) / blocksize; //round up
  int me = communicators().ddpRank(); //all ranks in a pipeline will have the same value for the ddp rank, but only the pipeline leader should communicate

  bool do_print = me == 0 && communicators().isPipelineLeader();

  if(do_print) std::cout << "Training with " << ndata << " data samples divided into " << nbatch << " batches of size " << batch_size
			 << " using DDP over " << blocksize << " ranks with " << nblocks << " iterations per epoch" << std::endl;
  
  std::vector<double> losses(nblocks*nepoch);
    
  for(int epoch=0;epoch<nepoch;epoch++){
    optimizer.epochStart(epoch);
    std::random_shuffle ( didx.begin(), didx.end(), [&](const int l){ return dist(gen); }  );

    for(int block=0;block<nblocks;block++){
      int blocksize_actual = std::min(nbatch - block*blocksize, blocksize);

      double loss = 0;
      Vector deriv(nparam, 0.);
      
      if(me < blocksize_actual){ //if not enough data to have all ranks do work in this block
	int bidx = block*blocksize + me; //which batch are we doing?

	//Get the batch
	batchedXYpair bxy = batchData(didx.data() + bidx*batch_size, batch_size, data);
	
	loss = model.loss(bxy.x, bxy.y);
	deriv = model.deriv();
      }
      ddpAverage(&loss,1,false); //no need to bcast the loss to the pipeline ranks
      ddpAverage(deriv.data(),deriv.data_len(),true); //share the deriv over all pipeline ranks
           
      if(do_print) std::cout << epoch << "-" << block << " : "<< loss << std::endl;
      
      double eps;
      Vector direction = optimizer.descentProfile(eps,deriv);
      
      model.step( direction, eps );

      losses[block+nblocks*epoch] = loss;
    }
  }
  return losses;
}


