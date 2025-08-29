struct train_stage_time_report{
  Timer total;
  Timer loader;
  Timer model;
  Timer reduce;

  train_stage_time_report(): total(true){}

  std::string report(double FLOPS){
    double Gflops = FLOPS/1e9 /model.time();
    std::ostringstream os;
    double other = total.time() - loader.time() - model.time() - reduce.time();
    os << "loader: " << loader.time() << "s, "
       << "model: " << model.time() << "s (" << Gflops << " Gflops), "
       << "reduction: " << reduce.time() << "s, "
       << "other: " << other << "s | "
       << "total: " << total.time();
    return os.str();
  }
};



template<typename DataLoader, typename LossWrappedModelType, typename Optimizer>
std::pair<
std::vector<typename LossWrappedModelType::FloatType>,
std::vector<typename LossWrappedModelType::FloatType> >
train(LossWrappedModelType &loss_func, const DataLoader &train_data, DataLoader const *valid_data, Optimizer &optimizer, int nepoch, int batch_size, bool suppress_logging = false){
  typedef typename LossWrappedModelType::FloatType FloatType;
  std::default_random_engine gen(1234); //important that every rank shuffles in the same way

  //We want to divide the data evenly over batches. This means we may need to discard some data
  if(batch_size > train_data.size())
    batch_size = train_data.size();
  if(valid_data && batch_size > valid_data->size())
    batch_size = valid_data->size();
    
  int nbatch_train = train_data.size() / batch_size;
  int ndata_train = nbatch_train * batch_size;

  int nbatch_valid = valid_data ? valid_data->size() / batch_size  : 0;
  int ndata_valid = nbatch_valid * batch_size;
  
  std::uniform_int_distribution<int> dist(0,ndata_train-1); //sampler for shuffling training data  
  std::vector<int> didx_train(ndata_train); //training data indices
  for(int i=0;i<ndata_train;i++) didx_train[i] = i;

  std::vector<int> didx_valid(ndata_valid);
  for(int i=0;i<ndata_valid;i++) didx_valid[i] = i;
  
  int nparam = loss_func.nparams();

  //For DDP we solve blocks of batches in parallel
  int ddp_blocksize = communicators().ddpNrank();
  int nblocks_ddp_train = (nbatch_train + ddp_blocksize - 1) / ddp_blocksize; //the number of blocks computed by each rank. Round up to ensure completion
  int nblocks_ddp_valid = (nbatch_valid + ddp_blocksize - 1) / ddp_blocksize;
  
  int me = communicators().ddpRank(); //all ranks in a pipeline will have the same value for the ddp rank, but only the pipeline leader should communicate
  bool do_print = me == 0 && communicators().isPipelineLeader() && !suppress_logging ;

  if(do_print){
    if(!valid_data)
      std::cout << "Training with " << ndata_train << " data samples divided into " << nbatch_train << " batches of size " << batch_size
		<< " using DDP over " << ddp_blocksize << " ranks with " << nblocks_ddp_train << " iterations per epoch" << std::endl;
    else
      std::cout << "Training with " << ndata_train << " training and " << ndata_valid << " validation data samples divided into, respectively, " << nbatch_train << " and " << nbatch_valid << " batches of size " << batch_size
		<< " using DDP over " << ddp_blocksize << " ranks with " << nblocks_ddp_train << " iterations per epoch" << std::endl;
  }
  
  std::vector<FloatType> losses_train(nblocks_ddp_train*nepoch), losses_valid(nblocks_ddp_valid*nepoch);
    
  for(int epoch=0;epoch<nepoch;epoch++){
    //////////// train epoch ///////////////
    optimizer.epochStart(epoch, do_print);
    std::random_shuffle ( didx_train.begin(), didx_train.end(), [&](const int l){ return dist(gen); }  ); //shuffle training data indices
    FloatType lmax_train=std::numeric_limits<FloatType>::lowest(),  lmin_train = std::numeric_limits<FloatType>::max(),  lavg_train = 0.;    
    train_stage_time_report t_train;
    
    for(int block=0;block<nblocks_ddp_train;block++){
      int ddp_blocksize_actual = std::min(nbatch_train - block*ddp_blocksize, ddp_blocksize);

      FloatType loss = 0;
      Vector<FloatType> deriv(nparam, 0.);
      
      if(me < ddp_blocksize_actual){ //if not enough data to have all ranks do work in this block
	int bidx = block*ddp_blocksize + me; //which batch are we doing?

	//Get the batch
	TIME(t_train.loader,
	     auto bxy = train_data.batch(didx_train.data() + bidx*batch_size, batch_size);
	     );

	TIME(t_train.model,
	loss = loss_func.loss(bxy.x, bxy.y, DerivYes);
	deriv = loss_func.deriv();
	     );
      }

      TIME(t_train.reduce, 
      ddpAverage(&loss,1,false); //no need to bcast the loss to the pipeline ranks
      ddpAverage(deriv,true); //share the deriv over all pipeline ranks
	   )
	   
      //if(do_print) std::cout << epoch << "-" << block << " : "<< loss << std::endl;
      lmax_train = std::max(lmax_train,loss);
      lmin_train = std::min(lmin_train,loss);
      lavg_train += loss;
      
      FloatType eps;
      Vector<FloatType> direction = optimizer.descentProfile(eps,deriv);
      
      loss_func.step( direction, eps );

      losses_train[block+nblocks_ddp_train*epoch] = loss;
    }
    lavg_train /= nblocks_ddp_train;

    t_train.total.pause();
    double train_FLOPS = nbatch_train * double(loss_func.FLOPS(0) + loss_func.FLOPS(1));
    
    //////////// end train epoch ///////////////
    
    //////////// validate epoch ///////////////
    if(valid_data){
      FloatType lmax_valid=std::numeric_limits<FloatType>::lowest(),  lmin_valid = std::numeric_limits<FloatType>::max(),  lavg_valid = 0.;
      train_stage_time_report t_valid;
      
      for(int block=0;block<nblocks_ddp_valid;block++){
	int ddp_blocksize_actual = std::min(nbatch_valid - block*ddp_blocksize, ddp_blocksize);

	FloatType loss = 0;
	if(me < ddp_blocksize_actual){ 
	  int bidx = block*ddp_blocksize + me;
	  TIME(t_valid.loader,
	  auto bxy = valid_data->batch(didx_valid.data() + bidx*batch_size, batch_size); //no need to shuffle
	       );
	  TIME(t_valid.model,
	  loss = loss_func.loss(bxy.x, bxy.y, DerivNo);
	       );
	}

	TIME(t_valid.reduce,
	ddpAverage(&loss,1,false);
	     );
	     
	lmax_valid = std::max(lmax_valid,loss);
	lmin_valid = std::min(lmin_valid,loss);
	lavg_valid += loss;
      
	losses_valid[block+nblocks_ddp_valid*epoch] = loss;
      }      
      lavg_valid /= nblocks_ddp_valid;
      t_valid.total.pause();
      
      double valid_FLOPS = nbatch_valid * double(loss_func.FLOPS(0));
      
      //////////// end validate epoch ///////////////
      
      if(do_print) std::cout << "Epoch : " << epoch << std::endl
			     << "training loss min: " << lmin_train << " avg: " << lavg_train << " max: " << lmax_train << std::endl
			     << "validation loss min: " << lmin_valid << " avg: " << lavg_valid << " max: " << lmax_valid << std::endl
			     << "training timings: " << t_train.report(train_FLOPS) << std::endl
			     << "validation timings: " << t_valid.report(valid_FLOPS) << std::endl;

      
    }else{ //if not validating, just print info on the training losses
      if(do_print) std::cout << "Epoch : " << epoch << std::endl
			     << "loss min: " << lmin_train << " avg: " << lavg_train << " max: " << lmax_train << std::endl
			     << "timings: " << t_train.report(train_FLOPS) << std::endl;
    }
  }//epoch
      
  return std::make_pair(losses_train, losses_valid);
}

template<typename DataLoader, typename LossWrappedModelType, typename Optimizer>
std::vector<typename LossWrappedModelType::FloatType> train(LossWrappedModelType &loss_func, const DataLoader &data, Optimizer &optimizer, int nepoch, int batch_size, bool suppress_logging){
  return train(loss_func, data, (DataLoader const *)nullptr, optimizer, nepoch, batch_size, suppress_logging).first;
}

template<typename DataLoader, typename LossWrappedModelType, typename Optimizer>
std::pair<std::vector<typename LossWrappedModelType::FloatType>, std::vector<typename LossWrappedModelType::FloatType> >
train(LossWrappedModelType &loss_func, const DataLoader &train_data, const DataLoader &valid_data, Optimizer &optimizer, int nepoch, int batch_size, bool suppress_logging){
  return train(loss_func, train_data, &valid_data, optimizer, nepoch, batch_size, suppress_logging);
}
