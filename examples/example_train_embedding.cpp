#include "example_transformers_common.hpp"

struct Loader{
  typedef std::vector< std::vector<std::string> const* > ComparisonType;
  typedef Tensor<double,3> InputType;
  
  struct batchData{
    ComparisonType y;
    InputType x;
  };

  const std::vector<  std::vector<std::string>   > &training_data;
  const std::vector<Matrix<double> > &training_data_tens; 

  Loader(const std::vector<  std::vector<std::string>   > &training_data,
	 const std::vector<Matrix<double> > &training_data_tens): training_data(training_data), training_data_tens(training_data_tens){}
  
  size_t size() const{ return training_data.size(); }

  batchData batch(int const* indices, int batch_size) const{
    batchData out;

    int C = training_data_tens[0].size(0);
    int E = training_data_tens[0].size(1);
    
    //y
    out.y.resize(batch_size);
    for(int b=0;b<batch_size;b++)
      out.y[b] = &training_data[indices[b]];

    //x
    int size[3] = {C,E,batch_size};
    out.x = InputType(size);
    for(int b=0;b<batch_size;b++)
      out.x.pokeLastDimension(training_data_tens[indices[b]], b);

    return out;
  }
};

template<typename Model>
std::vector<std::string> predictNext(const std::vector<std::string> &sentence, int C, int B,
				     Model &model, const std::vector<std::string> &vocab,
				     const std::unordered_map<std::string, Vector<double> > &vocab_vec){
  //which token are we interested in finding the next token for?
  int tok = sentence.size(); //not size()-1 because we will be prepending <BOS>
  
  //pad
  std::vector<std::string> context = buildContext(sentence,C);
  //embed
  Matrix<double> cemb = embed(context, vocab_vec);
  //batchify
  Tensor<double,3> cembb(cemb.size(0),cemb.size(1),B);
  for(int b=0;b<B;b++)
    cembb.pokeLastDimension(cemb,b);

  //get prediction
  Tensor<double,3> pred = model.predict(cembb);

  //find max prob for token of interest
  int pmax_idx = 0;
  {
    autoView(pred_v,pred,HostRead);
    std::cout << "Token probabilities:";

    assert(pred_v.size(1) == vocab.size());
    double pmax = 0.;
    for(int p=0;p<pred_v.size(1);p++){
      std::cout << " " << vocab[p] << ":" << pred_v(tok,p,0);
      
      if(pred_v(tok,p,0) > pmax){
	pmax = pred_v(tok,p,0);
	pmax_idx = p;
      }
    }
    std::cout << std::endl;
  }
  std::vector<std::string> out(sentence);
  out.push_back(vocab[pmax_idx]);
  return out;
}
  
  
int main(int argc, char** argv){
  initialize(argc, argv);

  //Here we are going to generate an embedding for a simple vocabulary
  std::vector<std::string> vocab = { "the", "cat", "ate", "mouse", "kibble", "sat", "on", "mat", "<PAD>", "<BOS>", "<EOS>", "<UNK>" };
  
  //We start with a 1-hot encoding of each word, for which each are assigned to an orthogonal unit vector in the embedding space
  std::unordered_map<std::string, Vector<double> > _1hot_vocab_vec; //as unit vector
  std::unordered_map<std::string, int > _1hot_vocab_idx; //as index
  
  int d_model_in = vocab.size();
  
  for(int i=0;i<vocab.size();i++){
    _1hot_vocab_vec[vocab[i]] = unitVec(i,d_model_in);
    _1hot_vocab_idx[vocab[i]] = i;
  }
    
  //We want to more densely represent these words
  int d_model_out = 4;

  //We need some sentences to train it on
  //Use a simple tokenizer breaking sentences up around spaces
  std::vector<  std::vector<std::string>   > training_data;
  training_data.push_back( tokenize("the cat sat on the mat") );
  training_data.push_back( tokenize("the cat ate the mouse") );
  training_data.push_back( tokenize("the cat ate kibble") );
  
  //We need to pad everything to the same context size
  size_t C = 0;
  for(auto &s : training_data)
    C = std::max(C,s.size());
  C+=2; //always start and end sentences with BOS/EOS markers
  
  for(auto &s : training_data)
    s = buildContext(s,C);
    
  //Embed the training data into tensors  C x d_model_in with positional encoding
  std::vector<Matrix<double> > training_data_tens(training_data.size());
  for(int i=0;i<training_data.size();i++)
    training_data_tens[i] = embed(training_data[i], _1hot_vocab_vec);  

  constexpr int embedding_dim = 1;
  constexpr int context_dim = 0;    
    
  if(0){
    //Start with a decoder-only model for the 1-hot encoded data to see how it works
    int nheads = 4;
    int d_act = 40;
    int d_model = d_model_in;
    auto pos_embed = embed_positions_sinusoidal_layer(input_layer<double,Tensor<double,3> >());
    
    auto decoder1 = transformer_decoder_block(pos_embed,
					      d_model, nheads, d_act, GeLU<double>());
    auto decoder2 = transformer_decoder_block(decoder1,
					      d_model, nheads, d_act, GeLU<double>());    
    int B = 3;
  
    auto softmax_head = softmax_layer<3>( //softmax to transform the logits to probabilities
					 batch_tensor_dnn_layer<3>( //this linear layer transforms the embedding dim into logits for each token in the vocabulary
								   norm_layer<3>(decoder2, embedding_dim, d_model, true,true), //layer norm over embedding dimension
								   embedding_dim, vocab.size(), d_model, noActivation<double>()
								    ),
					 embedding_dim);

    //int sizes[3] = {C,d_model,B};
    //testDeriv(softmax_head,sizes,sizes, 1e-8);
  
    LogLossFunc cf(_1hot_vocab_idx);
    auto model_with_loss = cost_func_wrap<LogLossFunc>(softmax_head,cf); 

    Loader loader(training_data, training_data_tens);

    int nepoch = 150;
    DecayScheduler<double> lr(0.01, 0.005);
    AdamOptimizer<double, DecayScheduler<double> > opt(lr);
    std::vector<double> loss = train(model_with_loss, loader, opt, nepoch, B);

    //Note that the output probabilities accurately reflect the training data
    std::vector<std::string> sentence;
    for(int c=0;c<C-2;c++){
      sentence = predictNext(sentence, C, B, model_with_loss, vocab, _1hot_vocab_vec);
      std::cout << cat(sentence) << std::endl;
    }
  }

  {
    //Now let's see how well we can do with an embedding
    int nheads = 4;
    int d_act = 40;
    int d_model = d_model_out;

    int d_hidden = 50;
    auto embedding = batch_tensor_dnn_layer<3>( batch_tensor_dnn_layer<3>(
									  input_layer<double,Tensor<double,3> >(),
									  embedding_dim, d_hidden, d_model_in, GeLU<double>()
									  ),	  
						embedding_dim, d_model, d_hidden, GeLU<double>()
						);

    auto pos_embed = embed_positions_sinusoidal_layer(embedding);
    
    auto decoder1 = transformer_decoder_block(pos_embed,
					      d_model, nheads, d_act, GeLU<double>());
    auto decoder2 = transformer_decoder_block(decoder1,
					      d_model, nheads, d_act, GeLU<double>());    
    int B = 3;
  
    auto softmax_head = softmax_layer<3>( //softmax to transform the logits to probabilities
					 batch_tensor_dnn_layer<3>( //this linear layer transforms the embedding dim into logits for each token in the vocabulary
								   norm_layer<3>(decoder2, embedding_dim, d_model, true,true), //layer norm over embedding dimension
								   embedding_dim, vocab.size(), d_model, noActivation<double>()
								    ),
					 embedding_dim);

    //int sizes[3] = {C,d_model,B};
    //testDeriv(softmax_head,sizes,sizes, 1e-8);
  
    LogLossFunc cf(_1hot_vocab_idx);
    auto model_with_loss = cost_func_wrap<LogLossFunc>(softmax_head,cf); 

    Loader loader(training_data, training_data_tens);

    int nepoch = 500;
    DecayScheduler<double> lr(0.01, 0.005);
    AdamOptimizer<double, DecayScheduler<double> > opt(lr);
    std::vector<double> loss = train(model_with_loss, loader, opt, nepoch, B);

    //Note that the output probabilities accurately reflect the training data
    std::vector<std::string> sentence;
    for(int c=0;c<C-2;c++){
      sentence = predictNext(sentence, C, B, model_with_loss, vocab, _1hot_vocab_vec);
      std::cout << cat(sentence) << std::endl;
    }
  }

  return 0;
}






  
