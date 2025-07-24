#include "example_transformers_common.hpp"

struct Loader{
  typedef std::vector< std::vector<std::string> const* > ComparisonType; //French sentence we are comparing to
  typedef std::pair<Tensor<double,3>,Tensor<double,3> > InputType; //embedded English and French translations
  
  struct batchData{
    ComparisonType y;
    InputType x;
  };

  const std::vector<  std::vector<std::string>   > &training_data_french;
  const std::vector<Matrix<double> > &training_data_tens_english;
  const std::vector<Matrix<double> > &training_data_tens_french;

  Loader(const std::vector<  std::vector<std::string>   > &training_data_french,
	 const std::vector<Matrix<double> > &training_data_tens_english,
	 const std::vector<Matrix<double> > &training_data_tens_french): training_data_french(training_data_french), training_data_tens_english(training_data_tens_english), training_data_tens_french(training_data_tens_french){}
  
  size_t size() const{ return training_data_french.size(); }

  batchData batch(int const* indices, int batch_size) const{
    batchData out;

    int C = training_data_tens_english[0].size(0);
    int E_eng = training_data_tens_english[0].size(1);
    int E_fr = training_data_tens_french[0].size(1);
    
    //y
    out.y.resize(batch_size);
    for(int b=0;b<batch_size;b++)
      out.y[b] = &training_data_french[indices[b]];

    //x
    int size_eng[3] = {C,E_eng,batch_size};
    int size_fr[3] = {C,E_fr,batch_size};

    out.x.first = Tensor<double,3>(size_eng);
    out.x.second = Tensor<double,3>(size_fr);
    
    for(int b=0;b<batch_size;b++){
      out.x.first.pokeLastDimension(training_data_tens_english[indices[b]], b);
      out.x.second.pokeLastDimension(training_data_tens_french[indices[b]], b);
    }
    return out;
  }
};

void addTrainingDataAndBuildVocab(std::vector< std::vector<std::string> > &training_data,
				  std::unordered_map< std::string, int >  &vocab, //vocab and index of token
				  const std::string &sentence){
  std::vector<std::string> toks = tokenize(sentence);
  for(auto const &tok : toks){
    auto vit = vocab.find(tok);
    if(vit == vocab.end()){
      int idx = vocab.size();
      vocab[tok] = idx;
    }
  }
  training_data.push_back(toks);
}


template<typename Model>
std::vector<std::string> predictNext(const std::vector<std::string> &english_sentence,
				     const std::vector<std::string> &partial_french_sentence,
				     int C, int B,
				     Model &model, const std::vector<std::string> &french_vocab,
				     const std::unordered_map<std::string, Vector<double> > &english_vocab_vec,
				     const std::unordered_map<std::string, Vector<double> > &french_vocab_vec
				     ){
  //which token are we interested in finding the next token for?
  int tok = partial_french_sentence.size(); 
  
  //pad
  std::vector<std::string> context_eng = english_sentence; ///assume the english sentence has been padded already
  std::vector<std::string> context_fr = buildContext(partial_french_sentence,C, true); //shift right
  std::cout << "French input: " << cat(context_fr) << std::endl;
  
  //embed
  Matrix<double> cemb_eng = embed(context_eng, english_vocab_vec);
  Matrix<double> cemb_fr = embed(context_fr, french_vocab_vec);
  
  //batchify
  Tensor<double,3> cembb_eng(cemb_eng.size(0),cemb_eng.size(1),B);
  Tensor<double,3> cembb_fr(cemb_fr.size(0),cemb_fr.size(1),B);
  for(int b=0;b<B;b++){
    cembb_eng.pokeLastDimension(cemb_eng,b);
    cembb_fr.pokeLastDimension(cemb_fr,b);
  }
  std::pair<Tensor<double,3>, Tensor<double,3> > in;
  in.first = std::move(cembb_eng);
  in.second = std::move(cembb_fr);
    
  //get prediction
  Tensor<double,3> pred = model.predict(in);

  //find max prob for token of interest
  int pmax_idx = 0;
  {
    autoView(pred_v,pred,HostRead);

    assert(pred_v.size(1) == french_vocab.size());
    double pmax = 0.;
    for(int t=0;t<C-1;t++){
      std::cout << (t == tok ? ">>>" : "") << "Next French token of token " << t << " (" << context_fr[t] << ") with probabilities:";
    
      for(int p=0;p<pred_v.size(1);p++){
	std::cout << " " << french_vocab[p] << ":" << pred_v(t,p,0) ;
      
	if(t==tok && pred_v(tok,p,0) > pmax){
	  pmax = pred_v(tok,p,0);
	  pmax_idx = p;
	}
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
  std::vector<std::string> out(partial_french_sentence);
  out.push_back(french_vocab[pmax_idx]);
  return out;
}
  

int main(int argc, char** argv){
  initialize(argc, argv);

  std::mt19937 rng(1234);
  typedef confDouble Config;
  typedef Tensor<double,3> TensorType;
  constexpr int embedding_dim = 1;
  typedef std::pair<TensorType,TensorType> InputType; //paired encoder and decoder inputs

  std::vector< std::vector<std::string> > training_data_english, training_data_french;
  std::unordered_map< std::string, int > vocab_english, vocab_french;
  addTrainingDataAndBuildVocab(training_data_english, vocab_english, "i love cats");
  addTrainingDataAndBuildVocab(training_data_french, vocab_french, "j' adore les chats");
  
  addTrainingDataAndBuildVocab(training_data_english, vocab_english, "i love chickens");
  addTrainingDataAndBuildVocab(training_data_french, vocab_french, "j' adore les poules");

  addTrainingDataAndBuildVocab(training_data_english, vocab_english, "i hate mice");
  addTrainingDataAndBuildVocab(training_data_french, vocab_french, "je déteste les souris");

  addTrainingDataAndBuildVocab(training_data_english, vocab_english, "i don't understand");
  addTrainingDataAndBuildVocab(training_data_french, vocab_french, "je ne comprends pas");

  //Add the extra markers
  std::vector<std::string> extra_markers = { "<EOS>", "<PAD>" };
  for(auto &e : extra_markers){
    vocab_english[e] = vocab_english.size();
    vocab_french[e] = vocab_french.size();
  }
  vocab_french["<BOS>"] = vocab_french.size(); //encoder sentences don't need to be "shifted right"
  
  int d_vocab_english = vocab_english.size();
  int d_vocab_french = vocab_french.size();
  
  //Build mappings of the vocabulary to 1hot embedded vectors
  std::unordered_map< std::string, Vector<double> > vocab_english_1hot, vocab_french_1hot;
  for(auto const &t : vocab_english)  vocab_english_1hot[t.first] = unitVec(t.second, d_vocab_english);
  for(auto const &t : vocab_french)  vocab_french_1hot[t.first] = unitVec(t.second, d_vocab_french);
	  
  //We need to pad all the training data to a common context size
  size_t C = 0;
  for(auto &s : training_data_french)
    C = std::max(C,s.size()+2); //always start and end decoder sentences with BOS/EOS markers
  for(auto &s : training_data_english)
    C = std::max(C,s.size()+1); //always end encoder sentences with EOS marker


  //Add the markers to the tokenized training data  
  for(auto &s : training_data_french)
    s = buildContext(s,C,true); //"shift right" (add <BOS>)
  for(auto &s : training_data_english)
    s = buildContext(s,C,false); //don't shift right

  //Now convert the training data into matrices of size C x d_vocab_*  using our 1-hot embeddings
  int ntraining_data = training_data_english.size();
  
  std::vector<Matrix<double> > training_data_tens_english(ntraining_data), training_data_tens_french(ntraining_data);
  for(int i=0;i<ntraining_data;i++){
    training_data_tens_english[i] = embed(training_data_english[i], vocab_english_1hot);
    training_data_tens_french[i] = embed(training_data_french[i], vocab_french_1hot);  
  }    
  
  ///////////////////////////////////////Build the model

  int nheads = 3;
  int d_act = 40; //neurons in activation layer
  
  //split the input into two subchains
  auto splt = pair_split_layer(input_layer<Config,InputType>()); 
  
  //for simplicity, as in https://arxiv.org/pdf/1706.03762, we will use a consistent embedding size. This means we need initial learned embedding layers to embed the differing vocabularies
  int d_model = 6;
  int d_hidden = 50;
  auto embed_enc = batch_tensor_dnn_layer<3>(embedding_dim, d_model, d_hidden, GeLU<double>(),
					     batch_tensor_dnn_layer<3>(embedding_dim, d_hidden, d_vocab_english, GeLU<double>(), *splt.first)
					     );

  auto embed_dec = batch_tensor_dnn_layer<3>(embedding_dim, d_model, d_hidden, GeLU<double>(),
					     batch_tensor_dnn_layer<3>(embedding_dim, d_hidden, d_vocab_french, GeLU<double>(), *splt.second)					     
					     );  
  
  //positional embedding for encoder and decoder
  auto pos_embed_enc = embed_positions_sinusoidal_layer(embed_enc); 
  auto pos_embed_dec = embed_positions_sinusoidal_layer(embed_dec);

  //encoder
  auto encoder = norm_layer<3>(embedding_dim, d_model, true, true,
			       transformer_encoder_block(d_model, nheads, d_act, GeLU<double>(), pos_embed_enc)
			       );

  //cross decoder
  auto decoder = transformer_cross_decoder_block(d_model, nheads, d_act, GeLU<double>(), encoder, pos_embed_dec);

  //softmax to transform the logits to probabilities
  auto softmax_head = softmax_layer<3>(embedding_dim, 
				       batch_tensor_dnn_layer<3>(embedding_dim, d_vocab_french, d_model, noActivation<double>(), //this linear layer transforms the embedding dim into logits for each token in the French vocabulary
								 norm_layer<3>(embedding_dim, d_model, true,true, decoder) //layer norm over embedding dimension								   
								 )
				       );

  ///////////////////////////////////// Train model
  
  //We'll use the log-loss to maximize the next token probability for each token on the French side    
  LogLossFunc cf(vocab_french); 
  auto model_with_loss = cost_func_wrap<LogLossFunc>(softmax_head,cf); 
  
  Loader loader(training_data_french,training_data_tens_english,training_data_tens_french);

  int B = 4;
  int nepoch = 170;
  DecayScheduler<double> lr(0.02, 0.003);
  AdamOptimizer<double, DecayScheduler<double> > opt(lr);
  std::vector<double> loss = train(model_with_loss, loader, opt, nepoch, B);

  ////////////////////////////////////// Employ model
  std::vector<std::string> vocab_french_v(vocab_french.size());
  for(auto const &t : vocab_french)
    vocab_french_v[t.second] = t.first;
    
  for(int es = 0; es < ntraining_data; es++){
    const std::vector<std::string> &english_sentence = training_data_english[es];
    std::cout << "Translating: " << cat(english_sentence) << std::endl;
    
    std::vector<std::string> french_sentence;
    for(int c=0;c<C-2;c++){
      french_sentence = predictNext(english_sentence, french_sentence, C, B, model_with_loss,  vocab_french_v, vocab_english_1hot, vocab_french_1hot);
      std::cout << cat(french_sentence) << std::endl;
    }

    std::cout << std::endl << std::endl;
  }
      
  return 0;
}
