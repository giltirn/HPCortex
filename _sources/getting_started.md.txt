## Getting started

In this section we will illustrate the fundamentals of building, training and using models in HPCortex via a simple example. The complete working version of this code can be found in `examples/example_dnn.cpp` which is compiled by default.

### Building a simple model

Models in HPCortex are structured as nested instances of layers, with the top-most layer being the output layer and the bottom-most the input layer. Layers are conveniently instantiated through wrapper functions that hide away some of the type-passing details and, optionally, initialize the layer's parameters.

The following simple example creates a fully-connected network with one hidden layer

	int n_out = 10;
	int n_in = 5;
	int n_hidden = 25;

	auto model = dnn_layer(n_out, n_hidden,
			       dnn_layer(n_hidden, n_in, ReLU<float>(),
					 input_layer<float>()				   
					 )
			       ); 

Note the following:

* In general, the arguments to a layer appear first within the function signature of a layer instantiation function, followed by the layer below.

* The model must always terminate on a *single* input layer. In this example, the input type is assumed to be a matrix, where the *column* index is the batch index. As this is the default input type it does not need to be explicitly specified; however, `input_layer<float, Matrix<float> >()` is equivalent to the above.

* Batching, or more correctly, *mini-batching*, is exploited at the fundamental level in HPCortex: All inputs and outputs to the model are batch-processed in parallel for more efficient computation.

* The dimensions of the internal parameter tensors must be specified at this stage for purposes of initialization, or else explicit initial values must be provided. Here we are allowing the parameters to be initialized automatically using the default [Glorot uniform random](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) (also cf. [here](https://fluxml.ai/Flux.jl/stable/reference/utilities/#Flux.glorot_uniform))

* The dimensions of the *data* (including the batch size) generally do not need to be specified at this stage; rather, the first use of the model will fix these parameters internally.

* The floating point type is specified explicitly as a parameter of the activation function (`ReLU` here) and the input layer. This type is passed through to the layers above automatically.

### Training the model

Training first requires the model to be wrapped in a loss function. Here we will use the inbuilt mean-squared error (MSE) loss function that computes the average of the element-wise squared differences between the prediction and the training data, including over the batch dimension. This loss function comes with its own convenience wrapper function:

	 auto loss = mse_cost( model );

The training process follows the standard approach of minimizing the loss by computing the gradient of the loss with respect to the model parameters, and taking steps along the (negative) gradient direction. HPCortex supports multiple **Optimizers** to perform this minimization. Here we will use the popular Adam optimizer with a specific initial learning rate:

    float learning_rate = 0.005;
    AdamOptimizer<float> optimizer(learning_rate);

We must next provide a **DataLoader**, whose job it is to provide batches of matched inputs and data. The input data type, once batched, must match the type provided to the input layer of the model specification above, while the data type is whatever is appropriate for the loss function, and does not necessarily have to be the same as the model output type provided the loss function knows how to combine the two.

Below, we generate some training and validation data in the form of a standard vector of `XYpair` containers,

	template<typename FloatType, int DimX, int DimY>
	struct XYpair{
	  Tensor<FloatType,DimX> x;
	  Tensor<FloatType,DimY> y;
	};

comprising the paired inputs and data. Data of this form has an inbuilt dataloader, `XYpairDataLoader`, associated with it.

	constexpr int DataDim = 1;
	typedef std::vector<XYpair<float,DataDim,DataDim> > XYpairVector;
	int ntrain = 200;
	int nvalid = 50;  
	XYpairVector data = generate_data(ntrain + nvalid);  
	XYpairVector train_data(data.begin(),data.begin()+ntrain);
	XYpairVector valid_data(data.begin()+ntrain,data.end());

	XYpairDataLoader<float,DataDim,DataDim> loader(train_data);

Here `generate_data()` is some (here unspecified) function that generates the data for this example. Note that `DataDim`, the dimension of the data, is one and not two, as the batch dimension is only used internally.

Finally, we can train the model using `train`:

	int nepoch = 100;
	int batch_size = 4;
	std::vector<float> loss_history = train(loss, loader, optimizer, nepoch, batch_size);

where we provided the number of epochs (complete passes over the training data set) and the batch size. The function returns the batch loss history.

### Inference

Once trained, the model can be employed for inference. In this case we will validate the model against the validation data that we separated out above.

At present, the layers in HPCortex often store the batch size internally and test that their inputs conform to this. As a result, the batch size generally cannot be changed once the model is trained. In particular, this means that if we want to perform inference for a single, unbatched input, we must first place it into a batched tensor. For the common case (as it is in our example) where the inputs and outputs are all tensors, the loss-function wrapper (`loss` above) offers a convenience function that performs this operation, `predict(value, batch_size)`, where value is an unbatched tensor. We will use this below.

	for(int i=0;i<valid_data.size();i++){
	  auto const &xy = valid_data[i];
	  Vector<float> prediction = loss.predict(xy.x, batch_size); //<--- perform inference
	  Vector<float> diff = prediction - xy.y;
	  double loss = norm2(diff) / n_out;    
	  std::cout << xy.x << " -> " << loss << std::endl;
	}   

In this example we manually compute the MSE validation loss and output it to the terminal.

Inference can also be performed by calling `loss.predict(batched_input)` for a batched input tensor, or by calling `model.value(batched_input)` on the model itself.


<!-- The derivative is computed using automatic backwards differentiation.
<!-- .. In the forward pass, each layer receives a tensor from the layer below, which is then processes and passes onto the layer above.
<!--   Likewise, in the backward (derivative) pass, each layer receives a tensor representing the derivative of the loss with respect to the *outputs* of that layer, as well as an array to populate with the derivatives of the loss