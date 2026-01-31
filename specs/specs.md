I want to create a generator for training and testing data for LLMs.
Each example will be created by chaining a list of transformations of a list of vectors and applying it on the a random vector.
Each transformation will change the list in a way that does not change the length of the list.
Examples of transformations are sorting, reversing, cumulative sum, adding a constant, etc. Also all the transformations will be modulo k.
There will be a hydra config where we can set the params such as the k.

Each example will be created first sampling the number of transformations from a range which will be given in the hydra config and then we sample a list of these transformations. This list will be sampled as a path from a graph. This graph will be directed and will several d layers where each layer will have m vertices (defined in config). Each vertex in one layer will be connected to all vertices in the layer above it. I.e vertices in the first layer will be connected to vertices in the second layer, etc. The vertices will be labeled by integers and there will be a mapping from these integers to the transformations. To sample the list we need we need to decide in which layer it will start and in which it will end. If the list should have length 2 then we need 2 vertices from two conscecutive layers. 

We need to check whether the parameters in the config make sense. We want to have 16 transformations in total so the graph cannot have more then 16 vertices.


We also want to create training and validation splits. The idea is that the training splits will contain disjunctive lists of transformations from the lists of transformations in the validation split. This means that we want to have lists of transformations that will be applied in the training set and another set that will be applied in the validation set.  The proportion should be around 80:20. The logic for sampling these lists should be easily updatable. 

The concrete example for the language model will be created by sampling a random vector then sampling a list of transformations and then applying the transformations on the vector. The LLM will be given the input and output vector and it will need to generate tokens corresponding to the application of these transformation. 

This means that for every transformation we need to be able to log its operations. We basically want to trace the steps of the transformation. I want to do it minimalistically, so one idea is to pass a list to the transformation together with the vector of integers into the function of the transformation and then appending the steps in some form into the list. The list will correspond to the trace.

From this trace we will then generate a string that will be tokenized and imitated by the LLM. We also want these traces to be easily parsable so we can parse the generated text and compute all kinds of statistics to see where the model is doing mistakes.

