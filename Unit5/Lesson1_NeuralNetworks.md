# Lesson OutCome
- Understand what are Neural Network and when to use them

## General idea of Neural Networks:

Artificial Neural Networks (ANN) are inspired by biological neural networks. ANNs are set of interconnected nodes (see the Figure) and they are trained by processing examples (set of input and target data). 

No matter how well we describe neural networks, we'll never do as well as YouTuber 3Blue1Brown. We'll go into more detail about coding and choosing layers and what have you, but the following videos are a master-class in making complicated things make sense:

1. [But what is a Neural Network?](https://www.youtube.com/watch?v=aircAruvnKk)
2. [Gradient descent, how neural networks learn.](https://www.youtube.com/watch?v=IHZwWFHWa-w)
3. [(Optional) What is backpropagation really doing?](https://www.youtube.com/watch?v=Ilg3gGewQ5U)



ANNs are made of a set of inputs (an input layer) which are connected to set of nodes (second layer). The connections of biological neurons are modeled as weights. All inputs are modified by the weights and summed (which is the linear combination of input vectors and elements of a weight matrix, see the Figure). Then an activation function implements to control the output. The proces continues through the network (moving along different layers) until getting to the output layer. 
The training process in an ANN is based on determining the difference between the output of an ANN (prediction) and the true targets. The goal of training is to minimize the difference (error). The ANN will adjust its weights (according to a learning rule) to minimize the error. This process (known as iteration) continues (adjusting the weights and see if the erros is becoming smaller) until it hits a certain criteria.

Elements of an ANN are: 

- **x**: a vector of input values/features.
- **y**: output of ANN (target values).
- Loss Function (C): Typically a loss function (also knows as cost function) is used for parameter estimation. It is some function of difference between the predicted and target values. Squared error loss (squared of difference between the target and predicted values) is an example of a loss function for regression tasks.  
- The number of layers (L): If an ANN has more than **one** hidden layer, it is a deep neural network (DNN). DNNs can have multiple layers; number of layers should be deicded/tuned by the user.
- Weights between layers (W): The weight matrix between two leyers.
- Activation functions: non linear functions that are used to decide if firing/activation happens. See details in the next section.



# Activation Functions
In ANNs, inputs of each layer are multiplied by the weights and summed (this is just calculating the weighted sum). To decide if the neuron will fire/ activated or not, activation functions are used. There are different activation functions available to choose one of which is the sigmoid function (which already is discussed in the logistic regression lesson). Thus, the output of the weighted sum becomes the input of the sigmoid function. As the sigmoid function has a range between 0 and 1 (while the domain of the function can be all real numbers from - inf to + inf) the outout of this function can be used as a marker of activating a node or not. 
There are other forms of activation functions which can be used. The link below is an easy explanation of activation functions: 

https://medium.com/the-theory-of-everything/understanding-activation-functions-in-neural-networks-9491262884e0

# Backpropagation

Backpropagation is a method in which the gradient of loss function with respect to weights (assigned to the ANN) is calculated. 


 
## Hyperparameters
In an ANN there might be several parameters that are needed to be tuned, here we discuss few of them: 

- Learning rate: In the learning process, learning rate defines the size of the steps from one iteration to the next one. The learning rate should be selected such that it is not too large to skip the optimum point(s) and not too small to make the process too slow. 

- Batch size: In each iteration instead of taking all the training samples into account (in ANNs mostly we have very large data sets) we can take a fraction of them. This can be make the process of training much faster.  

- The number of hidden layers: If an ANN has more than **one** hidden layer, it is a deep neural network (DNN). DNNs can have multiple layers; number of layers should be decided/tuned by the user.  


<img src="figs/NN_picture.png" width=520px>
<sup> source: Vieira, Sandra & Pinaya, Walter & Mechelli, Andrea. (2017). Using deep learning to investigate the neuroimaging correlates of psychiatric and neurological disorders: Methods and applications. Neuroscience & Biobehavioral Reviews. 74. 10.1016/j.neubiorev.2017.01.002. </sup> 


# Summary
- In this lesson we briefly described how neural networks work, and we discussed their general architecture.
- We also put links to few amazing videos for you to watch and learn more about neural networks!
