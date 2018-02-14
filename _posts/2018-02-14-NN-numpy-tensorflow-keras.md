---
layout: post
title: "Evolving my NN model from pure numpy to tensorflow to keras"
date: 2018-02-14
---

In my previous post I've shared my Jupyter notebook with an attempt to predict the survival of Titanic passengers based on
the [Kaggle dataset](https://www.kaggle.com/c/titanic) for beginners.

The whole thing can be split into essentially two parts: data preparation and modeling the predictions.
The first part (data preparation) - included some data exploration, filling missing values, dummy encoding categorical values, and normalization.
That's the part that more or less stayed the same. The other part, however, has evolved as I've learned some new tricks.

## Pure Numpy
In my original notebook, I've tried to predict the categories (survived/not) by building a L-layer neural network with
(L-1) layers with relu activation, and the last layer with sigmoid activation. And I did it in pure numpy. Hardcore stuff.

This meant I needed to do a bunch of actions myself.

#### Initialize parameters for the model

```python
def initialize_parameters_deep(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in the network

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
    """

    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters
```

#### Implement the linear part of an epoch

```python
def linear_forward(A, W, b):
    """
    The linear part of a layer's forward propagation.
    Arguments:
    A -- activations from previous layer (or input data)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter
    cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass
    """
    Z = np.add(np.matmul(W, A), b)
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    return Z, cache
```

#### As well as linear activation forward

```python
def linear_activation_forward(A_prev, W, b, activation):
    """
    Forward propagation for the LINEAR->ACTIVATION layer
    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value
    cache -- a python dictionary containing "linear_cache" and "activation_cache"; for computing the backward pass
    """
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)
    return A, cache
```

#### Combine the two into a linera forward pass

```python
def L_model_forward(X, parameters):
    """
    Forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()

    Returns:
    AL -- last post-activation value
    caches -- list of caches containing: every cache of linear_relu_forward() and the cache of linear_sigmoid_forward()
    """
    caches = []
    A = X
    L = len(parameters) // 2

    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation = "relu")
        caches.append(cache)

    AL, cache = linear_activation_forward(A, parameters["W"+str(L)], parameters["b"+str(L)], "sigmoid")
    caches.append(cache)

    assert(AL.shape == (1,X.shape[1]))

    return AL, caches
```

#### Compute the cost

```python
def compute_cost(AL, Y):
    """
    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)
    Returns:
    cost -- cross-entropy cost
    """
    m = Y.shape[1]
    cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))

    cost = np.squeeze(cost)
    assert(cost.shape == ())
    return cost
```

#### Implement backward propagation

Have you ever calculated the derivatives? ;)

```python
def linear_backward(dZ, cache):
    """
    The linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """

    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1.0/m) * np.matmul(dZ, A_prev.T)
    db = (1.0/m) * np.sum(dZ, axis=-1, keepdims=True)
    dA_prev = np.matmul(np.transpose(W), dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db
```

#### Implement linear activation backward

```python
def linear_activation_backward(dA, cache, activation):
    """
    The backward propagation for the LINEAR->ACTIVATION layer.

    Arguments:
    dA -- post-activation gradient for current layer l
    cache -- tuple of values (linear_cache, activation_cache)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db
```

#### Combine the two into the back propagation pass

```python
def L_model_backward(AL, Y, caches):
    """
    The backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing: every cache of linear_activation_forward() with "relu" and the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])

    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ...
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ...
    """
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL

    # Initializing the backpropagation
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
    current_cache = caches[L-1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation = 'sigmoid')


    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients.

        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA"+str(l+2)],current_cache,"relu")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads
```

#### Update parameters for the model

```python
def update_parameters(parameters, grads, learning_rate):
    """
    Updates parameters using gradient descent
    Arguments:
    parameters -- python dictionary containing parameters
    grads -- python dictionary containing gradients, output of L_model_backward

    Returns:
    parameters -- python dictionary containing updated parameters
    """

    L = len(parameters) // 2

    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]

    return parameters
```

#### Build a model
Finally, all the above functions get together in the model.

```python
def L_layer_model(X, Y, layers_dims, learning_rate = 0.004, num_iterations = 3000, print_cost=False):
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

    Arguments:
    X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps

    Returns:
    parameters -- parameters learnt by the model.
    """

    costs = []                         # keep track of cost
    # Parameters initialization.
    parameters = initialize_parameters_deep(layers_dims)

    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward(X, parameters)

        # Cost function
        cost = compute_cost(AL, Y)

        # Backward propagation.
        grads = L_model_backward(AL, Y, caches)

        # Update parameters. (without optimizations)
        parameters = update_parameters(parameters, grads, learning_rate)


        # Print the cost every 1000 training example
        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 1000 == 0:
            costs.append(cost)

    return parameters
```

There it was, my glorious first neural network that could have as many layers as I wanted to (even though that would've been only relu).

**Conclusion**:
* **Pros**: Building a neural network "from scratch" gives an understanding of how things actually work, allows to see the vectorization and all the math in action.
* **Cons**: Well, smarter people build all the tools one need into great libraries which are faster, calculate the backward propagation for you, and there are probably (ok, definitely) less bugs there.


## Same, but with tensorflow

So then I tried to make the same model with tensorflow, however again somewhat semi-manual stuff.
One step I didn't have in numpy, which is specific for tensorflow: creating placeholders.

#### Creating placeholders

```python
def create_placeholders(n_x, n_y=1):
    """
    Creates the placeholders for the tensorflow session.
    Arguments:
    n_x -- number of features

    Returns:
    X -- placeholder for the data input
    Y -- placeholder for the input labels
    """

    X = tf.placeholder(tf.float32, [n_x, None])
    Y = tf.placeholder(tf.float32, [n_y, None])

    return X, Y
```

#### Initializing parameters with tensorflow initializers
I've used Xavier for weights, and zeros_initializer for biases.

```python
def initialize_parameters(layer_dims):
    """
    Initializes parameters to build a neural network with tensorflow.
    Returns:
    parameters -- a dictionary of tensors containing Wl, bl
    """

    parameters = {}
    L = len(layer_dims)
    for l in range(1, L):
        parameters['W' + str(l)] = tf.get_variable('W' + str(l), [layer_dims[l],layer_dims[l-1]], initializer = tf.contrib.layers.xavier_initializer())
        parameters['b' + str(l)] = tf.get_variable('b' + str(l), [layer_dims[l], 1], initializer = tf.zeros_initializer())
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters
```

#### Forward propagation

Here's another bonus: check out how it shrunk compared to my numpy version!

```python
def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model: (LINEAR -> RELU)x(L-1)-> LINEAR -> sigmoid

    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing parameters Wl, bl

    Returns:
    ZL -- the output of the last LINEAR unit
    """
    A = X
    L = len(parameters) // 2

    for l in range(1, L):
        A_prev = A
        Z = tf.add(tf.matmul(parameters['W' + str(l)], A_prev), parameters['b' + str(l)])
        A = tf.nn.relu(Z)

    ZL = tf.add(tf.matmul(parameters['W' + str(L)], A), parameters['b' + str(L)])

    return ZL
```

#### Computing cost

```python
def compute_cost(ZL, Y):
    """
    Arguments:
    ZL -- output of forward propagation (output of the last LINEAR unit)
    Y -- "true" labels vector placeholder, same shape as ZL
    Returns:
    cost - Tensor of the cost function
    """

    logits = tf.transpose(ZL)
    labels = tf.transpose(Y)
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = labels))

    return cost
```

#### Splitting into mini-batches

To be fair, this wasn't in my original numpy model, but even with this part, the total amount of code is noticeably smaller.

```python
def random_mini_batches(X, Y, mini_batch_size = 64):
    """
    Creates a list of random minibatches from (X, Y)
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    mini_batch_size - size of the mini-batches, integer

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    m = X.shape[1]                  # number of training examples
    mini_batches = []


    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches
```

#### The model

Again, it would've been a bit shorter without the mini-batches.

```python
def model(X_train, Y_train, X_test, Y_test, layers_dims, learning_rate = 0.0001,
          num_epochs = 1500, minibatch_size = 32, lambd=0.01, print_cost = True):
    """
    Implements a L-layer tensorflow neural network: (LINEAR->RELU)x(L-1)->LINEAR->SIGMOID.

    Arguments:
    X_train -- training set
    Y_train -- test set
    X_test -- training set
    Y_test -- test set
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    (n_x, m) = X_train.shape
    n_y = Y_train.shape[0]
    costs = []

    X, Y = create_placeholders(n_x, n_y)
    parameters = initialize_parameters(layers_dims)
    ZL = forward_propagation(X, parameters)
    cost = compute_cost_L2(ZL, Y, parameters, lambd)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(num_epochs):
            epoch_cost = 0.                       # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size)

            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch

                # Running tensorflow graphon a minibatch.
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                epoch_cost += minibatch_cost / num_minibatches


            # Print the cost every epoch
            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)


        parameters = sess.run(parameters)
        print ("Parameters have been trained!")

        return parameters
```

**Conclusion**:
* **Pros**: No need to calculate the backward pass, yay! Updating parameters is also taken care of. I'd say that was the major source of joy for me.
* **Cons**: There's still quite a lot of code.


## Same with Keras
And then I tried Keras, and it blew my mind.


```python
from keras.models import Sequential
import keras.layers as ll
from keras import regularizers

model = Sequential(name="mlp")

model.add(ll.InputLayer([76]))

# network body
model.add(ll.Dense(units=50, activation='relu'))
model.add(ll.Dense(units=40, activation='relu'))
model.add(ll.Dense(units=30, activation='relu'))
model.add(ll.Dense(units=20, activation='relu'))
model.add(ll.Dense(10))
model.add(ll.Activation('relu'))

model.add(ll.Dense(1, activation='sigmoid'))

model.compile(loss=keras.losses.mean_squared_error,optimizer = keras.optimizers.Adam(lr=0.004, beta_1=0.9, beta_2=0.999, decay=0.0), metrics=["accuracy"])
model.fit(training_set, Y_train,
          validation_data=(dev_set, Y_dev), epochs=15);

```

That. is. it.

**Conclusions**:
I still believe that writing the original NN in numpy was worth it for gaining an insight into the way things work, and what happens at each layer of a neural network.
But next time I'm building a NN, I'll be using tensorflow and keras.

