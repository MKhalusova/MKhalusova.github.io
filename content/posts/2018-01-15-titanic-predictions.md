---
date: "2018-01-15T00:00:00Z"
title: Predicting survival of Titanic passengers
---

I've been hesitant to write this blog post. On one hand, I managed to build my first (i.e. not in the context of a course exercise)
neural network that predicts something - in this case, survival or death of Titanic passengers.
There's also quite involved data cleanup beforehand, which took some time.
Overall I feel like I have made my first tiny step towards "real" machine learning. It does feel like an achievement - this is huge compared to what I could do just a couple of months ago.
However, this mini-milestone is bittersweet, because unfortunately, no matter what I tried so far, I couldn't get the accuracy of this model higher than 78,5% on the test set.
That's why I was hesitant. I wanted to have my glorious 85% before I'd blog about it :)
I've changed my mind and I'm writing about what I have now for two reasons: one - I do think this is a milestone worth recording, and two - I'd like to hear some feedback: what am I doing wrong?,
what can I improve? Any advice is appreciated and I hope that with the help of this post I'll learn something.

Ok, let's jump into it.
The data and the problem description are from Kaggle competition: [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic)

In this challenge, there are two cvs with passenger data:
* **training set (train.csv)**: The training set should be used to build a machine learning model.
                            For the training set, the outcome (also known as the "ground truth”) is provided for each passenger.
* **test set (test.csv)**: The test set should be used to see how well the model performs on unseen data. For the test set, the ground truth is not provided.
                         For each passenger in the test set, the model needs to predict whether or not they survived the sinking of the Titanic.


Here's a description of the data:
* **Survival**: 	0 = No, 1 = Yes
* **pclass**:	Passenger class. 1 = 1st, 2 = 2nd, 3 = 3rd
* **sex**:	Sex
* **Age**:	Age in years
* **sibsp**:	Number of siblings / spouses aboard the Titanic
* **parch**:	Number of parents / children aboard the Titanic
* **ticket**:	Ticket number
* **fare**:	Passenger fare
* **cabin**:	Cabin number
* **embarked**:	Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton


So here's what I did (if you just want to see the code, here's my [Jupyter Notebook on GitHub](https://github.com/MKhalusova/titanic-predictions/blob/master/titanic-predictions.ipynb)).
First, import some libraries and load the data:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dnn_utils_v2 import *

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
test_df_copy = test_df
```

I'm going to skip the preview of the data, and get straight into what I did with it.
First, I combine the data from the training set and the test set (except the ground truth), so that I can do the clean up operations once, and not do the same thing twice - once per each set.

```python
targets = train_df.Survived
train_df.drop('Survived', 1, inplace=True)
combined_sets = train_df.append(test_df)
combined_sets.reset_index(inplace=True)
combined_sets.drop('index', inplace=True, axis=1)
```

Next, I drop the passenger Id as it seems to be just an index.

```python
combined_sets = combined_sets.drop(['PassengerId'], axis=1)
```

It's a good idea to check for missing values.

```python
combined_sets.isnull().sum()
```

Turns out, 263 passengers don't have Age specified, one passenger is missing information abour Fare, two have nothing in Embarked field, and for 1014 (ouch!) passengers a cabin is unknown.
We'll get back to the null values later, let's first do something about fields where we do have all the values, but they are categorical or strings.
First, let's [dummy encode](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html) passenger class.

```python
pclass_dummies = pd.get_dummies(combined_sets['Pclass'], prefix="Pclass")
combined_sets = pd.concat([combined_sets,pclass_dummies],axis=1)
combined_sets.drop('Pclass',axis=1,inplace=True)
```

Then, do the same thing with Sex column:

```python
gender_dummies = pd.get_dummies(combined_sets['Sex'],prefix='Sex')
combined_sets = pd.concat([combined_sets,gender_dummies],axis=1)
combined_sets.drop('Sex',axis=1,inplace=True)
```

Next thing, I'm extracting the title from the Name column. A title reflects social status of a person, gender, and in some cases, Age range. It can be helpful later when trying to fill the missing values:

```python
combined_sets['Title'] = combined_sets['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
```

Dummy encoding titles:

```python
title_dummies = pd.get_dummies(combined_sets['Title'], prefix="Title")
combined_sets = pd.concat([combined_sets,title_dummies],axis=1)
combined_sets.drop('Title',axis=1,inplace=True)
```

Extracting last name. This will help me later with some of the missing cabin values:

```python
combined_sets['Last_Name'] = combined_sets['Name'].map(lambda name:name.split(',')[0])
```

Apart from last name and title, I don't think there's any other useful information in the Name column, so I drop it at this point.

```python
combined_sets = combined_sets.drop(['Name'], axis=1)
```

Next, I add an extra feature - family size (including the passenger):

```python
combined_sets['FamilySize'] = combined_sets['SibSp'] + combined_sets['Parch'] +1
```

I've extracted the ticket number:

```python
combined_sets['Ticket_Number'] = combined_sets['Ticket'].map(lambda x:x.rsplit(' ', 1)[-1])
```

And created a column with the ticket prefix (the ones that don't have prefix, get 'XXX')

```python
def cleanTicket(ticket):
        ticket = ticket.replace('.','')
        ticket = ticket.replace('/','')
        ticket = ticket.split()
        if len(ticket) > 1:
            return ticket[0]
        else:
            return 'XXX'

combined_sets['Ticket'] = combined_sets['Ticket'].map(cleanTicket)
ticket_dummies = pd.get_dummies(combined_sets['Ticket'], prefix='Ticket')
combined_sets = pd.concat([combined_sets, ticket_dummies], axis=1)
combined_sets.drop('Ticket', inplace=True, axis=1)
```

Now, let's get to the missing values. First, let's take a look at Embarked column.
Turns out, both passengers with missing embarkation value have the same ticket number: 113572.

```python
# combined_sets[(combined_sets['Embarked'].isnull())]['Ticket_Number'] # 113572
combined_sets[(combined_sets['Ticket_Number'].map(lambda x: x.startswith('1135')))]['Embarked'].value_counts()
-----------------
C    7
S    5
Name: Embarked, dtype: int64
-----------------
```

And among those with similar ticket numbers, most embarked in Cherbourg (value C). I'm going to assume, they have embarked in Cherbourg.

```python
combined_sets['Embarked'].fillna('C', inplace=True)
```

Now that there are no missing values in 'Embarked', I can dummy encode it.

```python
embarked_dummies = pd.get_dummies(combined_sets['Embarked'],prefix='Embarked')
combined_sets = pd.concat([combined_sets,embarked_dummies],axis=1)
combined_sets.drop('Embarked',axis=1,inplace=True)
```

Let's see if we can guess the fare of that one passenger.

```python
combined_sets[combined_sets['Fare'].isnull()]
```

The passenger with missing fare has Pclass_3==1 Embarked_S== 1 FamilySize==1 Ticket_XXX==1 Title_Mr==1
I'll get the average fare from the passengers with same values, and assign it to him.


```python
m_fare = combined_sets[(combined_sets['Embarked_S']==1) & (combined_sets['Pclass_3']==1)
        & (combined_sets['FamilySize']==1) & (combined_sets['Ticket_XXX']==1)
        & (combined_sets['Title_Mr']==1) & (combined_sets['Age'] > 50)]['Fare'].mean()
combined_sets['Fare'].fillna(m_fare, inplace=True)
```

There's quite a lot of missing values in 'Age' (263). I'm going to average the age among groups with the same passenger class and title (skipping gender, because it's in the title too)

```python
def replace_age_with_mean(df, pclass, title):
    mask = ((df[pclass]==1) & (df[title]==1))
    med = df.loc[mask, 'Age'].mean()
    df.loc[mask, 'Age'] = df.loc[mask, 'Age'].fillna(med)
    return df
```


```python
titles = ['Title_Mrs','Title_Miss','Title_Mr','Title_Dr', 'Title_Master', 'Title_Ms']
pclass = ['Pclass_1', 'Pclass_2', 'Pclass_3']
for p in pclass:
    for title in titles:
        combined_sets = replace_age_with_mean(combined_sets, p, title)
```

After this step, there's still one missing value in Age, because there was only one Ms in the third class, thus there's no one to average.
She get's the average among all passenger with title Ms regardless of passenger class.

```python
m = combined_sets.loc[(combined_sets['Title_Ms']==1), 'Age'].mean()
combined_sets['Age'].fillna(m, inplace=True)
```

Time to deal with missing values for in 'Cabin'.
Turns out there are 693 NaN out of 709 in the third class, 254 out of 277 in the second class, and 67 out of 323 in the first class.
My intuition is that the passengers in third and second class may not have been even assigned cabins,
but the ones in the first class are genuine missing values. I'll try to estimate them.

Here, it'll be useful to know the last name of a person. Let's get the list of all the last names from the first class that are encountered more than once.
These are the ones who travelled with family.

```python
last_names = combined_sets[(combined_sets['Pclass_1']==1)]['Last_Name'].value_counts()
fams = last_names[last_names>1].index.tolist()
```

Now I'm going to extract two sub-lists from it. First - families where for some a cabin is known, and for others isn't.
The second - families where all members of it are missing value for 'Cabin'.

```python
fams_with_a_nan = []
# list of families where all Cabin values are NaN
fams_with_all_nan = []
for name in fams:
    if combined_sets[(combined_sets['Pclass_1']==1) & (combined_sets['Last_Name']==name)]['Cabin'].isnull().all()==True:
        fams_with_all_nan.append(name)
    else:
        if combined_sets[(combined_sets['Pclass_1']==1) & (combined_sets['Last_Name']==name)]['Cabin'].isnull().any()== True:
            fams_with_a_nan.append(name)
```

Those from the same family (where the cabin is known for some) are placed in the same cabin:

```python
for name in fams_with_a_nan:
    mask = ((combined_sets['Pclass_1']==1) & (combined_sets['Last_Name']==name))
    cabin = combined_sets[mask]['Cabin'].value_counts()
    cabin = cabin.index.tolist()[0]
    combined_sets.loc[mask, 'Cabin'] = combined_sets.loc[mask, 'Cabin'].fillna(cabin)
```

For the rest we cannot guess the exact cabin, so we can drop the number and leave just the letter.
```python
mask = ((combined_sets['Cabin'].isnull()==False))
combined_sets.loc[mask, 'Cabin'] = combined_sets.loc[mask, 'Cabin'].map(lambda c : c[0])
```

For the families without a known cabin, we'll check what fare they have and compare to similar fares.

```python
for name in fams_with_all_nan:
    fare = combined_sets[(combined_sets['Pclass_1']==1) & (combined_sets['Last_Name']==name)]['Fare'].mean()
    mask1 = ((combined_sets['Pclass_1']==1) & (combined_sets['Fare']>(fare-1)) & (combined_sets['Fare']<(fare+1)))
    cabin = combined_sets[mask1]['Cabin'].value_counts()
    cabin = cabin.index.tolist()
    if not cabin:
        cabin = np.nan
    else:
        cabin = cabin[0]
    mask2 = ((combined_sets['Pclass_1']==1) & (combined_sets['Last_Name']==name))
    combined_sets.loc[mask2, 'Cabin'] = combined_sets.loc[mask2, 'Cabin'].fillna(cabin)
```

There are still 52 NaN in Cabin in first class. I'm going to use Fare to predict the Cabin.
```python
fares = combined_sets[(combined_sets['Pclass_1']==1) & (combined_sets['Cabin'].isnull())]['Fare'].value_counts()
fares = fares.index.tolist()
```


```python
for fare in fares:
    mask1 = ((combined_sets['Pclass_1']==1) & (combined_sets['Fare']>(fare-1)) & (combined_sets['Fare']<(fare+1)))
    cabin = combined_sets[mask1]['Cabin'].value_counts()
    cabin = cabin.index.tolist()
    if not cabin:
        cabin = np.nan
    else:
        cabin = cabin[0]
    mask2 = ((combined_sets['Pclass_1']==1) & (combined_sets['Fare']==fare))
    combined_sets.loc[mask2, 'Cabin'] = combined_sets.loc[mask2, 'Cabin'].fillna(cabin)
```

There are still 3 missing values.

```python
combined_sets[(combined_sets['Pclass_1']==1)]['Cabin'].value_counts()
```

I'll assign the common value in the first class to them, which is C.

```python
mask = ((combined_sets['Pclass_1']==1))
combined_sets.loc[mask, 'Cabin'] = combined_sets.loc[mask, 'Cabin'].fillna('C')
```

The missing cabins in the third and second classes, I'm filling with 'U' for Unknown.

```python
combined_sets['Cabin'].fillna('U', inplace=True)
```

Now cabins can be dummy encoded too.

```python
cabin_dummies = pd.get_dummies(combined_sets['Cabin'], prefix='Cabin')
combined_sets = pd.concat([combined_sets,cabin_dummies], axis=1)
combined_sets.drop('Cabin', axis=1, inplace=True)
```

Now we can drop the Last Name and the ticket number.

```python
combined_sets = combined_sets.drop(['Last_Name'], axis=1)
combined_sets = combined_sets.drop(['Ticket_Number'], axis=1)
```

The cleanup is finished, there are no more categorical values, strings or missing values. So it's time to split it back into the train and test sets.

```python
train_df = combined_sets.head(891)
train_Y = targets
test_df = combined_sets.iloc[891:]
```

Now I need to convert the data into an array for my model.

```python
train_array = train_df.as_matrix()
Y_array = train_Y.as_matrix()
Y_array = Y_array.reshape(Y_array.shape[0],1)
```

I'll use `train_test_split` from `sklearn.model_selection` to split my training set into a training set and a development set to check my model.

```python
from sklearn.model_selection import train_test_split
training_set, dev_set, Y_train, Y_dev = train_test_split(train_array, Y_array)
training_set = np.transpose(training_set)
dev_set = np.transpose(dev_set)
Y_train = np.transpose(Y_train)
Y_dev = np.transpose(Y_dev)
```

Time to work on the model. I'm building a L-layer NN with (L-1) layers linear-relu, and the last layer with sigmoid.

Initializing parameters for the model:

```python
def initialize_parameters_deep(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """

    parameters = {}
    L = len(layer_dims)            # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters
```

Implementing the linear part of an epoch.

```python
def linear_forward(A, W, b):
    """
    The linear part of a layer's forward propagation.
    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter
    cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """
    Z = np.add(np.matmul(W, A), b)
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    return Z, cache
```

Linear activation forward:

```python
def linear_activation_forward(A_prev, W, b, activation):
    """
    Forward propagation for the LINEAR->ACTIVATION layer
    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value
    cache -- a python dictionary containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
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

And here's the linear forward pass.

```python
def L_model_forward(X, parameters):
    """
    Forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()

    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                the cache of linear_sigmoid_forward() (there is one, indexed L-1)
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

Computing cost after a forward pass:

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

Now, linear part of the backward pass.

```python
def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

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

Linear activation backward:

```python
def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.

    Arguments:
    dA -- post-activation gradient for current layer l
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
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

Back propagation pass

```python
def L_model_backward(AL, Y, caches):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])

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
        # Inputs: "grads["dA" + str(l + 2)], caches". Outputs: "grads["dA" + str(l + 1)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)]

        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA"+str(l+2)],current_cache,"relu")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads
```

Updating parameters for the model:

```python
def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent
    Arguments:
    parameters -- python dictionary containing your parameters
    grads -- python dictionary containing your gradients, output of L_model_backward

    Returns:
    parameters -- python dictionary containing your updated parameters
                  parameters["W" + str(l)] = ...
                  parameters["b" + str(l)] = ...
    """

    L = len(parameters) // 2 # number of layers in the neural network

    # Update rule for each parameter
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]

    return parameters
```

Setting the number of layers and their sizes as a constant

```python
layers_dims = [training_set.shape[0], 20, 10, 10, 1]
```

Finally, the model itself

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
    parameters -- parameters learnt by the model. They can then be used to predict.
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

    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters
```

Now, here comes the fun part where we get to train the model.

```python
parameters = L_layer_model(training_set, Y_train, layers_dims, num_iterations = 50000, print_cost = True)
```

To predict the values on the dev_set and the test set, we need one more function

```python
def predict(parameters, X):
    """
    Using the learned parameters, predicts a class for each example in X

    Arguments:
    parameters -- python dictionary containing your parameters
    X -- input data of size (n_x, m)

    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """

    # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
    A2, cache = L_model_forward(X, parameters)
    predictions = np.array([0 if i <= 0.5 else 1 for i in np.squeeze(A2)])

    return predictions
```

So this particular combination of layers, learning rate and number of iteration gives me the following:

```python
pred_train = predict(parameters,training_set)
print ('Accuracy: %d' % float((np.dot(Y_train,pred_train.T) + np.dot(1-Y_train,1-pred_train.T))/float(Y_train.size)*100) + '%')
----------------
Accuracy: 86%
----------------
```

As for the dev set, the accuracy is a bit lower:
```python
pred_dev = predict(parameters, dev_set)
print ('Accuracy: %d' % float((np.dot(Y_dev,pred_dev.T) + np.dot(1-Y_dev,1-pred_dev.T))/float(Y_dev.size)*100) + '%')
----------------
Accuracy: 81%
----------------
```

To get the predictions on the test set, I need to convert it into an array as well.
```python
test_array = test_df.as_matrix()
test_array = np.transpose(test_array)
Y_test_predictions = predict(parameters, test_array)
Y_test_predictions = Y_test_predictions.astype(int)
Y_test_df = pd.DataFrame(np.transpose(Y_test_predictions))
```


The only thing left is to prepare the answer in the right format to upload it to kaggle.
```python
test_df_copy['Survived'] = Y_test_df
test_df_copy = test_df_copy.drop(['Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare', 'Cabin','Embarked'], axis=1)
test_df_copy.to_csv('predictions.csv',index=False)
```

This implementation gives me maximum of 78.5 % accuracy. There are a lot of missing values in the original data, I make a lot of assumptions too, plus there's an element of luck, of course.
I've tried different number of layers, different sizes, I've tried optimizations, and various learning rates. I don't think it's realistic to get an estimate higher than 85% without cheating,
so that's where I was striving to get. But as of now I'm stuck and I'll leave it as is now, and get back to it some time later with some fresh knowledge :)



