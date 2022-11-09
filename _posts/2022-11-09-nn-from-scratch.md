---
title:  "A Neural Network From Scratch"
excerpt_separator: "<!--more-->"
date: 2022-11-09
categories:
  - Blog
tags:
  - jupyter 
  - machine-learning
---

# A Neural Net From Scratch
I recently watched [Jeremy Howard's](https://www.youtube.com/user/howardjeremyp) series on deep learning. In an early lecture he shows how to construct a neural network in a spreadsheet, which is clunky but works. His point is that there's nothing mystical about pytorch, tensorflow, or any other ML framework, and encourages the students to write a neural network from scratch.

Of course no one builds their deep learning model like that today, but it is an important exercise for two reasons:

1. Trying to solve the problem as if it was your own idea makes your understanding much more solid
2. [Reality has a surprising amount of detail](http://johnsalvatier.org/blog/2017/reality-has-a-surprising-amount-of-detail). You may know how a neural network works, but *actually* writing one requires thinking to a level of detail that is easy to underestimate.

Let's build a single layer neural network. First of all we must understand what the goal of the model is, and make all design decisions to get us closer to that goal. 

> The primary thing when you take a sword in your hands is your intention to cut the enemy, whatever the means. Whenever you parry, hit, spring, strike or touch the enemy's cutting sword, you must cut the enemy in the same movement. It is essential to attain this. If you think only of hitting, springing, striking or touching the enemy, you will not be able actually to cut him. (Miyamoto Musashi)

Our goal is to *learn what makes a neural network tick*. We will use as little help from prepackaged libraries and study only a well understood problem (we are not interested in data wrangling). The easiest choice is to solve the classic [handwritten digits classification](https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits) problem, so we go and download the dataset.

We load them into a `numpy` array (we're not using pytorch)
```python
data_raw = np.loadtxt("optdigits.tra",delimiter=",")
```
and separate the data from the labels 
```python
labels = data_raw[:,-1].astype(int)
data = data_raw[:,:-1]
```

Our first choice: how do we represent the labels? In the dataset they're a single integer, but if we represent them as integers in our code they behave in ways that are *conceptually wrong*: integers are ordered $(2 > 1)$ but that is entirely irrelevant for the purpose or recognizing the handwritten digits. In fact, it's actively misleading the neural network into learning a structure that does not exist in the data. 
To avoid this problem, we [one-hot](https://en.wikipedia.org/wiki/One-hot) encode the labels
```python
    def one_hot_encode(label):
        # num_classes is the  number of distinct labels
        res = np.zeros(num_classes)
        res[label] = 1
        return res
```

The images we are working with are $8 \times 8$, and we are trying to recognize $10$ digits, so the model must have 64 inputs and 10 outputs. Once we have the model, the fundamental prediction step goes like this:
- Take a datapoint, represented as a $64$-entry long array $x$
- multiply by the parameter matrix $W$, which has $64\times 10$ real entries
- add the bias vector $b$ (a $10$-entry long array)
- for each entry in $z = Wx + b$ compute the activation $\sigma(z)$, where $\sigma$ is a nonlinear function

We haven't even started talking about training, and we are hit at full speed by the reality-has-a-lot-of-detail truck:
- Should we write the $Wx + b$ function to work with a single array $x$ or pack a bunch of them in a matrix $X$? The first approach is simple, and it is clear to think about in terms of a loop (for each datapoint in this batch...). The matrix approach exploits broadcasting and the vectorized math in numpy, which makes it faster. 
- How do we reseprent the bias vector? We can either treat $W$ and $b$ separately, or extend $W$ with an extra column and append a $1$ to every datapoint. Again, the first option is easier to think about, but the second makes writing the code for gradient descent easier.
- What nonlinearity? The sigmoid has nice analytical properties, but a [ReLU]https://en.wikipedia.org/wiki/Rectifier_(neural_networks) usually has more stable gradients.
- How to represent the model in practice? Just write the matrix and a collection of functions acting on it? I prefer to write a class, and think of every action as a method.


Whew, that was a lot of questions. At the bottom of this post you can see what choices I ended up making. A few more details about training:
- We use simple stochastic gradient descent (with minibatches). No momentum, no restarts, or any fancy techniques. This works decently and is fast.
- The cost function is the error rate
- To avoid overfitting we look at the error rate *on the validation set* and stop training when that flattens.

Ok, let's start.


```python
import numpy as np
```

The default plots in `matplotlib` are ugly, this is a much better style


```python
import matplotlib.pyplot as plt
plt.style.use("ggplot")
%config InlineBackend.figure_formats = ['svg']
```

`tqdm` is a tiny library that makes loading bars. It's delightful.


```python
from tqdm.notebook import tqdm
```

Load data and extract the labels


```python
data_raw = np.loadtxt("optdigits.tra",delimiter=",")

# separate data from labels
labels = data_raw[:,-1].astype(int)
data = data_raw[:,:-1]
```

We use a ReLU, and vectorize the function using numpy: `relu(x)` will be applied to every entry of the iterable `x`.


```python
def relu(x):
    return x if x > 0 else 0

relu = np.vectorize(relu)
```


```python
class NeuralLayer:
    def __init__(self,data,labels):
        self.classes = np.unique(labels)
        self.labels = labels
        # we append 1 to every datapoint so W contains bias 
        self.data = np.hstack((data,np.ones((len(data),1))))
        
        # all indices larger than this are in the validation set
        self.max_train_idx = int(0.8*len(self.data))
        
        self.parameters = self.init_parameters(len(self.data[0]),len(self.classes))
        self.nonlinearity = relu
        self.batch_size = 30
        self.gradient = np.empty_like(self.parameters)
        self.err_history = []
        self.learn_rate = 1e-4
        
    def one_hot_encode(self,label):
        # one-hot encoding
        res = np.zeros(len(self.classes))
        res[label] = 1
        
        return res
    
    def init_parameters(self,n_inputs,n_classes):
        return 0.1*np.random.randn(n_inputs,n_classes)
    
    def pick_batch(self):
        # returns a list of indices, which can be used to extract a minibatch from the
        # data matrix
        return np.random.choice(self.max_train_idx,size=self.batch_size,replace=False)
    
    def valid_error(self):
        return self.err_rate(range(self.max_train_idx,len(self.data)))
    
    def show_im(self,idx):
        x = np.reshape(self.data[idx][:-1],(8,8)) 
        plt.imshow(x,cmap="Blues")
        plt.title(f"True label: {labels[idx]}")
        plt.axis("off")
        
    def predict(self,idx,ret_confidence=False):

        # weights at output
        y = self.nonlinearity(self.parameters.T@self.data[idx])

        if ret_confidence:
            confidence = y/torch.sum(y)
            return np.argmax(y),confidence
        else:
            return np.argmax(y)
    
    def err_rate(self,idxs):
        # check the accuracy rate of the net 
        # idxs = array-like of indices to check from the dataset data
        E = 0
        for idx in idxs:
            E += self.predict(idx) != self.labels[idx]
        
        return E/len(idxs)

    def compute_gradient(self):
        batch = self.pick_batch()
        
        dE = 0
        for idx in batch:
            y_true = self.one_hot_encode(self.labels[idx])
            y_pred = self.one_hot_encode(self.predict(idx))
            
            dE += 2*(y_pred - y_true)[None,:]*self.data[idx][:,None]
        
        return dE
    
    def sgd_step(self):
        
        dW = self.compute_gradient()
        self.parameters = self.parameters - self.learn_rate*dW
    
    def train(self,epochs=1):
        
        steps = len(data)//self.batch_size
        
        for _ in tqdm(range(epochs*steps)):
            err_rate = self.valid_error()
            print(f"Err rate: {err_rate}",end="\r")
            self.sgd_step()
            self.err_history.append(err_rate)
```


```python
layer = NeuralLayer(data,labels)
```


```python
idx = 1337
layer.show_im(idx)
print(f"Prediction: {layer.predict(idx)}")
```

    Prediction: 7



    
![svg](output_12_1.svg)
    



```python
layer.batch_size = 20
layer.learn_rate = 1e-4
layer.train(epochs=10)
```


      0%|          | 0/1910 [00:00<?, ?it/s]


    Err rate: 0.066666666666666675

Let's plot the error rate on the validation set


```python
plt.plot(layer.err_history)
plt.xlabel("SGD Step")
plt.ylabel("Valid. Err. Rate")
```




    Text(0, 0.5, 'Valid. Err. Rate')




    
![svg](output_15_1.svg)
    


Not bad. 


```python

```
