# PyTorch Tutorial

PyTorch is an open-source deep learning framework designed to simplify the process of building neural networks and machine learning models. With its dynamic computation graph, PyTorch allows developers to modify the network’s behavior in real-time, making it an excellent choice for both beginners and researchers.

## Tensors in PyTorch

A [**tensor**](https://www.geeksforgeeks.org/python/tensors-in-pytorch/) is a multi-dimensional array that is the fundamental data structure used in PyTorch (and many other machine learning frameworks).

We can create tensors for performing above in several ways:

```python
**import** **torch**

tensor_1d = torch.tensor([1, 2, 3])
print("1D Tensor (Vector):")
print(tensor_1d)
print()

tensor_2d = torch.tensor([[1, 2], [3, 4]])
print("2D Tensor (Matrix):")
print(tensor_2d)
print()

random_tensor = torch.rand(2, 3)
print("Random Tensor (2x3):")
print(random_tensor)
print()

zeros_tensor = torch.zeros(2, 3)
print("Zeros Tensor (2x3):")
print(zeros_tensor)
print()

ones_tensor = torch.ones(2, 3)
print("Ones Tensor (2x3):")
print(ones_tensor)
```

### Tensor Operations in PyTorch

PyTorch operations are essential for manipulating data efficiently, especially when preparing data for machine learning tasks.

- [**Indexing](https://www.geeksforgeeks.org/python/pytorch-index-based-operation/):** Indexing lets you retrieve specific elements or smaller sections from a larger tensor.
- [**Slicing](https://www.geeksforgeeks.org/machine-learning/tensor-slicing/):** Slicing allows you to take out a portion of the tensor by specifying a range of rows or columns.
- [**Reshaping](https://www.geeksforgeeks.org/python/reshaping-a-tensor-in-pytorch/):** Reshaping changes the shape or dimensions of a tensor without changing its actual data. This means you can reorganize the tensor into a different size while keeping all the original values intact.

Let's understand these operations with help of simple implementation:

```python
import torch

tensor = torch.tensor([[1, 2], [3, 4], [5, 6]])

element = tensor[1, 0]
print(f"Indexed Element (Row 1, Column 0): {element}") 
 
slice_tensor = tensor[:2, :]
print(f"Sliced Tensor (First two rows): \n{slice_tensor}")

reshaped_tensor = tensor.view(2, 3)
print(f"Reshaped Tensor (2x3): \n{reshaped_tensor}")
```

### Common Tensor Functions: Broadcasting, Matrix Multiplication, etc.

PyTorch offers a variety of common tensor functions that simplify complex operations.

- [**Broadcasting**](https://www.geeksforgeeks.org/deep-learning/tensor-broadcasting/) allows for automatic expansion of dimensions to facilitate arithmetic operations on tensors of different shapes.
- [**Matrix multiplication**](https://www.geeksforgeeks.org/python/python-matrix-multiplication-using-pytorch/) enables efficient computations essential for neural network operations.

```python
import torch

tensor_a = torch.tensor([[1, 2, 3], [4, 5, 6]])

tensor_b = torch.tensor([[10, 20, 30]]) 

broadcasted_result = tensor_a + tensor_b 
print(f"Broadcasted Addition Result: \n{broadcasted_result}")

matrix_multiplication_result = torch.matmul(tensor_a, tensor_a.T)
print(f"Matrix Multiplication Result (tensor_a * tensor_a^T): \n{matrix_multiplication_result}")
```

### GPU Acceleration with PyTorch

PyTorch facilitates GPU acceleration, enabling much faster computations, which is especially important in deep learning due to the extensive matrix operations involved. By transferring tensors to the GPU, you can significantly reduce training times and improve performance.

```python
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

tensor_size = (10000, 10000)  
a = torch.randn(tensor_size, device=device)  
b = torch.randn(tensor_size, device=device)  

c = a + b  

print("Result shape (moved to CPU for printing):", c.cpu().shape)

print("Current GPU memory usage:")
print(f"Allocated: {torch.cuda.memory_allocated(device) / (1024 ** 2):.2f} MB")
print(f"Cached: {torch.cuda.memory_reserved(device) / (1024 ** 2):.2f} MB")
```

## Building and Training Neural Networks with PyTorch

In this section, we'll implement a neural network using PyTorch, following these steps:

### **Step 1: Define the Neural Network Class**

In this step, we’ll define a class that inherits from `torch.nn.Module`. We’ll create a simple neural network with an input layer, a hidden layer, and an output layer.

```python
import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 4)  
        self.fc2 = nn.Linear(4, 1)  

    def forward(self, x):
        x = torch.relu(self.fc1(x))  
        x = self.fc2(x)               
        return x
```

### **Step 2: Prepare the Data**

Next, we’ll prepare our data. We will use a simple dataset that represents the XOR logic gate, consisting of binary input pairs and their corresponding XOR results.

```python
X_train = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]) 
y_train = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
```

### **Step 3: Instantiate the Model, Loss Function, and Optimizer**

Now it’s time for us to instantiate our model. We’ll also define a [**loss function**](https://www.geeksforgeeks.org/deep-learning/pytorch-loss-functions/)(Mean Squared Error) and choose an [**optimizer**](https://www.geeksforgeeks.org/machine-learning/how-to-implement-various-optimization-algorithms-in-pytorch/)(Stochastic Gradient Descent) to update the model’s weights based on the calculated loss.

```python
import torch.optim as optim

model = SimpleNN()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
```

### **Step 5: Training the Model**

Now we enter the training loop, where we will repeatedly pass our training data through the model to learn from it.

```python
for epoch in range(100):
    model.train()

    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/100], Loss: {loss.item():.4f}')
```

### Step 6: Testing the Model

Finally, we need to evaluate the model’s performance on new data to assess its generalization capability.

```python
model.eval()
with torch.no_grad():
    test_data = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    predictions = model(test_data)
    print(f'Predictions:\n{predictions}')
```

## Optimizing Model Training with PyTorch Datasets

### 1. **Efficient Data Handling with Datasets and DataLoaders**

[**Dataset and DataLoader**](https://www.geeksforgeeks.org/python/datasets-and-dataloaders-in-pytorch/) facilitates batch processing and shuffling, ensuring smooth data iteration during training.

```python
import torch
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self):
        self.data = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        self.labels = torch.tensor([0, 1, 0])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

dataset = MyDataset()
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

for batch in dataloader:
    print("Batch Data:", batch[0])  
    print("Batch Labels:", batch[1])
```

### **2. Enhancing Data Diversity through Augmentation**

[**Torchvision**](https://www.geeksforgeeks.org/deep-learning/computer-vision-with-pytorch/) provides simple tools for applying random transformations—such as rotations, flips, and scaling—enhancing the model's ability to generalize on unseen data.

```python
import torchvision.transforms as transforms
from PIL import Image

image = Image.open('example.jpg')  # Replace 'example.jpg' with your image file

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

augmented_image = transform(image)
print("Augmented Image Shape:", augmented_image.shape)
```

### **3. Batch Processing for Efficient Training**

Batch processing improves computational efficiency and accelerates training, especially on hardware accelerators.

```python
for epoch in range(2):  
    for inputs, labels in dataloader:
        
        outputs = inputs + 1  
        print(f"Epoch {epoch + 1}, Inputs: {inputs}, Labels: {labels}, Outputs: {outputs}")
```

By combining the power of Datasets, Dataloaders, data augmentation, and batch processing, PyTorch offers an effective way to handle data, streamline training, and optimize performance for machine learning tasks.

## Advanced Deep Learning Models in PyTorch

### 1. Convolutional Neural Networks (CNNs)

- PyTorch simplifies the implementation of CNNs using modules like [**torch.nn.Conv2d**](https://www.geeksforgeeks.org/computer-vision/apply-a-2d-convolution-operation-in-pytorch/) and [**pooling layers**](https://www.geeksforgeeks.org/computer-vision/apply-a-2d-max-pooling-in-pytorch/).
- Integrating [**batch normalization**](https://www.geeksforgeeks.org/deep-learning/batch-normalization-implementation-in-pytorch/) with torch.nn.BatchNorm2d helps stabilize learning and accelerate training by normalizing the output of convolutional layers.

### 2. Recurrent Neural Networks (RNNs)

- Implementing RNNs in PyTorch is straightforward with [**torch.nn.LSTM**](https://www.geeksforgeeks.org/deep-learning/difference-between-hidden-and-output-in-pytorch-lstm/) and torch.nn.GRU modules.
- RNNs, including LSTMs and GRUs, are perfect for sequential data tasks.

### 3. Generative Models

PyTorch makes it easy to constructGenerative Models, including:

- [**Generative Adversarial Networks (GANs)**](https://www.geeksforgeeks.org/deep-learning/generative-adversarial-networks-gans-in-pytorch/): Involve a generator and a discriminator that compete to create realistic data.
- [**Variational Autoencoders (VAEs)**](https://www.geeksforgeeks.org/deep-learning/implementing-an-autoencoder-in-pytorch/): Learn probabilistic mappings, facilitating various applications in data generation.

### Transfer Learning in PyTorch

1. [**Fine-Tuning Pretrained Models](https://www.geeksforgeeks.org/nlp/transfer-learning-and-fine-tuning-in-nlp/):** PyTorch makes fine-tuning pretrained models straightforward. By using models trained on extensive datasets like ImageNet, you can easily modify the final layers and retrain them on your dataset, capitalizing on the pretrained features while tailoring the model to your specific needs.
2. [**Implementing Transfer Learning with torchvision.models](https://www.geeksforgeeks.org/computer-vision/transfer-learning-for-computer-vision/):** torchvision.models module offers a variety of pretrained models, including ResNet, VGG, and Inception. Loading a pretrained model and replacing its classifier with your custom layers is simple, ensuring the model is tailored for your dataset.
3. [**Freezing and Unfreezing Layers](https://www.geeksforgeeks.org/deep-learning/how-to-implement-transfer-learning-in-pytorch/):** An essential aspect of transfer learning is the ability to freeze and unfreeze layers in the pretrained model. Freezing certain layers prevents their weights from updating, preserving learned features. This technique is beneficial for focusing on training newly added layers. Conversely, unfreezing layers allows for fine-tuning, enabling the model to adjust its weights based on your dataset for improved performance.

## Automatic Differentiation with `torch.autograd`

When training neural networks, the most frequently used algorithm is **back propagation**. In this algorithm, parameters (model weights) are adjusted according to the **gradient** of the loss function with respect to the given parameter.

To compute those gradients, PyTorch has a built-in differentiation engine called `torch.autograd`. It supports automatic computation of gradient for any computational graph.

Consider the simplest one-layer neural network, with input `x`, parameters `w` and `b`, and some loss function. It can be defined in PyTorch in the following manner:

```python
import torch

x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
```

### Tensors, Functions and Computational graph

This code defines the following **computational graph**:

![](https://docs.pytorch.org/tutorials/_images/comp-graph.png)

In this network, `w` and `b` are **parameters**, which we need to optimize. Thus, we need to be able to compute the gradients of loss function with respect to those variables. In order to do that, we set the `requires_grad` property of those tensors.

**Note:** You can set the value of `requires_grad` when creating a tensor, or later by using `x.requires_grad_(True)` method.

A function that we apply to tensors to construct computational graph is in fact an object of class `Function`. This object knows how to compute the function in the *forward* direction, and also how to compute its derivative during the *backward propagation* step. A reference to the backward propagation function is stored in `grad_fn` property of a tensor. You can find more information of `Function` [in the documentation](https://pytorch.org/docs/stable/autograd.html#function).

```python
print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")
```

### Computing Gradients

To optimize weights of parameters in the neural network, we need to compute the derivatives of our loss function with respect to parameters, namely, we need ∂loss∂w∂*w*∂*loss* and ∂loss∂b∂*b*∂*loss* under some fixed values of `x` and `y`. To compute those derivatives, we call `loss.backward()`, and then retrieve the values from `w.grad` and `b.grad`:

```python
loss.backward()
print(w.grad)
print(b.grad)
```

**Note**

- We can only obtain the `grad` properties for the leaf nodes of the computational graph, which have `requires_grad` property set to `True`. For all other nodes in our graph, gradients will not be available.
- We can only perform gradient calculations using `backward` once on a given graph, for performance reasons. If we need to do several `backward` calls on the same graph, we need to pass `retain_graph=True` to the `backward` call.

### Disabling Gradient Tracking

By default, all tensors with `requires_grad=True` are tracking their computational history and support gradient computation. However, there are some cases when we do not need to do that, for example, when we have trained the model and just want to apply it to some input data, i.e. we only want to do *forward* computations through the network. We can stop tracking computations by surrounding our computation code with `torch.no_grad()` block:

```python
z = torch.matmul(x, w)+b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w)+b
print(z.requires_grad)
```

Another way to achieve the same result is to use the `detach()` method on the tensor:

```python
z = torch.matmul(x, w)+b
z_det = z.detach()
print(z_det.requires_grad)
```

**There are reasons you might want to disable gradient tracking:**

- To mark some parameters in your neural network as **frozen parameters**.
- To **speed up computations** when you are only doing forward pass, because computations on tensors that do not track gradients would be more efficient.

### More on Computational Graphs

Conceptually, autograd keeps a record of data (tensors) and all executed operations (along with the resulting new tensors) in a directed acyclic graph (DAG) consisting of [Function](https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function) objects. In this DAG, leaves are the input tensors, roots are the output tensors. By tracing this graph from roots to leaves, you can automatically compute the gradients using the chain rule.

In a forward pass, autograd does two things simultaneously:

- run the requested operation to compute a resulting tensor
- maintain the operation’s *gradient function* in the DAG.

The backward pass kicks off when `.backward()` is called on the DAG root. `autograd` then:

- computes the gradients from each `.grad_fn`,
- accumulates them in the respective tensor’s `.grad` attribute
- using the chain rule, propagates all the way to the leaf tensors.

**Note: DAGs are dynamic in PyTorch** An important thing to note is that the graph is recreated from scratch; after each `.backward()` call, autograd starts populating a new graph. This is exactly what allows you to use control flow statements in your model; you can change the shape, size and operations at every iteration if needed.

### Optional Reading: Tensor Gradients and Jacobian Products

In many cases, we have a scalar loss function, and we need to compute the gradient with respect to some parameters. However, there are cases when the output function is an arbitrary tensor. In this case, PyTorch allows you to compute so-called **Jacobian product**, and not the actual gradient.

For a vector function $y⃗=f(x⃗)$, where $x⃗=⟨x_1,…,x_n⟩$ and $y⃗=⟨y_1,…,y_m⟩$, a gradient of y⃗ with respect to x⃗ is given by **Jacobian matrix**:

$J=(∂y1∂x1⋯∂y1∂xn⋱∂ym∂x1⋯∂ym∂xn)J=∂x1∂y1⋮∂x1∂ym⋯⋱⋯∂xn∂y1⋮∂xn∂ym$

Instead of computing the Jacobian matrix itself, PyTorch allows you to compute **Jacobian Product** vT⋅J*vT*⋅*J* for a given input vector v=(v1…vm)*v*=(*v*1…*vm*). This is achieved by calling `backward` with v*v* as an argument. The size of v*v* should be the same as the size of the original tensor, with respect to which we want to compute the product:

```python
inp = torch.eye(4, 5, requires_grad=True)
out = (inp+1).pow(2).t()
out.backward(torch.ones_like(out), retain_graph=True)
print(f"First call\n{inp.grad}")
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nSecond call\n{inp.grad}")
inp.grad.zero_()
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nCall after zeroing gradients\n{inp.grad}")
```

Notice that when we call `backward` for the second time with the same argument, the value of the gradient is different. This happens because when doing `backward` propagation, PyTorch **accumulates the gradients**, i.e. the value of computed gradients is added to the `grad` property of all leaf nodes of computational graph. If you want to compute the proper gradients, you need to zero out the `grad` property before. In real-life training an *optimizer* helps us to do this.

**Note:** Previously we were calling `backward()` function without parameters. This is essentially equivalent to calling `backward(torch.tensor(1.0))`, which is a useful way to compute the gradients in case of a scalar-valued function, such as loss during neural network training.

## Optimizing Model Parameters

Now that we have a model and data it’s time to train, validate and test our model by optimizing its parameters on our data. Training a model is an iterative process; in each iteration the model makes a guess about the output, calculates the error in its guess (*loss*), collects the derivatives of the error with respect to its parameters (as we saw in the [previous section](https://docs.pytorch.org/tutorials/beginner/basics/autograd_tutorial.html)), and **optimizes** these parameters using gradient descent. For a more detailed walkthrough of this process, check out this video on [backpropagation from 3Blue1Brown](https://www.youtube.com/watch?v=tIeHLnjs5U8).

### Prerequisite Code

We load the code from the previous sections on [Datasets & DataLoaders](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html) and [Build Model](https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html).

```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()
```

### Hyperparameters

Hyperparameters are adjustable parameters that let you control the model optimization process. Different hyperparameter values can impact model training and convergence rates ([read more](https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html) about hyperparameter tuning)

**We define the following hyperparameters for training:**

- **Number of Epochs** - the number of times to iterate over the dataset
- **Batch Size** - the number of data samples propagated through the network before the parameters are updated
- **Learning Rate** - how much to update models parameters at each batch/epoch. Smaller values yield slow learning speed, while large values may result in unpredictable behavior during training.

```python
learning_rate = 1e-3
batch_size = 64
epochs = 5
```

### Optimization Loop

Once we set our hyperparameters, we can then train and optimize our model with an optimization loop. Each iteration of the optimization loop is called an **epoch**.

**Each epoch consists of two main parts:**

- **The Train Loop** - iterate over the training dataset and try to converge to optimal parameters.
- **The Validation/Test Loop** - iterate over the test dataset to check if model performance is improving.

Let’s briefly familiarize ourselves with some of the concepts used in the training loop. Jump ahead to see the [Full Implementation](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html#full-impl-label) of the optimization loop.

### Loss Function

When presented with some training data, our untrained network is likely not to give the correct answer. **Loss function** measures the degree of dissimilarity of obtained result to the target value, and it is the loss function that we want to minimize during training. To calculate the loss we make a prediction using the inputs of our given data sample and compare it against the true data label value.

Common loss functions include [nn.MSELoss](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html#torch.nn.MSELoss) (Mean Square Error) for regression tasks, and [nn.NLLLoss](https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html#torch.nn.NLLLoss) (Negative Log Likelihood) for classification. [nn.CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss) combines `nn.LogSoftmax` and `nn.NLLLoss`.

We pass our model’s output logits to `nn.CrossEntropyLoss`, which will normalize the logits and compute the prediction error.

```python
# Initialize the loss function
loss_fn = nn.CrossEntropyLoss()
```

### Optimizer

Optimization is the process of adjusting model parameters to reduce model error in each training step. **Optimization algorithms** define how this process is performed (in this example we use Stochastic Gradient Descent). All optimization logic is encapsulated in the `optimizer` object. Here, we use the SGD optimizer; additionally, there are many [different optimizers](https://pytorch.org/docs/stable/optim.html) available in PyTorch such as ADAM and RMSProp, that work better for different kinds of models and data.

We initialize the optimizer by registering the model’s parameters that need to be trained, and passing in the learning rate hyperparameter.

```python
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
```

**Inside the training loop, optimization happens in three steps:**

- Call `optimizer.zero_grad()` to reset the gradients of model parameters. Gradients by default add up; to prevent double-counting, we explicitly zero them at each iteration.
- Backpropagate the prediction loss with a call to `loss.backward()`. PyTorch deposits the gradients of the loss w.r.t. each parameter.
- Once we have our gradients, we call `optimizer.step()` to adjust the parameters by the gradients collected in the backward pass.

### Full Implementation

We define `train_loop` that loops over our optimization code, and `test_loop` that evaluates the model’s performance against our test data.

```python
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
```

We initialize the loss function and optimizer, and pass it to `train_loop` and `test_loop`. Feel free to increase the number of epochs to track the model’s improving performance.

```python
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")
```

## Save and Load the Model

In this section we will look at how to persist model state with saving, loading and running model predictions.

```python
import torch
import torchvision.models as models
```

### Saving and Loading Model Weights

PyTorch models store the learned parameters in an internal state dictionary, called `state_dict`. These can be persisted via the `torch.save` method:

```python
model = models.vgg16(weights='IMAGENET1K_V1')
torch.save(model.state_dict(), 'model_weights.pth')
```

To load model weights, you need to create an instance of the same model first, and then load the parameters using `load_state_dict()` method.

In the code below, we set `weights_only=True` to limit the functions executed during unpickling to only those necessary for loading weights. Using `weights_only=True` is considered a best practice when loading weights.

```python
model = models.vgg16() # we do not specify ``weights``, i.e. create untrained model
model.load_state_dict(torch.load('model_weights.pth', weights_only=True))
model.eval()
```

**Note:** be sure to call `model.eval()` method before inferencing to set the dropout and batch normalization layers to evaluation mode. Failing to do this will yield inconsistent inference results.

### Saving and Loading Models with Shapes

When loading model weights, we needed to instantiate the model class first, because the class defines the structure of a network. We might want to save the structure of this class together with the model, in which case we can pass `model` (and not `model.state_dict()`) to the saving function:

```python
torch.save(model, 'model.pth')
```

We can then load the model as demonstrated below.

As described in [Saving and loading torch.nn.Modules](https://pytorch.org/docs/main/notes/serialization.html#saving-and-loading-torch-nn-modules), saving `state_dict` is considered the best practice. However, below we use `weights_only=False` because this involves loading the model, which is a legacy use case for `torch.save`.

```python
model = torch.load('model.pth', weights_only=False)
```