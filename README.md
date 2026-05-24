# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY

### Problem Statement

We want to build a regression model using a feedforward neural network. The task is to learn the mapping between input features and continuous target values. Unlike classification problems (where outputs are discrete classes), regression problems aim to predict real-valued outcomes, such as prices, measurements, or any other numerical data.

##### Neural Network Model Explanation
#### 1. Architecture

The model is a fully connected feedforward neural network implemented in PyTorch. It consists of:

Input layer (1 neuron) → accepts a single feature as input.

Hidden Layer 1 (6 neurons) → fully connected layer with ReLU activation.

Hidden Layer 2 (12 neurons) → fully connected layer with ReLU activation.

Hidden Layer 3 (20 neurons) → fully connected layer with ReLU activation.

Output Layer (1 neuron) → produces the final prediction (continuous value).

#### The flow of data is:

Input → Linear(1→6) → ReLU → Linear(6→12) → ReLU 
      → Linear(12→20) → ReLU → Linear(20→1) → Output

#### 2. Activation Function

ReLU (Rectified Linear Unit) is used in the hidden layers.

ReLU introduces non-linearity, helping the network learn complex patterns.

It avoids issues like the vanishing gradient problem that can occur with sigmoid/tanh.

#### 3. Loss Function

Mean Squared Error (MSELoss) is used.

MSE is the standard loss for regression problems.

It penalizes larger errors more strongly, encouraging the model to minimize prediction errors.

#### 4. Optimizer

RMSprop (Root Mean Square Propagation) is chosen as the optimizer.

RMSprop adapts the learning rate for each parameter individually, which makes it more stable.

It is widely used for training deep neural networks when data is noisy or gradients fluctuate.

#### 5. Training History

The model also keeps track of training loss values in self.history['loss'], which can be useful for visualization and analysis after training.

Theory Behind the Model

Neural networks are universal function approximators, meaning they can model complex relationships between inputs and outputs if given enough neurons and layers.

Linear Layers (nn.Linear) perform weighted sums of inputs plus a bias.

Mathematically: 
y=Wx+b

where 
W is the weight matrix, 
x is input, and 
b is bias.

##### ReLU Activation introduces non-linearity:
ReLU(x)=max(0,x)

##### Backpropagation & Gradient Descent:

The model learns by minimizing the loss function.

Gradients of the loss w.r.t. weights are computed using backpropagation.

Optimizer updates the weights to reduce error iteratively.

## Neural Network Model

<img width="1071" height="756" alt="image" src="https://github.com/user-attachments/assets/71da9f99-d098-4be1-8e30-b724db9ce93f" />

## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name:T MOUNISH

### Register Number:212223240098

```
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

dataset1 = pd.read_csv('SampleSheet.csv')
X = dataset1[['Input']].values
y = dataset1[['Output']].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=33)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Name:T Mounish
# Register Number:212223240098
class NeuralNet(nn.Module):
  def init(self):
        super().init()
        self.fc1 = nn.Linear(1,8)
        self.fc2 = nn.Linear(8,10)
        self.fc3 = nn.Linear(10,1)
        self.relu = nn.ReLU()
        self.history = {'loss':[]}
  
  def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the Model, Loss Function, and Optimizer
ai_brain = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(ai_brain.parameters(), lr=0.001)

# Name:T Mounish
# Register Number:212223240098
def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
  for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(ai_brain(X_train), y_train)
        loss.backward()
        optimizer.step()

        ai_brain.history['loss'].append(loss.item())
        
        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')

train_model(ai_brain, X_train_tensor, y_train_tensor, criterion, optimizer)

with torch.no_grad():
    test_loss = criterion(ai_brain(X_test_tensor), y_test_tensor)
    print(f'Test Loss: {test_loss.item():.6f}')

import matplotlib.pyplot as plt
loss_df.plot()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training")
plt.show()

X_n1_1 = torch.tensor([[9]], dtype=torch.float32)
prediction = ai_brain(torch.tensor(scaler.transform(X_n1_1), dtype=torch.float32)).item()
print(f'Prediction: {prediction}')


```

### Dataset Information
### OUTPUT
<img width="426" height="57" alt="image" src="https://github.com/user-attachments/assets/10b7c100-3871-4608-8016-aeb835f860a0" />



### Training Loss Vs Iteration Plot
<img width="580" height="455" alt="image" src="https://github.com/user-attachments/assets/ca09acdf-fc88-42a4-898e-2afa9c3c33e7" />


### New Sample Data Prediction

<img width="502" height="50" alt="image" src="https://github.com/user-attachments/assets/91f93b46-c0be-4ce5-8ad1-40f446c03b42" />


## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.ssfully developed and trained using PyTorch.
