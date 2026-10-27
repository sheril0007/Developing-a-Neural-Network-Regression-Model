# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Explain the problem statement

## Neural Network Model
Include the neural network model diagram.

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

### Name:SHERIL.P

### Register Number:212225230262

```python
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        #Include your code here



# Initialize the Model, Loss Function, and Optimizer



def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("Exp-1.csv")
df

xt,xst,yt,yst=train_test_split(x,y,test_size=0.33,random_state=33)

scaler1=MinMaxScaler()
scaler2=MinMaxScaler()
xt=scaler1.fit_transform(xt)
xst=scaler2.fit_transform(xst)

xt1=torch.FloatTensor(xt)
xst1=torch.FloatTensor(xst)
yt1=torch.FloatTensor(yt)
yst1=torch.FloatTensor(yst)

class neuralnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network=nn.Sequential(
          nn.Linear(1,20),
          nn.ReLU(),
          nn.Linear(20,10),
          nn.ReLU(),
          nn.Linear(10,1)
        )
    def forward(self,x):
        return self.network(x)

model=neuralnet()
criterion=nn.MSELoss()
optimizer=optim.Adam(model.parameters(),lr=0.001)

epochs=1000
for i in range(epochs):
    optimizer.zero_grad()
    pred=model(xt1)
    loss=criterion(pred,yt1)
    loss.backward()
    optimizer.step()
    if i%100==0:
        print(f"Epochs:{i} Loss:",loss.item())

with torch.no_grad():
    pred=model(xst1)
    loss=criterion(pred,yst1)
    print(loss.item())

input=torch.FloatTensor(scaler1.transform([[18]]))
pred=model(input)
print(pred.item())

```

### Dataset Information
<img width="937" height="575" alt="image" src="https://github.com/user-attachments/assets/cd992908-e603-44ca-8be4-72eb50e67fd4" />


### OUTPUT
<img width="635" height="420" alt="image" src="https://github.com/user-attachments/assets/25ff1f40-db2d-4d68-babe-2c4fd490382f" />

### Training Loss Vs Iteration Plot
<img width="858" height="532" alt="image" src="https://github.com/user-attachments/assets/42ff2f42-df4d-40a5-84ad-c77c5416633a" />


### New Sample Data Prediction
<img width="588" height="511" alt="image" src="https://github.com/user-attachments/assets/0a3f0689-ec6a-4fb5-86ed-9719235105a8" />


## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
