# 1) Design model (input, output size, forward pass)
# 2) Construct loss and optimizer
# 3) Training loop
#   - forward pass: compute prediction
#   - backward pass: gradients
#   - update weights

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 0) Prepare data
breast_cancer = datasets.load_breast_cancer()
X, y = breast_cancer.data, breast_cancer.target

n_samples, n_features = X.shape
print(f"Number of samples: {n_samples} and features: {n_features} ")

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.2, random_state=1234)

# scale

sc = StandardScaler() #scale data with 0 mean and unit varience to deal with logistic regression
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0],1)
y_test = y_test.view(y_test.shape[0],1)

# 1) model

# f = wx + b, sigmoid at end

class Logistic_Regression(nn.Module):

    def __init__(self, n_input_features):
        super(Logistic_Regression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self,x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted

model = Logistic_Regression(n_features)

# 2) loss and optimizer
learning_rate = 0.01
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr= learning_rate)

# 3) training loop
num_epochs = 2000
for epoch in range(num_epochs):
    # forward pass and loss
    y_predicted = model(X_train)
    loss = criterion(y_predicted, y_train)

    # backward pass
    loss.backward()

    # updateS
    optimizer.step()

    # zero gradients,   # backward function is always add up all the gradients into the .grad() attribute
    optimizer.zero_grad()

    if (epoch+1) % 100 == 0:
        print(f'epoch:{epoch+1}, loss = {loss.item():.4f}')


# evaluation 
with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f'\nAccuracy = {acc:.4f}')
