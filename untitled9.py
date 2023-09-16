# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 09:51:48 2023

@author: gucenmo
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class SimpleODE(nn.Module):
    def __init__(self):
        super(SimpleODE, self).__init__()

        self.fc1 = nn.Linear(1, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.sin(self.fc1(x))
        x = torch.sin(self.fc2(x))
        x = self.fc3(x)
        return x

class PINN:
    def __init__(self):
        self.model = SimpleODE()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_func = nn.MSELoss()

    def train(self):
        x_train = np.linspace(-10, 100, 1000)[:, None]
        y_train = np.exp(-x_train)

        x_train = torch.Tensor(x_train).requires_grad_(True)
        y_train = torch.Tensor(y_train).requires_grad_(True)


        num_epochs = 10000
        for epoch in range(num_epochs):
            self.optimizer.zero_grad()
            y_pred = self.model(x_train)
            loss = self.loss_func(y_pred, y_train)
            
            # 添加微分方程的物理约束
            y_grad = torch.autograd.grad(outputs=y_pred, inputs=x_train, grad_outputs=torch.ones_like(y_pred), create_graph=True)[0]
            ode_loss = self.loss_func(y_grad + y_pred, torch.zeros_like(y_pred))
            loss += ode_loss
            
            loss.backward()
            self.optimizer.step()
            if (epoch + 1) % 1000 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, ODE Loss: {ode_loss.item():.4f}")

    def predict(self, x_test):
        with torch.no_grad():
            x_test = torch.Tensor(x_test)
            y_pred = self.model(x_test)
        return y_pred.numpy()

pinn = PINN()
pinn.train()

x_test = np.linspace(0, 10, 1000)[:, None]
y_pred = pinn.predict(x_test)

plt.figure()
plt.plot(x_test, y_pred, label='PINN')
plt.plot(x_test, np.exp(-x_test), label='EXACT')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
