# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## Aim:
To write a program to implement the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1. Import the packages required.
2. Read the dataset.
3. Define X and Y array.
4. Define a function for costFunction,cost and gradient.
5. Define a function to plot the decision boundary and predict the Regression value.

## Program:
```
/*
Program to implement the Logistic Regression Using Gradient Descent.
Developed by: Smriti M
RegisterNumber: 212221040157
*/
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data=np.loadtxt("ex2data1.txt",delimiter=',')
X=data[:,[0,1]]
y=data[:,2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
    return 1/(1+np.exp(-z))

plt.plot()
X_plot=np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()

def costFunction (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    grad=np.dot(X.T,h-y)/X.shape[0]
    return J,grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

def cost (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    return J

def gradient (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    grad=np.dot(X.T,h-y)/X.shape[0]
    return grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
    x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
    y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
    xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
    X_plot=np.c_[xx.ravel(),yy.ravel()]
    X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
    y_plot=np.dot(X_plot,theta).reshape(xx.shape)
    
    plt.figure()
    plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
    plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
    plt.contour(xx,yy,y_plot,levels=[0])
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend()
    plt.show()

plotDecisionBoundary(res.x,X,y)

prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta,X):
    X_train =np.hstack((np.ones((X.shape[0],1)),X))
    prob=sigmoid(np.dot(X_train,theta))
    return (prob>=0.5).astype(int)
np.mean(predict(res.x,X)==y)
```

## Output:

![s1](https://user-images.githubusercontent.com/113674204/233014484-2f3a4752-7087-4be3-9041-6dfa3f97956a.png)

![s2](https://user-images.githubusercontent.com/113674204/233014575-e8820eec-2141-40b4-8c26-60abf5e8f178.png)

![s3](https://user-images.githubusercontent.com/113674204/233014656-c3236920-6599-446a-974b-cc3bdf22fcca.png)

![s4](https://user-images.githubusercontent.com/113674204/233014755-af50c63f-01d3-43a7-9d23-46828d8681e1.png)

![s5](https://user-images.githubusercontent.com/113674204/233014821-33e19ee3-88c4-4367-b628-f62ba32d0ccc.png)

![s6](https://user-images.githubusercontent.com/113674204/233014869-88f26db7-6230-45d3-a76d-27b9517ed8ea.png)

![s7](https://user-images.githubusercontent.com/113674204/233014930-4e222536-3e94-48a7-8b98-7cb121523a82.png)

![s8](https://user-images.githubusercontent.com/113674204/233014971-ef075700-a277-44a9-bd7c-5118f6b99699.png)

![s9](https://user-images.githubusercontent.com/113674204/233015078-1ce206b5-fa20-481b-babe-88c7259dce84.png)

![s10](https://user-images.githubusercontent.com/113674204/233015128-ae0fd5c3-ef99-4c8b-86f3-32eaaec87906.png)

## Result:
Thus the program to implement the Logistic Regression Using Gradient Descent is written and verified using python programming.
