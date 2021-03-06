import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from util import *

# Load data from UCI ML Repo
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None)

# Preview data
print(df.head())

print(df.tail())

# We will focus only on Iris-setosa (-1) and
# Iris-versicolor (1)
# Features: sepal length and petal length

y = df.iloc[0:100,4].values
y = np.where(y == 'Iris-setosa',-1,1)

X = df.iloc[0:100, [0,2]].values

# Plot data
plt.scatter(X[:50,0],X[:50,1],
        c='red',marker='o',label='setosa')
plt.scatter(X[50:100,0],X[50:100,1],
        c='blue',marker='x',label='versicolor')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')

plt.legend(loc = 'upper left')
plt.show()

# Train network
ppn = Perceptron(n_iter=10)
ppn.fit(X,y)
# Plot errors with respect to epochs
plt.plot(range(1, len(ppn.errors_)+1),
        ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.show()

# Plot decision boundary
plot_decision_regions(X,y,classifier=ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()

# See decision boundary wrt epochs
for n_iter in range(1,11):
    ppn = Perceptron(n_iter=n_iter)
    ppn.fit(X,y)
    plot_decision_regions(X,y,classifier=ppn)
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.title('Number of iterations = {}'.format(n_iter))
    plt.legend(loc='upper left')
    plt.show()

# Adaline GD
fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(10,4))

ada1 = AdalineGD(n_iter=10,eta=0.01).fit(X,y)
ax[0].plot(range(1,len(ada1.cost_)+1),
        np.log10(ada1.cost_),marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-Squared-Error)')
ax[0].set_title('Adaline - Learning Rate 0.01')

ada2 = AdalineGD(n_iter=100,eta=0.0001).fit(X,y)
ax[1].plot(range(1,len(ada2.cost_)+1),
        np.log10(ada2.cost_),marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('log(Sum-Squared-Error)')
ax[1].set_title('Adaline - Learning Rate 0.0001')

plt.show()

plot_decision_regions(X,y,classifier=ada1)
plt.title('Adaline - Learning Rate 0.01')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

plot_decision_regions(X,y,classifier=ada2)
plt.title('Adaline - Learning Rate 0.0001')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# Use standardization
X_std = np.copy(X)
X_std[:,0] = (X[:,0]-X[:,0].mean())/(X[:,0].std())
X_std[:,1] = (X[:,1]-X[:,1].mean())/(X[:,1].std())

# Adaline GD
ada = AdalineGD(n_iter = 15, eta=0.01)
ada.fit(X_std,y)

plot_decision_regions(X_std,y,classifier=ada)
plt.title('Adaline - Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

plt.plot(range(1,len(ada.cost_)+1),
        ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Sum-Squared-Error')
plt.show()

# Adaline SGD
# Use standardization
ada = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
ada.fit(X_std,y)

plot_decision_regions(X_std,y,classifier=ada)
plt.title('Adaline - Stochaistic Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc = 'upper left')
plt.tight_layout()
plt.show()

plt.plot(range(1,len(ada.cost_)+1),
        ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average Cost')
plt.show()

# Adaline SGD
# Do NOT use standardization
ada = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
ada.fit(X,y)

plot_decision_regions(X,y,classifier=ada)
plt.title('Adaline - Stochaistic Gradient Descent')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc = 'upper left')
plt.tight_layout()
plt.show()

plt.plot(range(1,len(ada.cost_)+1),
        ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average Cost')
plt.show()
