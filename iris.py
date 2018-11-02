import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from perceptron import *
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
