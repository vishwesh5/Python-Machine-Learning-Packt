# Visualize decision boundaries for two-dimensional dataset
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt

# Perceptron class

class Perceptron(object):
    """Perceptron classifier.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    random_state : int
      Random number generator seed for random weight
      initialization.

    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
    errors_ : list
      Number of misclassifications (updates) in each epoch.

    """
    def __init__(self, eta=0.01,
            n_iter=50, random_state=1):
        # Scalar
        self.eta = eta
        # Scalar
        self.n_iter = n_iter
        # Scalar
        self.random_state = random_state
        #print("[INFO] Initialized parameters")

    def fit(self, X, y):
        """Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
          Training vectors, where n_samples is the number of
          samples and
          n_features is the number of features.
        y : array-like, shape = [n_samples]
          Target values.

        Returns
        -------
        self : object

        """
        #print("[INFO] Fitting values")
        rgen = np.random.RandomState(self.random_state)
        # w_ Shape: (1+n_features)
        # w_[0] = bias
        # X Shape: (n_samples, n_features)
        # y Shape: (n_samples)
        self.w_ = rgen.normal(loc=0.0,scale=0.01,
                size=1+X.shape[1])
        self.errors_ = []

        for i in range(self.n_iter):
            errors = 0
            for xi,target in zip(X,y):
                update = self.eta * (target-self.predict(xi))
                self.w_[1:] += update*xi
                # bias
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self,X):
        """Calculate net input"""
        #print("[INFO] Calculating net input")
        # (n_samples, n_features).(n_features)
        # -> (n_samples)
        # Returns: n_samples
        # wT X + b
        return np.dot(X, self.w_[1:])+self.w_[0]

    def predict(self,X):
        """Return class label after unit step"""
        #print("[INFO] Predicting class label after unit step")
        return np.where(self.net_input(X) >= 0.0, 1, -1)

# Adaline Class with Gradient Descent

class AdalineGD(object):
    """ADAptive LInear NEuron classifier.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    random_state : int
      Random number generator seed for random weight
      initialization.


    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
    cost_ : list
      Sum-of-squares cost function value in each epoch.

    """
    def __init__(self, eta=0.01,
            n_iter=50,
            random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """ Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
          Training vectors, where n_samples is the number of
          samples and
          n_features is the number of features.
        y : array-like, shape = [n_samples]
          Target values.

        Returns
        -------
        self : object

        """
        # Shapes:
        # w_ -> (1+n_features); w_[0] = bias
        # X  -> (n_samples,n_features)
        # y  -> (n_samples)
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0,scale=0.01,
                size = 1+X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            # net_input -> (n_samples)
            net_input = self.net_input(X)
            # output -> (n_samples)
            output = self.activation(net_input)
            # errors -> (n_samples)
            errors = (y-output)
            # X.T -> (n_features,n_samples)
            # X.T.dot(errors) -> (n_features)
            self.w_[1:] += self.eta * X.T.dot(errors)
            # errors.sum() -> 1
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self,X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:])+self.w_[0]

    def activation(self,X):
        """Compute linear activation"""
        return X

    def predict(self,X):
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(X))>= 0.0,
                1, -1)

# Adaline SGD
class AdalineSGD(object):
    """ADAptive LInear NEuron classifier.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    shuffle : bool (default: True)
      Shuffles training data every epoch if True
      to prevent cycles.
    random_state : int
      Random number generator seed for random weight
      initialization.


    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
    cost_ : list
      Sum-of-squares cost function value averaged over all
      training samples in each epoch.


    """
    def __init__(self,eta=0.01,n_iter=10,
            shuffle=True,random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle=shuffle
        self.random_state=random_state

    def fit(self,X,y):
        """ Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
          Training vectors, where n_samples is the number
          of samples and
          n_features is the number of features.
        y : array-like, shape = [n_samples]
          Target values.

        Returns
        -------
        self : object

        """
        # n_features
        self._initialize_weights(X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                X,y = self._shuffle(X,y)
            cost = []
            for xi,target in zip(X,y):
                cost.append(self._update_weights(xi,target))
            avg_cost = sum(cost)/len(y)
            self.cost_.append(avg_cost)
        return self

    def partial_fit(self,X,y):
        """Fit training data without reinitializing the weights"""
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        # If n_samples > 1
        if y.ravel().shape[0] > 1:
            for xi,target in zip(X,y):
                self._update_weights(X,y)
        else:
            self._update_weights(X,y)
        return self

    def _shuffle(self,X,y):
        """Shuffle training data"""
        r = self.rgen.permutation(len(y))
        return X[r],y[r]

    def _initialize_weights(self,m):
        """Initialize weights to small random samples"""
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01,
                size=1+m)
        self.w_initialized = True

    def _update_weights(self,xi,target):
        """Apply Adaline learning rule to update the weights"""
        output = self.activation(self.net_input(xi))
        error = (target-output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error**2
        return cost

    def net_input(self,X):
        """Calculate net input"""
        return np.dot(X,self.w_[1:])+self.w_[0]

    def activation(self,X):
        """Compute linear activation"""
        return X

    def predict(self,X):
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)

def plot_decision_regions(X,y,classifier, resolution=0.02):

    # Setup marker generator and color map
    markers = ('s','x','o','^','v')
    colors = ('red','blue','lightgreen','gray','cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot decision surface
    x1_min,x1_max = X[:,0].min()-1,X[:,0].max()+1
    x2_min,x2_max = X[:,1].min()-1,X[:,1].max()+1
    xx1, xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution),
            np.arange(x2_min,x2_max,resolution))
    
    Z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1,xx2,Z,alpha=0.3,cmap=cmap)
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())

    # plot class samples
    for idx,cl in enumerate(np.unique(y)):
        plt.scatter(x = X[y==cl,0],
                y = X[y==cl,1],
                alpha=0.8,
                c=colors[idx],
                marker = markers[idx],
                label=cl,
                edgecolor='black')

