# Linear-Regression-from-Scratch

- Linear regression is a popular machine learning technique that involves fitting a linear equation to a dataset of input features and corresponding target variables, with the aim of predicting the target variable for new input features. In this tutorial, we will build a linear regression model from scratch using Python and numpy, without using any pre-built libraries.

Let's start by importing the necessary libraries:
python

import numpy as np
import matplotlib.pyplot as plt
Next, we will generate some sample data to use for our linear regression model:


np.random.seed(0)
n_samples = 100
X = np.random.randn(n_samples)
y = 2*X + np.random.randn(n_samples)
In this example, we have generated 100 samples of X and corresponding target variable y using a linear equation with some random noise added.

Next, we will define our linear regression model and the loss function that we will use to optimize the model:

python

class LinearRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
    
    def fit(self, X, y):
        X = np.vstack([X, np.ones(len(X))]).T
        self.coef_, self.intercept_ = np.linalg.lstsq(X, y, rcond=None)[0]
    
    def predict(self, X):
        return self.coef_*X + self.intercept_

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)
In this code, we have defined a class called LinearRegression, which has methods for fitting the model to the data and making predictions. The fit() method uses numpy's lstsq() function to solve the linear equation and compute the coefficients and intercept of the model. The predict() method uses these coefficients to make predictions for new input features.

We have also defined a loss function called mse_loss, which computes the mean squared error between the true target variable and the predicted target variable.

Next, we will initialize an instance of our LinearRegression class and fit the model to our sample data:


model = LinearRegression()
model.fit(X, y)
Finally, we can use the model to make predictions for new input features and visualize the results:


y_pred = model.predict(X)
plt.scatter(X, y)
plt.plot(X, y_pred, color='red')
plt.show()
This code will plot the sample data as a scatter plot and the predicted values as a red line.

Overall, building a linear regression model from scratch using numpy is a straightforward process that can provide a good understanding of the underlying principles of the technique. However, in practice, it is often more efficient to use pre-built libraries such as scikit-learn or TensorFlow for larger and more complex datasets.



