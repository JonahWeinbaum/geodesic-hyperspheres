import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

class HypersphereSVM:

    #Initializes Soft SVM
    def __init__(self, eta1=0.3, eta2=0.3, max_iters=100000, learning_rate=0.01):
        #In R^n
        self.a1 = np.array([-1., -1.])
        self.a2 = np.array([1., 1.])

        #In R
        self.r1 = 1.
        self.r2 = 1.
        
        self.eta1 = eta1
        self.eta2 = eta2

        self.learning_rate = learning_rate
        self.max_iters = max_iters

    #Performs Gradient Descent to Hypersphere SVM
    def fit(self, X, Y):
        I_plus = X[Y == 1]
        I_minus = X[Y == -1]

        for _ in range(self.max_iters):
           grad_a1 = (-2 * (self.a1 - self.a2)) + self.eta2*np.sum(2*(np.where([(np.sum((x_i-self.a1)**2)-self.r1**2 <= 0) for x_i in I_plus], I_plus, 0)-self.a1), axis=0)
           grad_a2 = (-2 * (self.a1 - self.a2)) + self.eta2*np.sum(2*(np.where([(np.sum((x_j-self.a2)**2)-self.r2**2 <= 0) for x_j in I_minus], I_minus, 0)-self.a2), axis=0)
           grad_r1 = 2*self.eta1*self.r1 + self.eta2*np.sum(-2*(np.where([(np.sum((x_i-self.a1)**2)-self.r1**2 <= 0) for x_i in I_plus],self.r1, 0)), axis=0)
           grad_r2 = 2*self.eta1*self.r2 + self.eta2*np.sum(-2*(np.where([(np.sum((x_j-self.a2)**2)-self.r2**2 <= 0) for x_j in I_minus],self.r2, 0)), axis=0)

           self.a1 = self.a1 - self.learning_rate*grad_a1
           self.a2 = self.a2 - self.learning_rate*grad_a2
           self.r1 = self.r1 - self.learning_rate*grad_r1
           self.r2 = self.r2 - self.learning_rate*grad_r2

           print(self.loss(X, Y))

    def loss(self, X, Y):
        I_plus = X[Y == 1]
        I_minus = X[Y == -1]
        v1 = -np.sum((self.a1-self.a2)**2)
        v2 = self.eta2*np.sum([max(0,(np.sum((x_i-self.a1)**2)-self.r1**2)) for x_i in I_plus])
        v3 = self.eta2*np.sum([max(0,(np.sum((x_j-self.a2)**2)-self.r2**2)) for x_j in I_minus])
        return v1 + v2 + v3
    
    
    #Computes SVM Prediction
    def predict(self, X):
        mat_a1 = np.tile(self.a1, (X.shape[0], 1))
        mat_a2 = np.tile(self.a2, (X.shape[0], 1))

        p1 = np.sum((mat_a1 - X)**2, axis=1) / self.r1**2
        p2 = np.sum((mat_a2 - X)**2, axis=1) / self.r2**2


        prediction = p1 - p2
       # prediction = np.where(prediction == 0, -1, 1)

        return prediction
        
# Generate some data for demonstration
X = np.array([[1., 2.], [2., 3.], [3., 1.], [4., 2.]])
Y = np.array([1., 1., -1., -1.])

svm = HypersphereSVM();
svm.fit(X, Y)
# # Plot data points
# plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)

# # Define plot limits
# xmin, xmax, ymin, ymax = plt.axis()

# # Define plot limits
# xmin, xmax = X[:, 0].min() - 2, X[:, 0].max() + 2
# ymin, ymax = X[:, 1].min() - 2, X[:, 1].max() + 2

# # Create mesh grid
# xx, yy = np.meshgrid(np.linspace(xmin, xmax, 100),
#                      np.linspace(ymin, ymax, 100
#                                  ))
# # Predict decision boundary
# Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)


# plt.contourf(xx, yy, Z, alpha = 0.3)
# plt.contour(xx, yy, Z, levels=[-1, 0, 1], colors='k', linestyles=['dotted', 'solid', 'dashed'])

# # Plot margin lines
# for c in [-1, 0, 1]:
#     plt.scatter([], [], color='k', linewidth=2)
#     plt.contour(xx, yy, Z, levels=[c], colors='k', linestyles=['dotted' if c == -1 else 'dashed'])

# # Plot the circle representing the SVM's hypersphere
# c1 = Circle(svm.a1, svm.r1, color='blue', fill=False, linewidth=2, label="Hypersphere")
# plt.gca().add_artist(c1)

# c2 = Circle(svm.a2, svm.r2, color='red', fill=False, linewidth=2, label="Hypersphere")
# plt.gca().add_artist(c2)


# plt.gca().set_aspect('equal', adjustable='box')

# plt.xlabel('X1')
# plt.ylabel('X2')
# plt.title('SVM Decision Boundary')
# plt.show() 
