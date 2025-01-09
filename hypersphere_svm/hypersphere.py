import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

class HypersphereSVM:

    #Initializes Soft SVM
    def __init__(self):
        #In R^n
        self.a1 = np.array([2., 2.])
        self.a2 = np.array([4., 2.5])

        #In R
        self.r1 = 1
    
        self.r2 = 0.6
        self.c1 = 0
        self.c2 = 0

    #Performs Gradient Descent to Hypersphere SVM
    def fit(self, X, Y):
        return

    def loss(self, X, Y):
        v1 = -np.sum(X**2, axis = 1)
        v2 = self.c1*(self.r1**2 + self.r2**2)
        v3 = self.c2*(np.max(0
    
    
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

# Plot data points
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)

# Define plot limits
xmin, xmax, ymin, ymax = plt.axis()

# Define plot limits
xmin, xmax = X[:, 0].min() - 2, X[:, 0].max() + 2
ymin, ymax = X[:, 1].min() - 2, X[:, 1].max() + 2

# Create mesh grid
xx, yy = np.meshgrid(np.linspace(xmin, xmax, 1000),
                     np.linspace(ymin, ymax, 1000
                                 ))
# Predict decision boundary
Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)


plt.contourf(xx, yy, Z, alpha = 0.3)
plt.contour(xx, yy, Z, levels=[-1, 0, 1], colors='k', linestyles=['dotted', 'solid', 'dashed'])

# Plot margin lines
for c in [-1, 0, 1]:
    plt.scatter([], [], color='k', linewidth=2)
    plt.contour(xx, yy, Z, levels=[c], colors='k', linestyles=['dotted' if c == -1 else 'dashed'])

# Plot the circle representing the SVM's hypersphere
c1 = Circle(svm.a1, svm.r1, color='blue', fill=False, linewidth=2, label="Hypersphere")
plt.gca().add_artist(c1)

c2 = Circle(svm.a2, svm.r2, color='red', fill=False, linewidth=2, label="Hypersphere")
plt.gca().add_artist(c2)


plt.gca().set_aspect('equal', adjustable='box')

plt.xlabel('X1')
plt.ylabel('X2')
plt.title('SVM Decision Boundary')
plt.show()
