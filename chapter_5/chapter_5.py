# Python code to compute the correlation coefficient
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

x = stats.multivariate_normal.rvs([0, 0], [[3, 1], [1, 1]], 10000)
plt.figure();
plt.scatter(x[:, 0], x[:, 1])
rho, _ = stats.pearsonr(x[:, 0], x[:, 1])
print(rho)

# Python code to compute a mean vector
import numpy as np
import scipy.stats as stats

X = stats.multivariate_normal.rvs([0, 0], [[1, 0], [0, 1]], 100)
mX = np.mean(X, axis=1)

# Python code to compute covariance matrix
import numpy as np
import scipy.stats as stats

X = stats.multivariate_normal.rvs([0, 0], [[1, 0], [0, 1]], 100)
covX = np.cov(X, rowvar=False)
print(covX)

# Python code to generate random numbers from multivariate Gaussian
import numpy as np
import scipy.stats as stats

X = stats.multivariate_normal.rvs([0, 0], [[0.25, 0.3], [0.3, 1.0]], 100)

# Python code: Overlay random numbers with the Gaussian contour.
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

X = stats.multivariate_normal.rvs([0, 0], [[0.25, 0.3], [0.3, 1.0]], 1000)
x1 = np.arange(-2.5, 2.5, 0.01)
x2 = np.arange(-3.5, 3.5, 0.01)
X1, X2 = np.meshgrid(x1, x2)
Xpos = np.empty(X1.shape + (2,))
Xpos[:, :, 0] = X1
Xpos[:, :, 1] = X2
F = stats.multivariate_normal.pdf(Xpos, [0, 0], [[0.25, 0.3], [0.3, 1.0]])
plt.scatter(X[:, 0], X[:, 1])
plt.contour(x1, x2, F)

# Python Code to perform eigendecomposition
import numpy as np

A = np.random.randn(100, 100)
A = (A + np.transpose(A)) / 2
S, U = np.linalg.eig(A)
s = np.diag(S)

# Python code to perform the whitening
import numpy as np
import scipy.stats as stats
from scipy.linalg import fractional_matrix_power

x = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 1000)
mu = np.array([1, -2])
Sigma = np.array([[3, -0.5], [-0.5, 1]])
Sigma2 = fractional_matrix_power(Sigma, 0.5)
y = np.dot(Sigma2, x.T) + np.matlib.repmat(mu, 1000, 1).T

# Python code to perform whitening
import numpy as np
import scipy.stats as stats
from scipy.linalg import fractional_matrix_power

y = np.random.multivariate_normal([1, -2], [[3, -0.5], [-0.5, 1]], 100)
mY = np.mean(y, axis=0)
covY = np.cov(y, rowvar=False)
covY2 = fractional_matrix_power(covY, -0.5)
x = np.dot(covY2, (y - np.matlib.repmat(mY, 100, 1)).T)

# Python code to perform the principal-component analysis
import numpy as np

x = np.random.multivariate_normal([1, -2], [[3, -0.5], [-0.5, 1]], 1000)
covX = np.cov(x, rowvar=False)
S, U = np.linalg.eig(covX)
print(U)
