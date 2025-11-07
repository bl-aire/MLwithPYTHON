import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import pandas as pd



# Numpy multidimensional arrays
x = np.array([[1,2,3], [4,5,6]])

print("x:\n{}".format(x))
print(f"x:\n{x}")



# Create a 2D identity matrix(square matrix with ones on the main diagonal and zeroes elsewhere)
eye = np.eye(4)
print(f"eye:\n{eye}")



# Convert identity matrix to scipy sparse matrix in CSR format
sparse_matrix = sparse.csr_matrix(eye)
print(f"sparse_matrix:\n{sparse_matrix}")



# Create sparse representation directly via the Coordinate format (4 rows and 4 columns)
data = np.ones(4)
row_indices = np.arange(4)
col_indices = np.arange(4)
eye_coo = sparse.coo_matrix((data, (row_indices, col_indices)), shape=(4, 4))
print(f"COO:\n{eye_coo}")



# Create sparse representation by setting the coordinates for non-zero numbers
rows = np.array([0, 0, 1, 2, 2])
cols = np.array([0, 2, 1, 0, 2])
vals = np.array([1, 2, 3, 4, 5])

coo_matrix = sparse.csr_matrix((vals, (rows, cols)), shape=(3, 3))
print(f"coo_matrix:\n{coo_matrix}")



# Convert to CSR or CSC
csr_matrix = coo_matrix.tocsr()
print(f"csr_matrix:\n{csr_matrix}")

csc_matrix = coo_matrix.tocsc()
print(f"csc_matrix:\n{csc_matrix}")



# Matplotlib

x = np.linspace(-10, 10, 100)
y = np.sin(x)
plt.plot(x, y, marker='o')
plt.show()


# Pandas

data = {
    "Name" : ["Blessing", "Alpha"],
    "Age" : [20, 25],
    "Location" : ["London", "Paris"]
}

data_pandas = pd.DataFrame(data)
print(f"data_pandas:\n{data_pandas}")

