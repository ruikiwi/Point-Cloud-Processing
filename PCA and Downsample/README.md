pca_normal.py


This file implements Principal Component Analysis (PCA) and Surface Normal Estimation on point cloud data. 

**Principal Component Analysis (PCA)**: A function that takes in point cloud data, computes its covariance matrix, and returns the eigenvalues and eigenvectors.

**Surface Normal Estimation**: Built a KD-Tree with open3d, and use the neighborhood points to define the surface. The surface normal vector is the least significant eigen vector of that region. 
