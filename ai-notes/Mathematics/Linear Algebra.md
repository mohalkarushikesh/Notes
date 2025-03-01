# Linear Algebra: Vectors, Matrices, Eigenvalues/Eigenvectors

Linear algebra is foundational in data representation and neural networks. Let’s dive into vectors, matrices, and eigenvalues/eigenvectors, exploring their mathematics and applications.

---

## Vectors: The Building Blocks

A vector is an ordered collection of numbers, typically represented as a column or row in \( \mathbb{R}^n \), where \( n \) is the dimensionality. For example, a 3D vector might be:

![Vector Example](https://latex.codecogs.com/png.latex?\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ v_3 \end{bmatrix})

### Mathematics
- **Operations**: Vectors support addition (\( \mathbf{u} + \mathbf{v} \)), scalar multiplication (\( c \mathbf{v} \)), and the dot product:
  ![Dot Product](https://latex.codecogs.com/png.latex?\mathbf{u} \cdot \mathbf{v} = u_1 v_1 + \cdots + u_n v_n)
  which measures similarity or projection.
- **Norm**: The length of a vector:
  ![Norm](https://latex.codecogs.com/png.latex?||\mathbf{v}|| = \sqrt{v_1^2 + \cdots + v_n^2})
  is key for normalization (e.g., unit vectors).

### In Data Representation
- **Feature Vectors**: Data points are vectors, e.g., a house as:
  ![Feature Vector](https://latex.codecogs.com/png.latex?\mathbf{x} = \begin{bmatrix} \text{size} \\ \text{price} \\ \text{rooms} \end{bmatrix})
- **High Dimensions**: A 256x256 RGB image flattens to a 196,608-dimensional vector (256 × 256 × 3).

### In Neural Networks
- **Input Layer**: Input is a vector \( \mathbf{x} \), e.g., \( \mathbb{R}^{784} \) for MNIST digits.
- **Activations**: Layers produce output vectors via transformations.
- **Gradients**: Backpropagation uses gradient vectors, e.g., \( \frac{\partial L}{\partial \mathbf{w}} \).

---

## Matrices: The Transformers

A matrix is a 2D array, \( A \in \mathbb{R}^{m \times n} \), acting as a linear transformation from \( \mathbb{R}^n \) to \( \mathbb{R}^m \).

### Mathematics
- **Matrix Multiplication**: For \( A \in \mathbb{R}^{m \times n} \) and \( \mathbf{x} \in \mathbb{R}^n \):
  ![Matrix Multiplication](https://latex.codecogs.com/png.latex?A \mathbf{x} \in \mathbb{R}^m)
  computes a weighted combination.
- **Transpose**: \( A^T \) flips rows and columns.
- **Inverse**: For square matrices, \( A A^{-1} = I \) (identity matrix).

### In Data Representation
- **Datasets**: An \( m \times n \) matrix holds \( m \) samples with \( n \) features.
- **Covariance Matrix**: For dataset \( X \):
  ![Covariance Matrix](https://latex.codecogs.com/png.latex?C = \frac{1}{m-1} X^T X)
  captures feature correlations.

### In Neural Networks
- **Weight Matrices**: A layer’s weights \( W \in \mathbb{R}^{m \times n} \) compute:
  ![Layer Output](https://latex.codecogs.com/png.latex?\mathbf{y} = W \mathbf{x} + \mathbf{b})
- **Backpropagation**: Gradient:
  ![Gradient](https://latex.codecogs.com/png.latex?\frac{\partial L}{\partial W} = \frac{\partial L}{\partial \mathbf{y}} \mathbf{x}^T)

**Example**: For \( \mathbf{x} = \begin{bmatrix} 1 \\ 2 \end{bmatrix} \), \( W = \begin{bmatrix} 3 & 4 \\ 5 & 6 \end{bmatrix} \), \( \mathbf{b} = \begin{bmatrix} 0 \\ 1 \end{bmatrix} \):
![Example Computation](https://latex.codecogs.com/png.latex?\mathbf{y} = \begin{bmatrix} 3 & 4 \\ 5 & 6 \end{bmatrix} \begin{bmatrix} 1 \\ 2 \end{bmatrix} + \begin{bmatrix} 0 \\ 1 \end{bmatrix} = \begin{bmatrix} 11 \\ 17 \end{bmatrix})

---

## Eigenvalues and Eigenvectors: The Essence of Structure

For a square matrix \( A \in \mathbb{R}^{n \times n} \), an eigenvector \( \mathbf{v} \neq 0 \) and eigenvalue \( \lambda \) satisfy:
![Eigen Equation](https://latex.codecogs.com/png.latex?A \mathbf{v} = \lambda \mathbf{v})

### Mathematics
- **Characteristic Equation**: Solve:
  ![Characteristic Equation](https://latex.codecogs.com/png.latex?\det(A - \lambda I) = 0)
- **Diagonalization**: If \( A \) has \( n \) independent eigenvectors:
  ![Diagonalization](https://latex.codecogs.com/png.latex?A = P D P^{-1})

**Example**: For \( A = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix} \):
- Polynomial: 
  ![Polynomial](https://latex.codecogs.com/png.latex?\det\begin{bmatrix} 2-\lambda & 1 \\ 1 & 2-\lambda \end{bmatrix} = (2-\lambda)^2 - 1 = \lambda^2 - 4\lambda + 3 = 0)
- Eigenvalues: \( \lambda = 3, 1 \).
- Eigenvectors: For \( \lambda = 3 \):
  ![Eigenvector 1](https://latex.codecogs.com/png.latex?\begin{bmatrix} -1 & 1 \\ 1 & -1 \end{bmatrix} \mathbf{v} = 0, \quad \mathbf{v} = \begin{bmatrix} 1 \\ 1 \end{bmatrix})
  For \( \lambda = 1 \):
  ![Eigenvector 2](https://latex.codecogs.com/png.latex?\mathbf{v} = \begin{bmatrix} 1 \\ -1 \end{bmatrix})

### In Data Representation
- **PCA**: Eigenvectors of \( C = \frac{1}{m-1} X^T X \) are principal components; eigenvalues are variances.
- **Graph Analysis**: Eigenvectors of adjacency matrices reveal structure.

### In Neural Networks
- **Optimization**: Hessian eigenvalues indicate loss curvature.
- **RNN Stability**: \( |\lambda_{\text{max}}| \) of \( W \) affects gradient behavior.
- **Initialization**: Xavier scaling keeps eigenvalues near 1.

**Example in PCA**: Project data with \( C = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix} \) onto \( \begin{bmatrix} 1 \\ 1 \end{bmatrix} \) (eigenvalue 3).

---

## Connecting the Dots
- **Forward Pass**: \( \mathbf{y} = W \mathbf{x} \); eigenvalues influence behavior.
- **Learning**: Gradients update \( W \), informed by data structure (e.g., PCA).
- **Applications**: From compression to dynamics, these concepts shine.

Want a specific example, like backpropagation or eigenvalue regularization?
