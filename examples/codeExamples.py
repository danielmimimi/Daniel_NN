import sqlite3
import numpy as np
from numpy import ndarray
from typing import List,Callable
import matplotlib.pyplot as plt

def deriv(func: Callable[[ndarray], ndarray],
    input_: ndarray,
    delta: float = 0.001) -> ndarray:
    '''
    Evaluates the derivative of a function "func" at every element in the
    "input_" array.
    '''
    return (func(input_ + delta) - func(input_ - delta)) / (2 * delta)

def square(x: ndarray) -> ndarray:
    '''
    Square each element in the input ndarray.
    '''
    return np.power(x, 2)

def leaky_relu(x: ndarray) -> ndarray:
    '''
    Apply "Leaky ReLU" function to each element in ndarray.
    '''
    return np.maximum(0.2 * x, x)

def sigmoid(x: ndarray) -> ndarray:
    '''
    Apply the sigmoid function to each element in the input ndarray.
    '''
    return 1 / (1 + np.exp(-x))



print(square(3))
print(leaky_relu(-0.3))

Array_Function = Callable[[ndarray], ndarray]
# A Chain is a list of functions
Chain = List[Array_Function]


def chain_deriv_2(chain: Chain,
    input_range: ndarray) -> ndarray:
    '''
    Uses the chain rule to compute the derivative of two nested functions:
    (f2(f1(x))' = f2'(f1(x)) * f1'(x)
    '''
    assert len(chain) == 2, \
    "This function requires 'Chain' objects of length 2"
    assert input_range.ndim == 1, \
    "Function requires a 1 dimensional ndarray as input_range"
    f1 = chain[0]
    f2 = chain[1]
    # df1/dx
    f1_of_x = f1(input_range)
    # df1/du
    df1dx = deriv(f1, input_range)
    # df2/du(f1(x))
    df2du = deriv(f2, f1(input_range))
    # Multiplying these quantities together at each point
    return df1dx * df2du

chain_1 = [square,sigmoid]
chain_2 = [sigmoid,square]

PLOT_RANGE = np.arange(-3, 3, 0.01)

plt.figure(1)
plt.plot(PLOT_RANGE,chain_1[1](chain_1[0](PLOT_RANGE)))
plt.plot(PLOT_RANGE,chain_deriv_2(chain_1,PLOT_RANGE))


plt.figure(2)
plt.plot(PLOT_RANGE,chain_2[1](chain_2[0](PLOT_RANGE)))
plt.plot(PLOT_RANGE,chain_deriv_2(chain_2,PLOT_RANGE))

print(11)



def multiple_inputs_add(x: ndarray,
                        y: ndarray,
                        sigma: Array_Function) -> float:
    '''
    Function with multiple inputs and addition, forward pass.
    '''
    assert x.shape == y.shape
    a = x + y
    return sigma(a)



def matmul_forward(X: ndarray,
W: ndarray) -> ndarray:
    '''
    Computes the forward pass of a matrix multiplication.
    '''
    assert X.shape[1] == W.shape[0], \
    '''
    For matrix multiplication, the number of columns in the first array should
    match the number of rows in the second; instead the number of columns in the
    first array is {0} and the number of rows in the second array is {1}.
    '''.format(X.shape[1], W.shape[0])
    # matrix multiplication
    N = np.dot(X, W)
    return N

def matmul_backward_first(X: ndarray,
W: ndarray) -> ndarray:
    '''
    Computes the backward pass of a matrix multiplication with respect to the
    first argument.
    '''
    # backward pass
    dNdX = np.transpose(W, (1, 0))
    return dNdX

def matrix_forward_extra(X: ndarray,
W: ndarray,
sigma: Array_Function) -> ndarray:
    '''
    Computes the forward pass of a function involving matrix multiplication,
    one extra function.
    '''
    assert X.shape[1] == W.shape[0]
    # matrix multiplication
    N = np.dot(X, W)
    # feeding the output of the matrix multiplication through sigma
    S = sigma(N)
    return S

def matrix_function_backward_1(X: ndarray,
    W: ndarray,
    sigma: Array_Function) -> ndarray:
    '''
    Computes the derivative of our matrix function with respect to
    the first element.
    '''
    assert X.shape[1] == W.shape[0]
    # matrix multiplication
    N = np.dot(X, W)
    # feeding the output of the matrix multiplication through sigma
    S = sigma(N)
    # backward calculation
    dSdN = deriv(sigma, N)
    # dNdX
    dNdX = np.transpose(W, (1, 0))
    # multiply them together; since dNdX is 1x1 here, order doesn't matter
    return np.dot(dSdN, dNdX)

np.random.seed(190203)

X = np.random.randn(1,3)
W = np.random.randn(3,1)

# GRADIENT CHECKING - WTF - really dependent on nput not on function applied??
print(X)
print(matrix_function_backward_1(X, W, sigmoid))
X_ = X
X_[0,2] = X_[0,2] +0.01
print(X_)
print(matrix_function_backward_1(X_, W, sigmoid))
print(',,,')





def matrix_function_forward_sum(X: ndarray,
                                W: ndarray,
                                sigma: Array_Function) -> float:
    '''
    Computing the result of the forward pass of this function with
    input ndarrays X and W and function sigma.
    '''
    assert X.shape[1] == W.shape[0]

    # matrix multiplication
    N = np.dot(X, W)

    # feeding the output of the matrix multiplication through sigma
    S = sigma(N)

    # sum all the elements
    L = np.sum(S)

    return L

def matrix_function_backward_sum_1(X: ndarray,
                                   W: ndarray,
                                   sigma: Array_Function) -> ndarray:
    '''
    Compute derivative of matrix function with a sum with respect to the
    first matrix input
    '''
    assert X.shape[1] == W.shape[0]

    # matrix multiplication
    N = np.dot(X, W)

    # feeding the output of the matrix multiplication through sigma
    S = sigma(N)

    # sum all the elements
    L = np.sum(S)

    # note: I'll refer to the derivatives by their quantities here,
    # unlike the math where we referred to their function names

    # dLdS - just 1s
    dLdS = np.ones_like(S)

    # dSdN
    dSdN = deriv(sigma, N)
    
    # dLdN
    dLdN = dLdS * dSdN

    # dNdX
    dNdX = np.transpose(W, (1, 0))

    # dLdX
    dLdX = np.dot(dSdN, dNdX)

    return dLdX

np.random.seed(190204)
X = np.random.randn(3, 3)
W = np.random.randn(3, 2)

print("X:")
print(X)

print("L:")
print(round(matrix_function_forward_sum(X, W, sigmoid), 4))
print()
print("dLdX:")
print(matrix_function_backward_sum_1(X, W , sigmoid))

X1 = X.copy()
X1[0, 0] += 0.001

print(round(
        (matrix_function_forward_sum(X1, W, sigmoid) - \
         matrix_function_forward_sum(X, W, sigmoid)) / 0.001, 4))
