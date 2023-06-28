import numpy as np
from helpers import (
    batch_iter,
    compute_gradient,
    mse_loss,
    gradient_logistic,
    loss_logistic,
)

np.random.seed(0)


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent on MSE

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        initial_w: numpy array of shape=(D, ). The initial guess (or the initialization)
        for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        w: the model parameters as a numpy array of shape (D, )
        loss: the loss value (scalar) for the model parameters
    """
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_gradient(y, tx, w)
        w = w - gamma * gradient

    loss = mse_loss(y, tx, w)
    print("GD: W = " + str(w) + ",\n training loss = " + str(loss) + "\n")
    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent on MSE

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        initial_w: numpy array of shape=(D, ). The initial guess (or the initialization)
        for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        w: the model parameters as a numpy array of shape (D, )
        loss: the loss value (scalar) for the model parameters
    """
    w = initial_w
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx):
            gradient = compute_gradient(minibatch_y, minibatch_tx, w)
            w = w - gamma * gradient

    loss = mse_loss(y, tx, w)
    print("SGD: W = " + str(w) + ",\n training loss = " + str(loss) + "\n")
    return w, loss


def least_squares(y, tx):
    """Calculate the least squares solution.
       returns optimal weights and mse loss

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.
    """
    w = np.linalg.lstsq(tx, y, rcond=None)[0]
    #     w = np.linalg.solve(tx, y)
    loss = mse_loss(y, tx, w)
    print("LS: W = " + str(w) + ",\n training loss = " + str(loss) + "\n")
    return w, loss


def ridge_regression(y, tx, lambda_):
    """ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: mse loss for the weights
    """
    N, D = tx.shape
    lp = 2 * N * lambda_
    a = tx.T @ tx
    b = lp * np.identity(D)
    inv = np.linalg.inv(a + b)
    w = inv @ tx.T @ y
    loss = mse_loss(y, tx, w)
    print("RR: W = " + str(w) + ",\n training loss = " + str(loss) + "\n")
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma, im=1):
    """Logistic regression using GD or SGD (y = 0|1)"""
    """
    Args:
        y (np.array): Labels of shape (N, ).
        tx (np.array): Dataset of shape (N, D).
        initial_w: numpy array of shape=(D, ). The initial guess (or the initialization) 
        max_iters (integer): Maximum number of iterations.
        gamma (int): The learning rate of  the gradient step.
    Returns:
        np.array: weights of shape(D, )
    """
    w = initial_w
    for it in range(max_iters):
        gradient = gradient_logistic(tx, y, w, 0, im=im)
        w = w - gamma * gradient
    loss = loss_logistic(tx, y, w, 0, im=im)
    print("LogReg: W = " + str(w) + ",\n training loss = " + str(loss) + "\n")
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma, im=1):
    """Regularized logistic regression using GD or SGD (y = 0|1), with regularization term lambda||w||^2"""
    """
    Args:
        y (np.array): Labels of shape (N, ).
        tx (np.array): Dataset of shape (N, D).
        lambda_  (int): parameter controls the amount of regularization
        initial_w: numpy array of shape=(D, ). The initial guess (or the initialization) 
        max_iters (integer): Maximum number of iterations.
        gamma (int): The learning rate of  the gradient step.
        im (int or np.array): weights for unbalanced data
    Returns:
        np.array: weights of shape(D, )
    """
    weights = initial_w
    for it in range(max_iters):
        gradient = gradient_logistic(tx, y, weights, lambda_, im=im)
        weights = weights - gamma * gradient

    return weights, loss_logistic(tx, y, weights, 0, im=im)
