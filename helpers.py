"""Some helper functions for project 1."""
import csv
import numpy as np

np.random.seed(0)


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y == "b")] = -1

    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, "w") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})


def compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: shape=(N, )
        tx: shape=(N,D)
        w: shape=(D, ). The vector of model parameters.

    Returns:
        An array of shape (D, ) (same shape as w), containing the gradient of the MSE loss at w.
    """
    return -(tx.T @ (y - tx @ w)) / y.shape[0]


def mse_loss(y, tx, w):
    """Computes the loss using MSE.

    Args:
        y: shape=(N, ) the labels
        tx: shape=(N,D) the features
        w: shape=(D,). the model parameters

    Returns:
        mse: the value of the loss (a scalar), corresponding to the input parameters w.
    """
    return np.mean((y - tx @ w) ** 2) / 2


def batch_iter(y, tx, batch_size=1, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        np.random.seed(0)
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def imbal(labels):
    im = labels.copy()
    im[np.where(im == 0)] = 2
    return im


def sigmoid(pred):
    """Sigmoid function

    Args:
        pred (np.array): Input data of shape (N, )

    Returns:
        np.array: Probabilites of shape (N, ), where each value is in [0, 1].
    """
    pos = pred >= 0
    neg = pred < 0
    pred[pos] = 1 / (1 + np.exp(-pred[pos]))
    pred[neg] = 1 - 1 / (1 + np.exp(pred[neg]))
    return pred


def loss_logistic(data, labels, w, lamda_, im=1):
    """Logistic regression loss function for binary classes

    Args:
        data (np.array): Dataset of shape (N, D).
        labels (np.array): Labels of shape (N, ).
        w (np.array): Weights of logistic regression model of shape (D, )
        lamda_ (int): parameter controls the amount of regularization
        im (int or np.array): weights for unbalanced data
    Returns:
        int: Loss of logistic regression.
    """

    return (
        np.mean((-np.log(sigmoid(-data @ w)) - labels * data.dot(w)) * im)
        + (lamda_) * np.linalg.norm(w) ** 2
    )


def logistic_regression_classify(data, w):
    """Classification function for binary class logistic regression.

    Args:
        data (np.array): Dataset of shape (N, D).
        w (np.array): Weights of logistic regression model of shape (D, )
    Returns:
        np.array: Label assignments of data of shape (N, )
    """

    predictions = sigmoid(data.dot(w))
    predictions[predictions < 0.5] = 0
    predictions[predictions >= 0.5] = 1
    return predictions


def gradient_logistic(data, labels, w, lamda_, im):
    """Logistic regression gradient function for binary classes

    Args:
        data (np.array): Dataset of shape (N, D).
        labels (np.array): Labels of shape (N, ).
        w (np.array): Weights of logistic regression model of shape (D, )
        lamda_ (int) : parameter controls the amount of regularization
    Returns:
        np. array: Gradient array of shape (D, )
    """
    return (
        data.T.dot((sigmoid(data.dot(w)) - labels) * im) / data.shape[0]
        + (lamda_) * w * 2
    )
