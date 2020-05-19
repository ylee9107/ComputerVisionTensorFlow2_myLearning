"""




"""

# ======================== Import the required libraries ========================
import numpy as np
import pandas as pd


# ======================== Define the Sigmoid Activation Function ========================
def sigmoid(x):
    y = 1/ (1 + np.exp(-x))
    return y

# Derivative of the Sigmoid funtion:
def derivated_sigmoid(y):
    return y * (1 - y)


# ======================== Define the  Loss Function ========================
def loss_L2(pred, target):
    # opt. we divide by the batch size: by using " / pred.shape[0]"
    return np.sum(np.square(pred - target)) / pred.shape[0]

# Derivative of the Loss funtion:
def derivated_loss_L2(pred, target):
    return 2 * (pred - target)


# ======================== Define the Binary Cross-Entropy Function ========================
def binary_cross_entropy(pred, target):
    return -np.mean(np.multiply(np.log(pred), target) + np.multiply(np.log(1 - pred), (1 - target)))


# Derivative of the binary_cross_entropy funtion:
def derivated_binary_cross_entropy(pred, target):
    return (pred - target) / (pred * (1 - pred))


# ========================  FullyConnectedLayer class ========================
class FullyConnectedLayer(object):
    """ This builds a Fully Connected NN Layer.
    Args:
        - num_inputs (int): The number of input values or input vector size.
        - layer_size (int): The number of output values or output vector size.
        - activation_func (callable): The activation function for this layer.
    Attributes:
        - W (ndarray): Weight values for each of the input.
        - b (float): The bias value, which is added to the weighted sum.
        - size (int): The size of the layer or number of neurons present.
        - activation_func (callable): The activation function.
        - x (ndarray): Store the last provided input vector for backpropagation.
        - y (ndarray): The stored corresponding output for backpropagation.
        - derivated_activation_func (callable): The stored corresponding derivated act. func. for backpropagation.
        - dL_dW (ndarray): The derivative of the Loss function with respect to the weights W.
        - dL_db (ndarray): The derivative of the Loss function with respect to the bias b.
    """

    def __init__(self, num_inputs, layer_size, activation_func, deriv_activation_func=None):
        super().__init__()

        # Random initialisation of the weight vector and bias:
        self.W = np.random.standard_normal((num_inputs, layer_size))
        self.b = np.random.standard_normal(layer_size)
        self.size = layer_size
        self.activation_func = activation_func
        self.deriv_activation_func = deriv_activation_func
        self.x, self.y = None, None
        self.dL_dW, self.dL_db = None, None

    def forward(self, x):
        """ This builds the Forward pass of the input signal through the neuron.
        Args:
            - x (ndarray): Input Vector with shape (batch_size, num_input)
        Returns:
            - activation (ndarray): The activation value (= y) with shape (batch_size, layer_size)

        """
        z = np.dot(x, self.W) + self.b

        # Store the last provided input & output vector for backpropagation
        self.y = self.activation_func(z)
        self.x = x

        return self.y

    def backward(self, dL_dy):
        """ This builds the Backwards pass for backpropagating the loss.
        It will compute all the derivatices and store them with w.r.t the layer parameters and it will also
        return the computed loss w.r.t its inputs for further propagation.
        Args:
            - dL_dy (ndarray): The loss derivative w.r.t. the layer's output (dL/dy = l'_{k+1}).
        Returns:
            - dL_dx (ndarray): The loss derivative w.r.t. the layer's input (dL/dx).

        """
        # This is the derivative of the function (f'):
        dy_dz = self.deriv_activation_func(self.y)

        # This part follows -> dL/dz = dL/dy * dy/dz = l'_{k+1} * f'
        dL_dz = (dL_dy * dy_dz)

        dz_dw = self.x.T
        dz_dx = self.W.T

        # This part follows-> dz/db = d(W.x + b)/db = 0 + db/db = "ones"-vector
        dz_db = np.ones(dL_dy.shape[0])

        # Compute the derivatives w.r.t the layer's parameters and store them for opt. optimization:
        self.dL_dW = np.dot(dz_dw, dL_dz)
        self.dL_db = np.dot(dz_db, dL_dz)

        # Compute the derivative w.r.t the input, to be passed to the previous layers (their `dL_dy`):
        dL_dx = np.dot(dL_dz, dz_dx)

        return dL_dx

    def optimise(self, epsilon):
        """ This builds the optimisation step for the layer's parameters.
        It also uses the stored derivative values for its computation.
        Args:
            - epsilon (float): Learning Rate for updating.

        """
        self.W -= epsilon * self.dL_dW
        self.b -= epsilon * self.dL_db


# ======================== SimpleNetwork class ========================
class SimpleNetwork(object):
    """ This builds a simple neural network model, that is a simple fully connected NN.
    Args:
        - num_inputs (int): The number of input values or input vector size.
        - num_outputs (int): The output vector size.
        - hidden_layers_sizes (list): A list of sizes for each of the hidden layer that is added to the NN.
        - activation_func (callable): Activation function applied to all the layers.
        - derivated_activation_func (callable): The derivated activation function.
        - loss_func (callable): The Loss func. used to train the network.
        - derivated_loss_func (callable): The derivative of the loss func. that is used for backpropagation.
    Attributes:
        - layers (list): The list of layers that would form this simple network model.
        - loss_func (callable): The loss function to train this network.
        - derivated_loss_func (callable): The derivative of the loss func. that is used for backpropagation.
    """

    def __init__(self, num_inputs, num_outputs, hidden_layers_sizes=(64, 32),
                 activation_func=sigmoid, deriv_activation_func=derivated_sigmoid,
                 loss_func=loss_L2, deriv_loss_func=derivated_loss_L2):
        super().__init__()

        # Build the list of layers for the network:
        layer_sizes = [num_inputs, *hidden_layers_sizes, num_outputs]
        self.layers = [
            FullyConnectedLayer(layer_sizes[i],
                                layer_sizes[i + 1],
                                activation_func,
                                deriv_activation_func) for i in range(len(layer_sizes) - 1)
        ]

        self.loss_func = loss_func
        self.deriv_loss_func = deriv_loss_func

    def forward(self, x):
        """ Forward the input vector X through the layers of the network.
        Args:
            - x (ndarray): Input Vector with shape (batch_size, num_input)
        Returns:
            - activation (ndarray): The activation value (= y) with shape (batch_size, layer_size)

        """
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def predict(self, x):
        """ Compute the output that is corresponding to X, and returns the index of the largest output value.
        Args:
            - x (ndarray): Input vector with shape (1, num_inputs).
        Returns:
            - best_class (int): Predicted class ID.
        """
        estimations = self.forward(x)
        best_class = np.argmax(estimations)
        return best_class

    def backward(self, dL_dy):
        """ This builds the Backwards pass for backpropagating the loss through the layers.
        This will require the .forward() method to be called.
        Args:
            - dL_dy (ndarray): The loss derivative w.r.t. the layer's output (dL/dy = l'_{k+1}).
        Returns:
            - dL_dx (ndarray): The loss derivative w.r.t. the layer's input (dL/dx).

        """
        # Iterates from the output layer to the input one
        for layer in reversed(self.layers):
            dL_dy = layer.backward(dL_dy)
        return dL_dy

    def optimise(self, epsilon):
        """ This builds the optimisation step for the layer's parameters.
        It also uses the stored derivative values for its computation.
        Args:
            - epsilon (float): Learning Rate for updating.

        """
        # Note: the order doesn't matter here
        for layer in self.layers:
            layer.optimise(epsilon)

    def evaluate_accuracy(self, X_val, y_val):
        """ Evaluates the network's accuracy based on the a validation dataset.
        Args:
            - X_val (ndarray): Validation dataset input.
            - y_val (ndarray): The corresponding ground-truth validation dataset.
        Returns:
            - accuracy (float): Network computed Accuracy (= number of correct predictions/dataset size).

        """
        nb_corrects = 0
        for i in range(len(X_val)):
            pred_class = self.predict(X_val[i])
            if (pred_class == y_val[i]):
                nb_corrects += 1

        return nb_corrects / len(X_val)

    def train(self, X_train, y_train, X_val=None, y_val=None,
              batch_size=32, nb_epochs=5, learning_rate=1e-3, print_frequency=20):
        """ This builds the Training method for the SimpleNetwork.
        Where for a given dataset and its ground truth labels, it will evaluate the current network accuracy.
        Args:
            - X_train (ndarray): Input training set.
            - y_train (ndarray): Corresponding ground truth for the training set.
            - X_val (ndarray): Input validation set.
            - y_val (ndarray): Corresponding ground truth for the validation set.
            - batch_size (int): The mini-batch size.
            - nb_epochs (int): The number of training epochs (iteration) over the whole dataset.
            - learning_rate (float): The learning rate to scale the derivatives for updating the weights and bias.
            - print_frequency (int): Frequency to print out the metrics (in epochs).
        Returns:
            - losses (list): The list of training losses for each of the epochs.
            - accuracies (list): The list of validation accuracy computed values for each of the epochs.
        """
        # Define the nb of batches per epoch: rounds the result down to the nearest whole number
        nb_batches_per_epoch = len(X_train) // batch_size

        # Check if there is a validation set and use it.
        do_validaiton = X_val is not None and y_val is not None

        # Define the list of losses and accuracies:
        losses, accuracies = [], []

        # Training: for each of the training epoch do...
        for i in range(nb_epochs):

            # for each batch composing the dataset
            epoch_loss = 0
            for b in range(nb_batches_per_epoch):
                # Get the current batch of the iteration.
                batch_idx_begin = b * batch_size
                batch_idx_end = batch_idx_begin + batch_size

                # Select the data based on the batch index:
                x = X_train[batch_idx_begin:batch_idx_end]
                targets = y_train[batch_idx_begin:batch_idx_end]

                # OPTIMISE on current batch ->
                # Forward pass:
                predictions = y = self.forward(x)

                # Compute the loss:
                L = self.loss_func(predictions, targets)

                # Compute the derivative of the loss:
                dL_dy = self.deriv_loss_func(predictions, targets)

                # Backwards pass: backpropagation.
                self.backward(dL_dy)

                # Update the weights and bias:
                self.optimise(learning_rate)

                # Update the current loss in this epoch:
                epoch_loss += L

            # Compute and update the loss:
            epoch_loss /= nb_batches_per_epoch

            # Update the list: losses list.
            losses.append(epoch_loss)

            # Compute and update the accuracy:
            if do_validaiton:
                accuracy = self.evaluate_accuracy(X_val, y_val)
                # Update the list: accuracies list.
                accuracies.append(accuracy)
            else:
                accuracy = np.NaN

            # Print out the losses and accuracies calc.
            if i % print_frequency == 0 or i == (nb_epochs - 1):
                print("Epoch {:4d}: training loss = {:.6f} | val accuracy = {:.2f}%".format(
                    i, epoch_loss, accuracy * 100)
                )
        return losses, accuracies











