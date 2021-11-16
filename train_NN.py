## Imports
import pickle

# For plotting
import numpy as np
# For NN
import tensorflow as tf
from scipy import optimize
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam  # , SGD, Adadelta, Adagrad, Nadam


# Set seed for reproducability
tf.random.set_seed(1234)
np.random.seed(1234)


def load_data(fileDir, ReNo, fileExt):
    fileFullDir = fileDir + str(ReNo) + fileExt

    with open(fileFullDir, "rb") as fid:
        data = pickle.load(fid, encoding='bytes')

    return data


class PhysicsInformedNN(object):
    def __init__(self, X, Y, layers, acti="tanh", tf_epochs=10000, lr=0.04):
        """
        The PINN.

        Takes in input (and respective output) sets as X and Y respectively.

        Takes in hyper-parameters including layer structure, activation functions, number of epochs and
        learning rate for Adam optimizer.
        """

        # Set dtype to the highest level of precision. Perhaps a lower precision gives faster solve time?
        tf.keras.backend.set_floatx("float64")

        self.X = X
        self.Y = Y

        ## Init by taking in inputs
        # Separating the collocation coordinates:
        self.x = tf.convert_to_tensor(self.X[:, 0:1])
        self.y = tf.convert_to_tensor(self.X[:, 1:2])
        self.Re = tf.convert_to_tensor(self.X[:, 2:3])

        # Get the input bounds:
        self.ub = self.X.max(0)
        self.lb = self.X.min(0)
        self.lb[2] = 0

        ## Model
        # Using the keras library to build the underlying model:
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.InputLayer(input_shape=(layers[0],)))
        self.model.add(tf.keras.layers.Lambda(
            lambda X: 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0))
        self.initializer = "he_normal"
        if acti == "tanh":
            self.initializer = "glorot_normal"
        for width in layers[1:-1]:
            self.model.add(tf.keras.layers.Dense(
                width, activation=acti,
                kernel_initializer=self.initializer))
        self.model.add(tf.keras.layers.Dense(
            layers[-1], activation=None,
            kernel_initializer=self.initializer))

        # Set keras optimizer
        self.opt = Adam(lr=lr, decay=0.0)

        # Computing the sizes of weights/biases for future decomposition
        self.sizes_w = []
        self.sizes_b = []
        for i, width in enumerate(layers):
            if i != 1:
                self.sizes_w.append(int(width * layers[1]))
                self.sizes_b.append(int(width if i != 0 else layers[1]))

        # Hyper-parameter saving
        self.tf_epochs = tf_epochs

    # The actual PINN
    def ns_net(self):
        # Using the new GradientTape paradigm of TF2.0,
        # which keeps track of operations to get the gradient at runtime
        with tf.GradientTape(persistent=True) as tape:
            # Watching the two inputs we'll need later, x and t
            tape.watch(self.x)
            tape.watch(self.y)
            # Packing together the inputs
            X_tf = tf.stack([self.x[:, 0], self.y[:, 0], self.Re[:, 0]], axis=1)

            # Getting the prediction
            Y = self.model(X_tf)
            p = Y[:, 2:3]

            # Deriving INSIDE the tape (since we'll need the x derivative of this later, u_xx)
            u_x = tape.gradient(Y[:, 0:1], self.x)
            u_y = tape.gradient(Y[:, 0:1], self.y)
            v_x = tape.gradient(Y[:, 1:2], self.x)
            v_y = tape.gradient(Y[:, 1:2], self.y)

        # Finding gradient of p outside the tape to save memory (& hopefully speed-up)
        p_x = tape.gradient(p, self.x)
        p_y = tape.gradient(p, self.y)

        # Getting the other derivatives
        u_xx = tape.gradient(u_x, self.x)
        u_yy = tape.gradient(u_y, self.y)
        v_xx = tape.gradient(v_x, self.x)
        v_yy = tape.gradient(v_y, self.y)

        # Letting the tape go
        del tape

        return Y[:, 0:1], Y[:, 1:2], Y[:, 2:3], u_x, u_y, v_x, v_y, u_xx, u_yy, v_xx, v_yy, p_x, p_y

    @tf.autograph.experimental.do_not_convert
    # Custom loss function
    def loss(self, Y_actual, Y_pred):
        """
        Neural Network custom loss function.

        Takes in two inputs, Y_actual and Y_pred
        Outputs a scalar loss term, which is based on the Navier-Stokes equations.

        MSE0 returns the error between the predicted Y and actual Y.
        MSE1 returns the error term of the conservation of volume equation (flow is assumed incompressible).
        MSE2 and MSE3 represent the horizontal and vertical conservation of momentum equations.
        """

        u_actual = Y_actual[:, 0:1]
        v_actual = Y_actual[:, 1:2]
        p_actual = Y_actual[:, 2:3]

        u_pred = Y_pred[:, 0:1]
        v_pred = Y_pred[:, 1:2]
        p_pred = Y_pred[:, 2:3]

        u, v, p, u_x, u_y, v_x, v_y, u_xx, u_yy, v_xx, v_yy, p_x, p_y = self.ns_net()
        u_t = 0
        v_t = 0

        """
        mse_0 = \
            tf.reduce_mean(tf.square(u_actual - u_pred)) + \
            tf.reduce_mean(tf.square(v_actual - v_pred)) + \
            tf.reduce_mean(tf.square(p_actual - p_pred))
        """

        mse_0 = tf.reduce_mean(tf.square(Y_actual - Y_pred))

        mse_1 = tf.reduce_mean(tf.square(u_x + v_y))

        mse_2 = tf.reduce_mean(tf.square(u_t + (u) * (u_x) + (v) * (u_y) + (p_x) - (1 / self.Re) * (u_xx + u_yy)))

        mse_3 = tf.reduce_mean(tf.square(v_t + (u) * (v_x) + (v) * (v_y) + (p_y) - (1 / self.Re) * (v_xx + v_yy)))

        mse = mse_0 + mse_1 + mse_2 + mse_3

        return mse

    def wrap_training_variables(self):
        var = self.model.trainable_variables
        # var.extend([self.U, self.rho])
        return var

    def fit_pinn(self):
        self.tf_optimization(self.X, self.Y)
        print("TF done")
        self.lbfgs_optimization(self.X, self.Y)
        print("L-BFG-S done")

    def tf_optimization(self, X, Y):
        for epoch in range(self.tf_epochs):
            loss_value = self.tf_optimization_step(X, Y)
            # if epoch % 1000 == 0:
                # tf.print(f"Epoch {epoch}: Loss = {self.loss(Y, self.model(X))}")

    @tf.function
    def tf_optimization_step(self, X, Y):
        loss_value, grads = self.grad(X, Y)
        self.opt.apply_gradients(
            zip(grads, self.wrap_training_variables()))
        return loss_value

    @tf.function
    def grad(self, X, Y):
        with tf.GradientTape() as tape:
            loss_value = self.loss(Y, self.model(X))
        grads = tape.gradient(loss_value, self.wrap_training_variables())
        return loss_value, grads

    def lbfgs_optimization(self, X, Y):
        """
        The lbfgs optimization function makes use of scipy's in-built optimizer to reduce the loss function.
        Scipy's optimizer takes in arguments:
        1. The function to be minimized
            a. For LBFGS, with Jac = True, this is assumed to output (f, g) representing the functional output
               and gradient respectively.
        2. The initial "guess"--this for us is the array of weights.
        Essentially, we wish to minimize the loss with respect to the weights.
        Notably, f, g and x must all be arrays with shape (n,)
        """
        # optimizer options
        options = {'disp': True, 'maxfun': 50000, 'maxiter': 50000, 'maxcor': 50, 'maxls': 50,
                   'ftol': 1.0 * np.finfo(float).eps, 'gtol': 1.0 * np.finfo(float).eps}

        # define x0 as an array with shape (n,)
        x0 = self.flatten_trainable_variables()
        self.results = optimize.minimize(fun=self.val_and_grad, x0=x0, args=(X, Y),
                                               jac=True, method='L-BFGS-B', options=options)

    def flatten_trainable_variables(self):
        wtv = pinn.wrap_training_variables()
        w = np.array([])
        for i in wtv:
            w = np.append(w, i.numpy())
        return w

    def apply_trainable_variables(self, w):
        for i, layer in enumerate(self.model.layers[1:]):
            start_weights = sum(self.sizes_w[:i]) + sum(self.sizes_b[:i])
            end_weights = sum(self.sizes_w[:i + 1]) + sum(self.sizes_b[:i])
            weights = w[start_weights:end_weights]
            w_div = int(self.sizes_w[i] / self.sizes_b[i])
            weights = tf.reshape(weights, [w_div, self.sizes_b[i]])
            biases = w[end_weights:end_weights + self.sizes_b[i]]
            weights_biases = [weights, biases]
            layer.set_weights(weights_biases)

    def val_and_grad(self, w, X, Y):
        self.apply_trainable_variables(w)
        with tf.GradientTape() as tape:
            loss_value = self.loss(Y, self.model(X))
        grads = tape.gradient(loss_value, self.wrap_training_variables())
        k = np.array([])
        for i in range(len(grads)):
            k = np.append(k, tf.reshape(grads[i], [-1]).numpy())
        return loss_value.numpy(), k


if __name__ == "__main__":
    # prepare training data
    ReNos = np.array([1000])  # 200,400,1000,2000,4000,8000
    inputNo = 3
    outputNo = 3
    inputSamples = np.zeros((1, inputNo))
    outputSamples = np.zeros((1, outputNo))

    # Load in data
    for i in range(len(ReNos)):
        data = load_data("./ReFlowdata/cavity_Re", ReNos[i], ".pkl")
        sampleNo = len(data[0])
        # input
        Temp = np.zeros((sampleNo, inputNo))
        for i in range(inputNo):
            for j in range(sampleNo):
                Temp[j, i] = data[i][j]
        inputSamples = np.concatenate((inputSamples, Temp), axis=0)

        # output
        Temp = np.zeros((sampleNo, outputNo))
        for i in range(3, 6):
            for j in range(sampleNo):
                Temp[j, i - 3] = data[i][j]
        outputSamples = np.concatenate((outputSamples, Temp), axis=0)

    inputSamples = np.delete(inputSamples, 0, axis=0)
    outputSamples = np.delete(outputSamples, 0, axis=0)

    # shuffle training data
    N = len(inputSamples)
    I = np.arange(N)
    np.random.shuffle(I)
    inputSamples = inputSamples[I, :]
    outputSamples = outputSamples[I, :]

    sampleNo = inputSamples.shape[0]  #
    oriInput = inputSamples.copy()
    oriOutput = outputSamples.copy()

    origX = inputSamples
    origY = outputSamples

    # Trim data
    noSamples = 128 # This represents the average number of samples per Reynolds number that will be used in training.
    inputSamples = np.delete(origX, np.s_[len(ReNos) * noSamples - 1:-1:1], 0)
    outputSamples = np.delete(origY, np.s_[len(ReNos) * noSamples - 1:-1:1], 0)
    valX = np.delete(origX, np.s_[0:len(ReNos) * noSamples:1], 0)
    valY = np.delete(origY, np.s_[0:len(ReNos) * noSamples:1], 0)

    noLayers = 8
    noNeurons = 50
    acti = 'tanh'
    lr = 0.001

    layers = [inputNo]
    for _ in range(noLayers):
        layers.append(noNeurons)
    layers.append(outputNo)

    # Set save locations
    saveLocationPINN = "./trained_single_Re_new/PINN_Re{0}_{1}_{2}L{3}N_{4}_random.ckpt".format(str(ReNos[0]), str(noSamples), str(noLayers), str(noNeurons), acti)
    saveLocationMLP = "./trained_single_Re_new/MLP_Re{0}_{1}_{2}L{3}N_{4}_random.ckpt".format(str(ReNos[0]), str(noSamples), str(noLayers), str(noNeurons), acti)


    ## PINN initiate, train and save
    pinn = PhysicsInformedNN(inputSamples, outputSamples, layers, lr=lr, acti=acti, tf_epochs=20000)
    pinn.fit_pinn()
    pinn.model.save_weights(filepath=saveLocationPINN, overwrite=True)

    ## Multilayer Perceptron--for comparison
    # Set backend
    tf.random.set_seed(1234)
    tf.keras.backend.set_floatx("float64")
    ub = inputSamples.max(0)
    lb = inputSamples.min(0)
    lb[2] = 0

    mlp = tf.keras.Sequential()
    mlp.add(tf.keras.layers.InputLayer(input_shape=(layers[0],)))
    mlp.add(tf.keras.layers.Lambda(
        lambda X: 2.0 * (X - lb) / (ub - lb) - 1.0))
    initializer = "he_normal"
    if acti == "tanh":
        initializer = "glorot_normal"
    for width in layers[1:-1]:
        mlp.add(tf.keras.layers.Dense(
            width, activation=acti,
            kernel_initializer=initializer))
    mlp.add(tf.keras.layers.Dense(
        layers[-1], activation=None,
        kernel_initializer=initializer))

    opt = Adam(lr=lr, decay=0.0)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, mode='min', verbose=1, patience=100, min_lr=1.0e-8)
    e_stop = EarlyStopping(monitor='val_loss', min_delta=1.0e-8, patience=200, verbose=0, mode='auto')

    mlp.compile(loss='mse', optimizer=opt)
    hist = mlp.fit(inputSamples, outputSamples, validation_split=0.05,
                   epochs=10000, callbacks=[reduce_lr, e_stop], verbose=0, shuffle=True)
    print("Done MLP")
    mlp.save_weights(filepath=saveLocationMLP, overwrite=True)