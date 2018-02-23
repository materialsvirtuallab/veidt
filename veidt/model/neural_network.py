from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from ..abstract import Model


class MultiLayerPerceptron(Model):
    """
    Basic neural network model.

    :param layer_sizes: Hidden layer sizes, e.g., [3, 3]
    :param describer: Describer object to convert input objects to
        descriptors.
    :param preprocessor: : Processor to use. Defaults to StandardScaler
    :param activation: Activation function
    :param loss: Loss function. Defaults to mae
    """

    def __init__(self, layer_sizes, describer, preprocessor=None,
                 activation="relu", loss="mse"):
        self.layer_sizes = layer_sizes
        self.describer = describer
        self.output_describer = None
        self.preprocessor = preprocessor
        self.activation = activation
        self.loss = loss
        self.model = None

    def fit(self, inputs, outputs, test_size=0.2, adam_lr=1e-2, **kwargs):
        """
        :param inputs: List of inputs
        :param outputs: List of outputs
        :param test_size: Size of test set. Defaults to 0.2.
        :param adam_lr: learning rate of Adam optimizer
        :param kwargs: Passthrough to fit function in keras.models
        """
        from keras.optimizers import Adam
        from keras.models import Sequential
        from keras.layers import Dense
        descriptors = self.describer.transform(inputs)
        if self.preprocessor is None:
            self.preprocessor = StandardScaler()
            scaled_descriptors = self.preprocessor.fit_transform(descriptors)
        else:
            scaled_descriptors = self.preprocessor.transform(descriptors)
        adam = Adam(adam_lr)
        x_train, x_test, y_train, y_test = train_test_split(
            scaled_descriptors, outputs, test_size=test_size)

        model = Sequential()
        model.add(Dense(self.layer_sizes[0], input_dim=len(x_train[0]),
                        activation=self.activation))
        for l in self.layer_sizes[1:]:
            model.add(Dense(l, activation=self.activation))
        model.add(Dense(1))
        model.compile(loss=self.loss, optimizer=adam, metrics=[self.loss])
        model.fit(x_train, y_train, verbose=0, validation_data=(x_test, y_test),
                  **kwargs)
        self.model = model

    def predict(self, inputs):
        descriptors = self.describer.transform(inputs)
        scaled_descriptors = self.preprocessor.transform(descriptors)
        return self.model.predict(scaled_descriptors)

    def save(self, model_fname, scaler_fname):
        """
        use kears model.save method to save model in .h5
        use scklearn.external.joblib to save scaler(the .save
        file is supposed to be much smaller than saved as
        pickle file)
        :param model_fname:
        :param scaler_fname:
        :return:None
        """
        self.model.save(model_fname)
        joblib.dump(self.preprocessor, scaler_fname)

    def load(self, model_fname, scaler_fname):
        from keras.models import load_model
        self.model = load_model(model_fname)
        self.preprocessor = joblib.load(scaler_fname)
