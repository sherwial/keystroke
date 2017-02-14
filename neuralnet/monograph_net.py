from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adadelta
import h5py

class MonographNetwork:
    def __init__(self):
        self.model = Sequential() # Implement sequential model (stacked layers)

        self.model.add(Dense(1, input_dim=1, init='uniform', activation='linear')) # Input layer
        self.model.add(Dense(16, init='uniform', activation='tanh'))  # Hidden layer
        self.model.add(Dense(20, init='uniform', activation='relu'))  # Hidden layer
        # self.model.add(Dense(3, init='uniform', activation='relu'))  # Hidden layer
        self.model.add(Dense(1, init='uniform', activation='linear')) # Output layer

        adadelta = Adadelta(lr=5, decay=0.001) # Learning rate was 12.  Learned that was too high
        self.model.compile(loss='mse', optimizer=adadelta, metrics=['accuracy'])  # Compile model

    def train(self, x, y, num_epoch=90, b_size=10):
        # Trains the data according to user specified epoch count and batch size
        self.model.fit(x, y, validation_split=0.01, nb_epoch=num_epoch, batch_size=b_size, verbose=0)

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self):
        pass


    def save_weights(self, filename):
        # Saves the weights in a file
        self.model.save_weights(filename, overwrite=True)

    def load_weights(self, filename):
        self.model.load_weights(filename, by_name=False)


    def guess(self, x):
        # Returns a network output with an input x
        return self.model.predict(x)