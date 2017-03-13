from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adadelta
import h5py

class DigraphNetwork:
    def __init__(self):
        self.model = Sequential() # Implement sequential model (stacked layers)
        self.model.add(Dense(1, input_dim=2, init='uniform', activation='linear')) # Input layer
        self.model.add(Dense(40, init='uniform', activation='tanh')) # Hidden layer
        # self.model.add(Dense(10, init='uniform', activation='tanh')) # Hidden layer
        self.model.add(Dense(1, init='uniform', activation='linear')) # Output layer
        adadelta = Adadelta(lr=.9, decay=0.001) # Learning rate was 12.  Learned that was too high
        self.model.compile(loss='mse', optimizer=adadelta, metrics=['accuracy'])  # Compile model

    def train(self, x, y, num_epoch=90, b_size=10):
        self.model.fit(x, y, nb_epoch=num_epoch, batch_size=b_size, verbose=0)

    def save_weights(self, filename):
        self.model.save_weights(filename, overwrite=True)

    def load_weights(self, filename):
        self.model.load_weights(filename, by_name=False)

    def guess(self, x):
        return self.model.predict(x)