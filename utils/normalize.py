import numpy as np

class MinMaxNormalize:
    def __init__(self, data_arrays):
        self.minimum_point = np.inf
        self.maximum_point = -np.inf

        # Develop minimum and maximum points (bounds) of the range to normalize
        for data_point in data_arrays:
            if data_point < self.minimum_point:
                self.minimum_point = data_point
            elif data_point > self.maximum_point:
                self.maximum_point = data_point


    # Normalize a value
    def normalize(self, point):

        # Make sure point is not outside of range 0-1
        if point > self.maximum_point:
            self.maximum_point = point
        elif point < self.minimum_point:
            self.minimum_point = point

        # implement min-max algorithm
        return ((point - float(self.minimum_point)) / (float(self.maximum_point) - float(self.minimum_point))) * 1 + 0 # implement min-max algorithm

    def inverse_normalize(self, point):

        # Make sure point is not outside of range 0-1
        if point > self.maximum_point:
            self.maximum_point = point
        elif point < self.minimum_point:
            self.minimum_point = point

        # implement inverse normalization
        return point*(self.maximum_point - self.minimum_point) + self.minimum_point