class DiscreteMono():
    """
    Represents all occurrences of a single monograph
    Has the ability to store Key Order information
    """
    def __init__(self):
        self.average_dwell = 0
        self.total_graphs = 0

    def add(self, dwell):
        # Adds a monograph instance to the collective
        total_dwell = self.average_dwell * self.total_graphs

        self.total_graphs += 1
        self.average_dwell = (total_dwell + dwell) / float(self.total_graphs)

    def get_dwell(self):
        return self.average_dwell
