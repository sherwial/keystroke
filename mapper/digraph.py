class DiscreteDigraph:
    """
    Represents all instances of a specific digraph in an enrollment sample
    Has the ability to store Key Order information
    """
    def __init__(self):
        self.average_fly = 0.0
        self.total_graphs = 0

    def add(self, fly):
        # Adds another instance of the digraph
        total_fly = float(self.average_fly) * float(self.total_graphs)

        self.total_graphs += 1
        self.average_fly = float((total_fly + fly)) / float(self.total_graphs)


    def get_count(self):
        return self.total_graphs

    def get_avg_fly(self):
        return self.average_fly