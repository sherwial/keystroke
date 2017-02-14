from monograph import DiscreteMono
from digraph import DiscreteDigraph


class MonographMap():
    def __init__(self):
        self.dict = {}
        self.mko_map = {}

    def build_mko_map(self):
        self.mko_map = {}
        temp_list = []
        for key in self.dict.keys():
            temp_list.append([self.dict[key].get_dwell(), key])
        sorted_list = sorted(temp_list)
        for i, item in enumerate(sorted_list): self.mko_map[item[1]] = i

    def get_mko_map(self):
        return self.mko_map.copy()

    def add_monograph(self, key, dwell_time):
        if key not in self.dict.keys():
            self.dict[key] = DiscreteMono()
        self.dict[key].add(dwell_time)


class DigraphMap():
    def __init__(self):
        self.dict = {}
        self.dko_map = {}

    def get_empty_graphs(self):
        # TODO
        pass

    def get_digraph(self, graph):
        try:
            return self.dict[graph[1]][graph[0]]
        except:
            return None



    def build_dko_map(self):
        """
        :return: Python dictionary of characters matched to key orders
        """
        self.dko_map = {}
        to_key_list = []
        for to_key in self.dict.keys():
            total_count = 0
            total_time = 0.0
            for from_key in self.dict[to_key].keys():
                total_count = self.dict[to_key][from_key].get_count() + total_count
                total_time = self.dict[to_key][from_key].get_avg_fly() * self.dict[to_key][from_key].get_count() + total_time
            total_avg = float(total_time / total_count)
            to_key_list.append([total_avg, to_key])
        sorted_list = sorted(to_key_list)
        for i, item in enumerate(sorted_list): self.dko_map[item[1]] = i

    def get_dko_map(self):
        # Returns a dko_map
        return self.dko_map.copy()

    def add_digraph(self, frm, to, fly):
        if to in self.dict.keys():
            if frm not in self.dict[to]:
                self.dict[to][frm] = DiscreteDigraph()
        else:
            self.dict[to] = {}
            self.dict[to][frm] = DiscreteDigraph()
        self.dict[to][frm].add(fly)


def mono_map(mono_dict):
    """
    :param mono_array: list of monograph instances.
    :return: A dictionary mapping key order to average dwell time
    """
    m_map = MonographMap()
    for key in mono_dict:
        for dwell_time in mono_dict[key]:
            m_map.add_monograph(key, dwell_time)
    return m_map # Returns a look up table for key orders


def di_map(digraph_dict):
    """
    :param digraph_array: list of digraph instances.
    :return: A DigraphMap object createed with the data
    """
    d_map = DigraphMap()
    for item in digraph_dict:
        for value in digraph_dict[item]:
            d_map.add_digraph(item[0], item[1], value)

    return d_map


def digraph_remap(digraph_map, digraph_network):
    pass
