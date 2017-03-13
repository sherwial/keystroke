import pickle
import itertools
from neuralnet import MonographNetwork, DigraphNetwork

try:
        with open("profiles.pickle", "rb") as f: profiles = pickle.load(f)
        with open("data.pickle", "rb") as f: data_array = pickle.load(f)
        with open("usernames.pickle", "rb") as f: username_array = pickle.load(f)


        net_dict = {}

        net_dict = {}
        for name in username_array:
                m_net = MonographNetwork()
                d_net = DigraphNetwork()
                m_net.load_weights("savedweights/" + name + "_mono.h5")
                d_net.load_weights("savedweights/" + name + "_di.h5")
                net_dict[name] = [m_net, d_net]

        network_dict = net_dict
        mapping_scheme = list(itertools.product(username_array, username_array))

except:
        pass