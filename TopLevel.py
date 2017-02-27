from neuralnet import MonographNetwork, DigraphNetwork
import os
import time
import multiprocessing
from pathos.multiprocessing import ProcessingPool as Pool
import shutil
from mapper import mono_map, di_map
from utils import neuralnet_mapper_link
import numpy
import itertools
import gc
import cPickle as pickle
import threading
import sys
import myglobaldata

MONOGRAPH_PATH_RAW = "KeystrokeData/monograph/"
DIGRAPH_PATH_RAW = "KeystrokeData/digraph/"
MONOGRAPH_PATH = "KeystrokeData/MonographPickleFiles/"
DIGRAPH_PATH = "KeystrokeData/DigraphPickleFiles/"

def train(enrollment_data):
    # CREATE THE MAP OBJECTS
    monograph_map = mono_map(enrollment_data['mono'])  # Get the map object for monographs
    digraph_map = di_map(enrollment_data['di'])  # Get the map object for digraphs

    # BUILD AND GET KEY ORDER TABLES
    monograph_map.build_mko_map()
    digraph_map.build_dko_map()
    current_mono_map = monograph_map.get_mko_map()
    current_di_map = digraph_map.get_dko_map()

    # print "     Obtaining training data..."
    # OBTAIN THE TRAINING DATA FROM THE MAPS
    training_inputs_mono, training_outputs_mono = \
        neuralnet_mapper_link.to_training_data_mono(enrollment_data['mono'], current_mono_map)
    training_inputs_di, training_outputs_di, normalizer_di = \
        neuralnet_mapper_link.to_training_data_di(enrollment_data['di'], current_di_map)

    # CREATE THE NETWORKS
    mono_net = MonographNetwork()
    di_net = DigraphNetwork()

    # TRAIN THE NETWORKS
    mono_net.train(training_inputs_mono, training_outputs_mono, 200, int(len(training_inputs_mono) / 3))
    di_net.train(training_inputs_di, training_outputs_di, 500, int(len(training_inputs_di) / 5))

    # SAVE THE NETWORKS
    mono_net.save_weights('savedweights/' + enrollment_data['name'].split('.')[0] + "_mono.h5")
    di_net.save_weights('savedweights/' + enrollment_data['name'].split('.')[0] + "_di.h5")

    # Return profile

    monograph_map.build_mko_map()
    monograph_map.build_dko_map()
    profile = {}
    profile['mono_map'] = monograph_map.get_mko_map()
    profile['di_map'] = digraph_map.get_dko_map()
    profile['norm_di'] = normalizer_di
    return profile

def generate_cross_validation_data_from_users(username_array):
    """
    Obtains and splits data from users into chunks as specified by Ahmed and Traore

    [(1500),(500),(500)...x40...,(500),(500)]

    """
    data_dict = {}
    for name in username_array:
        with open(MONOGRAPH_PATH_RAW + name + ".txt") as monofile:
            data = [[item.split(':')[0], float(item.split(':')[1])] for item in monofile.read().split('\t')]
            if len(data) < 21500:
                data_dict[name] = None
                continue
            if len(data) < 21500:
                print "Bad code"
            numpy.random.shuffle(data)
            enrollment_sample = []
            for i in range(0, 1500):
                enrollment_sample.append(data[i])
            validation_samples = []
            count = 0
            newsample = []
            for i in range(1500, len(data)):
                if count < 500:
                    newsample.append(data[i])
                    count += 1
                else:
                    count = 0
                    validation_samples.append(newsample)
                    newsample = []
            monograph = [enrollment_sample] + validation_samples
        with open(DIGRAPH_PATH_RAW + name +".txt") as digraphfile:
            data = [[item.split(':')[0], item.split(':')[1], float(item.split(':')[2])] for item in digraphfile.read().split('\t')]
            if len(data) < 21500:
                data_dict[name] = None
                continue
            if len(data) < 21500:
                print "Bad code"
            numpy.random.shuffle(data)
            enrollment_sample = []
            for i in range(0, 1500):
                enrollment_sample.append(data[i])
            validation_samples = []
            count = 0
            newsample = []
            for i in range(1500, len(data)):
                if count < 500:
                    newsample.append(data[i])
                    count += 1
                else:
                    count = 0
                    validation_samples.append(newsample)
                    newsample = []
            digraph = [enrollment_sample] + validation_samples
        data_dict[name] = [monograph, digraph]
    return data_dict


def extract_enrollment_samples(cross_validation_profile):
    """
    Turn the first sample (1500 graphs) into a format the neural network training function
    can process
    """
    enrollment_samples = {}

    monographs = {}
    for monograph in cross_validation_profile[0][0]:
        if monograph[0] not in monographs.keys():
            monographs[monograph[0]] = []
        monographs[monograph[0]].append(monograph[1])
    digraphs = {}
    for digraph in cross_validation_profile[1][0]:
        graph = str(digraph[0])+str(digraph[1])
        if graph not in digraphs.keys():
            digraphs[graph] = []
        digraphs[graph].append(digraph[2])

    enrollment_samples["mono"] = monographs
    enrollment_samples["di"] = digraphs
    return enrollment_samples

class CrossEvaluationAlg(object):
    def __init__(self, username_array, data_array, profiles):
        # Build username array

        self.username_array = [file.split('.')[0] for root, dir, files in os.walk(MONOGRAPH_PATH_RAW) for file in files]

        # Build data from pickle files
        self.data_array = generate_cross_validation_data_from_users(self.username_array)
        for key in self.data_array.keys():
            if self.data_array[key] is None:
                print "     " + key
                self.data_array.pop(key)
                self.username_array.remove(key)
        with open("data.pickle", "wb") as f:pickle.dump(self.data_array,f)

        # Build profiles from enrollment samples (in data array)
        try:shutil.rmtree('savedweights')
        except:pass
        os.mkdir('savedweights')
        p = multiprocessing.Pool(4)
        enrollment_samples_users = p.map(extract_enrollment_samples, [self.data_array[key] for key in self.data_array.keys()])
        for index, val in enumerate(enrollment_samples_users):
            enrollment_samples_users[index]["name"] = (self.username_array[index])
        start = time.time()
        print "Mapping data and training networks... " + str(start)
        pool = multiprocessing.Pool(7)
        self.profiles = pool.map(train, enrollment_samples_users)
        with open("profiles.pickle", "wb") as f:pickle.dump(self.profiles, f)
        print "Ending time: " + str(time.time() - start)

        with open("usernames.pickle", "wb") as f: pickle.dump(self.username_array, f)

def cross_evaluate(tuple_names):
    name = tuple_names[0]
    attacker_name = tuple_names[1]
    print name + " attacking " + attacker_name
    profile = myglobaldata.profiles[myglobaldata.username_array.index(name)]
    m_net = myglobaldata.network_dict[name][0]
    d_net = myglobaldata.network_dict[name][1]
    monograph_map = profile["mono_map"].get_mko_map()
    digraph_map = profile["di_map"].get_dko_map()
    attacking_data_mono = myglobaldata.data_array[attacker_name][0]
    attacker_data_di = myglobaldata.data_array[attacker_name][1]

    trials_array = []
    for i in range(1, 40):
        curr_attck_data_m = attacking_data_mono[i]
        curr_attck_data_d = attacker_data_di[i]

        def generate_difference_mono(graph):
            try:
                ko = monograph_map[graph[0]]
            except:
                return 0
            approx = m_net.guess(numpy.array([ko]))[0][0]
            return abs((graph[1] - approx) * 100 / approx)

        sum_array = map(generate_difference_mono, [graph for graph in curr_attck_data_m])
        total_count = float(len(sum_array))
        summation = sum(sum_array)
        mono_deviation = summation / total_count

        def generate_difference_di(graph):
            try:
                ko1 = digraph_map[graph[0]]
                ko2 = digraph_map[graph[1]]
            except:
                return 0
            approx = profile['norm_di'].inverse_normalize(d_net.guess(numpy.array([[ko1, ko2]]))[0][0])
            return abs((graph[2] - approx) * 100 / approx)

        sum_array = map(generate_difference_di, [graph for graph in curr_attck_data_d])
        summation = sum(sum_array)
        total_count = float(len(sum_array))
        di_deviation = summation / total_count

        beta = 0.5
        current_result = beta*mono_deviation + (1-beta)*di_deviation
        trials_array.append(current_result)

    return trials_array



class BuildResults(object):
    def __init__(self):
        self.p = multiprocessing.Pool(4)

    def run(self):
        return self.p.map(cross_evaluate, myglobaldata.mapping_scheme)

    # def cross_evaluate(self, tuple_names):
    #     name = tuple_names[0]
    #     attacker_name = tuple_names[1]
    #     print name + " attacking " + attacker_name
    #     print myglobaldata.username_array
    #     profile = myglobaldata.profiles[myglobaldata.username_array.index(name)]
    #     m_net = myglobaldata.network_dict[name][0]
    #     d_net = myglobaldata.network_dict[name][1]
    #
    #     monograph_map = profile["mono_map"].get_mko_map()
    #     digraph_map = profile["di_map"].get_dko_map()
    #     attacking_data_mono = myglobaldata.data_array[attacker_name][0]
    #     attacker_data_di = myglobaldata.data_array[attacker_name][1]
    #
    #     trials_array = []
    #     for i in range(1, 40):
    #         curr_attck_data_m = attacking_data_mono[i]
    #         curr_attck_data_d = attacker_data_di[i]
    #
    #         def generate_difference_mono(graph):
    #             try:
    #                 ko = monograph_map[graph[0]]
    #             except:
    #                 return 0
    #             approx = m_net.guess(numpy.array([ko]))[0][0]
    #             return abs((graph[1] - approx) * 100 / approx)
    #
    #         sum_array = map(generate_difference_mono, [graph for graph in curr_attck_data_m])
    #         total_count = float(len(sum_array))
    #         summation = sum(sum_array)
    #         mono_deviation = summation / total_count
    #
    #         def generate_difference_di(graph):
    #             try:
    #                 ko1 = digraph_map[graph[0]]
    #                 ko2 = digraph_map[graph[1]]
    #             except:
    #                 return 0
    #             approx = profile['norm_di'].inverse_normalize(d_net.guess(numpy.array([[ko1, ko2]]))[0][0])
    #             return abs((graph[2] - approx) * 100 / approx)
    #
    #         sum_array = map(generate_difference_di, [graph for graph in curr_attck_data_d])
    #         summation = sum(sum_array)
    #         total_count = float(len(sum_array))
    #         di_deviation = summation / total_count
    #
    #         beta = 0.5
    #         current_result = beta*mono_deviation + (1-beta)*di_deviation
    #         trials_array.append(current_result)
    #
    #         gc.collect()
    #     return trials_array

if __name__ == '__main__':
    # c = CrossEvaluationAlg(None, None, None)
    b = BuildResults()
    results = b.run()
