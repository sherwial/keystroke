from mapper import mono_map, di_map
from utils import neuralnet_mapper_link
from neuralnet import MonographNetwork, DigraphNetwork
import pickle
import numpy as np
from matplotlib import pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
import multiprocessing
import shutil
import json
import time
import argparse
import csv
from descision import DecisionModule

def train(profile):
        # CREATE THE MAP OBJECTS
        monograph_map = mono_map(profile['data_mono'])  # Get the map object for monographs
        digraph_map = di_map(profile['data_di'])  # Get the map object for digraphs

        # BUILD AND GET KEY ORDER TABLES
        monograph_map.build_mko_map()
        digraph_map.build_dko_map()
        current_mono_map = monograph_map.get_mko_map()
        current_di_map = digraph_map.get_dko_map()

        # print "     Obtaining training data..."
        # OBTAIN THE TRAINING DATA FROM THE MAPS
        training_inputs_mono, training_outputs_mono = \
            neuralnet_mapper_link.to_training_data_mono(profile['data_mono'], current_mono_map)
        training_inputs_di, training_outputs_di, normalizer_di= \
            neuralnet_mapper_link.to_training_data_di(profile['data_di'], current_di_map)

        # CREATE THE NETWORKS
        mono_net = MonographNetwork()
        di_net = DigraphNetwork()

        # TRAIN THE NETWORKS
        mono_net.train(training_inputs_mono, training_outputs_mono, 200, int(len(training_inputs_mono) / 3))
        di_net.train(training_inputs_di, training_outputs_di, 500, int(len(training_inputs_di) / 5))

        # SAVE THE NETWORKS
        mono_net.save_weights('savedweights/'+profile['filename'].split('.')[0]+"_mono.h5")
        di_net.save_weights('savedweights/'+profile['filename'].split('.')[0]+"_di.h5")

        # PLACE UPDATED PROFILE IN QUEUE
        profile['mono_map'] = monograph_map
        profile['di_map'] = digraph_map
        profile['norm_di'] = normalizer_di

        # GUESS THE OUTPUTS FROM NETWORKS
        outputs_mono = mono_net.guess(training_inputs_mono)
        outputs_di = di_net.guess(training_inputs_di)

        # PLOT EVERYTHING
        fig = plt.figure()
        mono_data_plt = fig.add_subplot(2, 2, 1)
        mono_data_plt.scatter(training_inputs_mono, training_outputs_mono, s=2)
        plt.ylabel("Dwell Time (ms)")
        mono_output_plt = fig.add_subplot(2, 2, 2)
        mono_output_plt.scatter(training_inputs_mono, outputs_mono, s=2)
        plt.ylabel("Approx. Dwell Time (ms)")
        xAxis = []
        yAxis = []
        for item in training_inputs_di: xAxis.append(item[0])
        for item in training_inputs_di: yAxis.append(item[1])
        di_output_plt = fig.add_subplot(224, projection="3d")
        di_output_plt.scatter(xAxis, yAxis, outputs_di, s=3)
        di_output_plt.set_zlabel("Approx. Fly Time (ms)")
        xAxis = []
        yAxis = []
        zAxis = []
        for digraph in profile['data_di'].keys():
            try:
                ko1 = current_di_map[digraph[0]]
                ko2 = current_di_map[digraph[1]]
            except:
                pass

            for fly_time in profile['data_di'][digraph]:
                xAxis.append(ko1)
                yAxis.append(ko2)
                zAxis.append(fly_time)
        di_data_plt = fig.add_subplot(223, projection="3d")
        di_data_plt.scatter(xAxis, yAxis, zAxis, s=3)
        di_data_plt.set_zlabel("Fly Time (ms)")
        plt.tight_layout()
        fig.savefig("../" + profile['filename'].split('.')[0] + ".png")
        plt.close(fig)
        plt.close('all')

        return profile

class TestAlg():

    def __init__(self, results_filename):
        start = time.time()
        self.MONOGRAPH_PATH = "EnrollmentDataAdjusted/MonographPickleFiles/"
        self.DIGRAPH_PATH = "EnrollmentDataAdjusted/DigraphPickleFiles/"
        try:shutil.rmtree('savedweights')
        except:pass
        os.mkdir('savedweights')
        count = 0
        # Initializes the test profiles
        self.profiles = {} # Stores each profile with its necessary info (map, weight filenames, data)
        print "Assembling data..."
        for root, dir, files in os.walk(self.MONOGRAPH_PATH):
            for filename in files:
                key = filename.split('.')[0]
                data_mono = pickle.load(open(self.MONOGRAPH_PATH+filename, 'rb'))
                data_di = pickle.load(open(self.DIGRAPH_PATH+filename, 'rb'))
                self.profiles[key] = {"filename":filename,"data_mono":data_mono,"data_di":data_di}
                # count += 1
                # if count >= 3:
                #     break
            break

        # MAP DATA and TRAIN THE NETWORKS
        print "Mapping data and training networks..."
        values = [self.profiles[key] for key in self.profiles.keys()]
        pool = multiprocessing.Pool(7)
        newVals = pool.map(train, values)
        self.update_profiles(newVals)
        matching_start = time.time()
        print "     training time: " + str(time.time() - start)

        # MATCH PROFILES
        print "Matching profiles..."
        d = DecisionModule(self.profiles)
        results_array = d.start()

        def normalize_results(dictionary):
            name = dictionary["name"]
            index = dictionary[name]

            for key in dictionary.keys():
                if key!="name":dictionary[key] /= index
            return dictionary
        results_array = map(normalize_results, results_array)

        # WRITE TO CSV
        fieldnames = [key for key in results_array[0].keys() if key != "name"]
        fieldnames = ["name"] + fieldnames
        csvfile = open(results_filename+".csv", 'wb')
        writer = csv.DictWriter(csvfile, fieldnames)
        writer.writeheader()
        for row in results_array:
            writer.writerow(row)
        print "     matching time: " + str(time.time() - matching_start)
        print "Done"

    def update_profiles(self, values):
        """
        Updates the dictionary of profiles
        using an array of updated profiles
        obtained through the neuralnet testing
        """
        for item in values:
            key = item['filename'].split('.')[0]
            self.profiles[key] = item

if __name__ == '__main__':
     parser = argparse.ArgumentParser()
     parser.add_argument('results_name')
     args = parser.parse_args()
     t = TestAlg(args.results_name)