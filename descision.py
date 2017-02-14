import numpy as np
from neuralnet import MonographNetwork, DigraphNetwork
from pathos.multiprocessing import ProcessingPool as Pool
class DecisionModule(object):
    def __init__(self, profiles):
        self.map = Pool(8).map
        self.profiles = profiles
        self.profiles_array = []

        for profile_name in self.profiles:
            self.profiles_array.append(self.profiles[profile_name])

    def start(self):
        self.result = self.map(self.test_profiles, self.profiles_array)
        return self.result

    def test_profiles(self, user_profile):
        # Load the neuralnets for the user
        dinet = DigraphNetwork()
        dinet.load_weights('savedweights/'+user_profile["filename"].split('.')[0]+"_di.h5")
        mononet = MonographNetwork()
        mononet.load_weights('savedweights/'+user_profile["filename"].split('.')[0]+'_mono.h5')

        current_mono_map = user_profile['mono_map'].get_mko_map()
        current_di_map = user_profile['di_map'].get_dko_map()

        results_dict = {}
        results_dict['name'] = user_profile['filename'].split('.')[0]

        for profile in self.profiles.values():

            summation = 0.0
            total_count = float(0.0)
            for graph in profile['data_mono']:
                try:
                    ko = current_mono_map[graph]
                except:
                    continue
                for dwell_time in profile['data_mono'][graph]:
                    approx = mononet.guess(np.array([ko]))[0][0]
                    summation = summation + abs((dwell_time - approx) * 100 / approx)
                    total_count += 1

            mono_deviation = summation / total_count

            summation = 0.0
            total_count = float(0.0)
            for graph in profile['data_di']:
                try:
                    ko1 = current_di_map[graph[0]]
                    ko2 = current_di_map[graph[1]]
                except:
                    continue
                for fly_time in profile['data_di'][graph]:
                    approx = user_profile['norm_di'].inverse_normalize(dinet.guess(np.array([[ko1, ko2]]))[0][0])
                    summation = summation + abs((fly_time - approx) * 100 / approx)
                    total_count += 1


            di_deviation =  summation / total_count

            attacking_name = profile['filename'].split('.')[0]
            results_dict[attacking_name] = mono_deviation/2 + di_deviation/2
        return results_dict