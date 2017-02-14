import numpy as np
from normalize import MinMaxNormalize
def to_training_data_mono(mono_dict, map):
    """
    :param filtered_list_of_graphs: outliers removed and only characters
    :param map:
    :return:
    """
    training_in = []
    training_out = []
    for monograph in mono_dict:
        try:
            ko = map[monograph]
            for dwell_time in mono_dict[monograph]:
                # if dwell_time < 700:
                    training_in.append(ko)
                    training_out.append(dwell_time)
        except:
            pass
            # print "Failed with " + str(monograph)
    return np.array(training_in), np.array(training_out)


def to_training_data_di(digraph_dict, map):
    """
    :param filtered_list_of_graphs: outliers removed and only characters
    :param map:
    :return:
    """

    array = []
    for item in digraph_dict:
        for number in digraph_dict[item]:
            array.append(number)


    normalizer = MinMaxNormalize(array)
    training_in = []
    training_out = []
    for digraph in digraph_dict:
        try:
            ko = [map[digraph[0]], map[digraph[1]]]
            for fly_time in digraph_dict[digraph]:
                training_in.append(ko)
                training_out.append(normalizer.normalize(fly_time))

        except:
            pass
            # print "            Failed with " + str(digraph[1]) + " " + str(digraph[1])
        #     raise Exception()

    return np.array(training_in), np.array(training_out), normalizer

