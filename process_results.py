import cPickle as pickle
import itertools
from matplotlib import pyplot as plt
import json
import numpy

with open("results.pickle", "rb") as f: results = pickle.load(f)
with open("usernames.pickle", "rb") as f: usernames = pickle.load(f)

validation_results = {}
threshold = range(0, 100)

mapping_scheme = itertools.product(usernames, usernames)

for i, tuple in enumerate(mapping_scheme):
    if tuple[0] not in validation_results.keys():
        validation_results[tuple[0]] = {}
    validation_results[tuple[0]][tuple[1]] = results[i]

processed_results = []
threshold = range(0, 100)
for thresh in threshold:
    false_acceptance_count = 0
    acceptance_total = 0
    false_rejection_count = 0
    rejection_total = 0
    for user in validation_results.keys():
        for attacker in validation_results[user].keys():
            if attacker == user:
                for result in validation_results[user][attacker]:
                    if result > thresh:
                        false_rejection_count += 1
                    rejection_total += 1
            else:
                for result in validation_results[user][attacker]:
                    if result <= thresh:
                        false_acceptance_count += 1
                    acceptance_total += 1
    processed_results.append([thresh,
                              float(false_rejection_count) / rejection_total,
                              float(false_acceptance_count)/ acceptance_total])

x, y1, y2 = zip(*processed_results)
fig1 = plt.figure()
sub1 = fig1.add_subplot(1,2,1)
sub1.plot(x, y1, 'b',markersize=3,label='FRR')
sub1.plot(x, y2, 'r', markersize=3,label='FAR')
sub1.legend(loc="upper right")
sub2 = fig1.add_subplot(1,2,2)
sub2.plot(y1, y2)
sub2.set_xlabel('FRR')
sub2.set_ylabel('FAR')



plt.tight_layout()
plt.savefig("figure.png")
plt.close("all")

for user in validation_results.keys():
    average = float(sum(validation_results[user][user])) / len(validation_results[user][user])

    for attacker in validation_results[user]:
        for i,item in enumerate(validation_results[user][attacker]):
            validation_results[user][attacker][i] = item / float(average)

processed_results_normalized = []
threshold = [float(i)/1000 for i in range(0, 2000)]
for thresh in threshold:
    false_acceptance_count = 0
    acceptance_total = 0
    false_rejection_count = 0
    rejection_total = 0
    for user in validation_results.keys():
        for attacker in validation_results[user].keys():
            if attacker == user:
                for result in validation_results[user][attacker]:
                    if result > thresh:
                        false_rejection_count += 1
                    rejection_total += 1
            else:
                for result in validation_results[user][attacker]:
                    if result <= thresh:
                        false_acceptance_count += 1
                    acceptance_total += 1
    processed_results_normalized.append([thresh,
                              float(false_rejection_count) / rejection_total,
                              float(false_acceptance_count)/acceptance_total])

# print json.dumps(processed_results, sort_keys=False, indent=4, separators=(',', ': '))


x, y1, y2 = zip(*processed_results_normalized)
fig1 = plt.figure()
sub1 = fig1.add_subplot(1,2,1)
sub1.plot(x, y1, 'b',markersize=3,label='FRR')
sub1.plot(x, y2, 'r', markersize=3,label='FAR')
sub1.legend(loc="upper right")

sub2 = fig1.add_subplot(1,2,2)
sub2.plot(y1, y2)
sub2.set_xlabel('FRR')
sub2.set_ylabel('FAR')
# sub2.yaxis.set_ticks(numpy.arange(0, 1, .1))
# sub2.xaxis.set_ticks(numpy.arange(0,0.04, .008))

plt.tight_layout()
plt.savefig("figure2.png")
plt.close("all")









