import pickle
from matplotlib import pyplot as plt

with open("results.pickle", "rb") as f: results = pickle.load(f)
with open("usernames.pickle", "rb") as f: usernames = pickle.load(f)

processed_results = []
threshold = range(0, 100)

print len(results)
print len(usernames)

print results
#
# for thresh in threshold:
#     false_acceptance_count = 0
#     false_rejection_count = 0
#     total_count = 0
#
#
#
#
#     for i, value in enumerate(usernames):
#         for attacking_name in results[i].keys():
#             if attacking_name == value:
#                 if results[i][attacking_name] > thresh:
#                     false_rejection_count += 1
#
#             else:
#                 if results[i][attacking_name] <= thresh:
#                     false_acceptance_count += 1
#             total_count += 1
#
#     processed_results.append([thresh,
#                               float(false_rejection_count) / total_count,
#                               float(false_acceptance_count)/total_count])
#
#
# fig = plt.figure()
#
# ax = fig.add_subplot(1, 1, 1)
# ax.scatter([i[1] for i in processed_results],[i[2] for i in processed_results])
# fig.show()
# plt.savefig("figure.png")
# plt.close("all")
